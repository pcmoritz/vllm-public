/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "../dispatch_utils.h"

#include <cuda.h>
#include <math.h>
#include <sstream>

#include "cute/tensor.hpp"
#include "cutlass/conv/convolution.h"
#include "cutlass/array.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass_extensions/epilogue/thread/fused_activations.h"

// FIXME(woosuk)
#define TLLM_THROW(...)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        throw std::runtime_error("ERROR!");                                                                         \
    } while (0)

#define TLLM_CHECK_WITH_INFO(...) ;;\

#include "moe_gemm_kernels.h"

namespace tensorrt_llm {

namespace detail
{
// TODO these are copied from CUTLASS because the cutlass version is missing __device__ decorator
template <class StrideIntT>
CUTLASS_HOST_DEVICE cute::Stride<StrideIntT, cute::Int<1>, cute::Int<0>> make_cute_packed_stride(
    cute::Stride<StrideIntT, cute::Int<1>, cute::Int<0>> s, cute::Shape<int, int, int> shape_MKL)
{
    static_assert(std::is_integral_v<StrideIntT>,
        "Stride must have an integral type so it can be set dynamically. Static strides not supported.");
    auto s_copy = s;
    cute::get<0>(s_copy) = static_cast<StrideIntT>(cute::get<1>(shape_MKL));
    return s_copy;
}

template <class StrideIntT>
CUTLASS_HOST_DEVICE cute::Stride<cute::Int<1>, StrideIntT, cute::Int<0>> make_cute_packed_stride(
    cute::Stride<cute::Int<1>, StrideIntT, cute::Int<0>> s, cute::Shape<int, int, int> shape_MKL)
{
    static_assert(std::is_integral_v<StrideIntT>,
        "Stride must have an integral type so it can be set dynamically. Static strides not supported.");
    auto s_copy = s;
    cute::get<1>(s_copy) = static_cast<StrideIntT>(cute::get<0>(shape_MKL));
    return s_copy;
}

} // namespace detail

__device__ void computeHopperInputStrides(
    HopperGroupedGemmInput layout_info, int gemm_m, int gemm_n, int gemm_k, int out_idx)
{
    layout_info.stride_a[out_idx] = detail::make_cute_packed_stride(
        HopperGroupedGemmInput::StrideA{}, cute::make_shape(gemm_m, gemm_k, cute::Int<1>{}));
    layout_info.stride_b[out_idx] = detail::make_cute_packed_stride(
        HopperGroupedGemmInput::StrideB{}, cute::make_shape(gemm_n, gemm_k, cute::Int<1>{}));
    if (layout_info.stride_c)
    {
        assert(false && "CUTLASS does not support a 1xN bias");
        //        layout_info.stride_c[out_idx] = cute::make_stride(0, cute::Int<1>{}, 0);
        layout_info.stride_c[out_idx] = detail::make_cute_packed_stride(
            HopperGroupedGemmInput::StrideC{}, cute::make_shape(1, gemm_n, cute::Int<1>{}));
    }
    layout_info.stride_d[out_idx] = detail::make_cute_packed_stride(
        HopperGroupedGemmInput::StrideD{}, cute::make_shape(gemm_n, gemm_m, cute::Int<1>{}));
}

template <class T, class WeightType>
__device__ void computeHopperInputPointers(HopperGroupedGemmInput layout_info, int gemm_m, int gemm_n, int gemm_k,
    int num_tokens_before_expert, int expert, T const* in, WeightType const* weights, T const* bias,
    HopperGroupedGemmInput::OutputTypeAdaptor_t<T>* output, int const out_idx)
{
    // The input prior to this contains K elements per token, with `num_tokens_before_expert` tokens
    layout_info.ptr_a[out_idx] = in + num_tokens_before_expert * gemm_k;

    // Each expert's weight matrix is a constant size NxK, with `expert` experts
    layout_info.ptr_b[out_idx] = weights + expert * (gemm_n * gemm_k);

    if (bias)
    {
        // Each expert's bias is a constant size N, with `expert` experts
        layout_info.ptr_c[out_idx] = bias + expert * gemm_n;
    }

    // The output prior to this contains N elements per token, with `num_tokens_before_expert` tokens
    layout_info.ptr_d[out_idx] = output + num_tokens_before_expert * gemm_n;
}

// TODO Some of this setup could be cached
template <class T, class WeightType>
__global__ void computeStridesHopperKernel(int64_t const* total_rows_before_expert, HopperGroupedGemmInput layout_info,
    int gemm_n, int gemm_k, int const num_experts, T const* in, WeightType const* weights, float const* fp8_dequant,
    T const* bias, typename HopperGroupedGemmInput::OutputTypeAdaptor_t<T>* output)
{
    // First, compute the global tid. We only need 1 thread per expert.
    int const expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= num_experts)
    {
        return;
    }

    auto const num_tokens_including_expert = total_rows_before_expert[expert];
    auto const num_tokens_before_expert = expert > 0 ? total_rows_before_expert[expert - 1] : 0;
    auto const num_tokens_to_expert = num_tokens_including_expert - num_tokens_before_expert;
    auto const gemm_m = num_tokens_to_expert;

    layout_info.shape_info.problem_shapes[expert]
        = HopperGroupedGemmInput::ProblemShape::UnderlyingProblemShape(gemm_n, gemm_m, gemm_k);

    if (fp8_dequant)
    {
        layout_info.alpha_scale_ptr_array[expert] = fp8_dequant + expert;
    }

    computeHopperInputStrides(layout_info, gemm_m, gemm_n, gemm_k, expert);

    computeHopperInputPointers(
        layout_info, gemm_m, gemm_n, gemm_k, num_tokens_before_expert, expert, in, weights, bias, output, expert);
}

// ============================== Gated Activation =================================
template <class T, class ActFn>
__global__ void doGatedActivationKernel(
    T* output, const T* gemm_result, const int64_t* num_valid_tokens_ptr, size_t inter_size)
{
    const int tid = threadIdx.x;
    const int token = blockIdx.x;
    if (num_valid_tokens_ptr && token >= *num_valid_tokens_ptr)
    {
        return;
    }

    ActFn fn{};
    output = output + token * inter_size;
    gemm_result = gemm_result + token * inter_size * 2;
    for (int i = tid; i < inter_size; i += blockDim.x)
    {
        T fc1_value = gemm_result[i];
        // BF16 isn't supported, use FP32 for activation function
        float gate_value = gemm_result[i + inter_size];
        T gate_act = fn(gate_value);
        output[i] = fc1_value * gate_act;
    }
}

template <class T>
void doGatedActivation(T* output, const T* gemm_result, const int64_t* num_valid_tokens_ptr, int inter_size,
    int num_tokens, ActivationType activation_type, cudaStream_t stream)
{
    const int blocks = num_tokens;
    const int threads = std::min(inter_size, 1024);

    // TODO Instead of T use a vectored type if performance would benefit
    // TODO For some reason Volta fails on GELU_taylor here with Warp Illegal Instruction.
    auto* fn = activation_type == ActivationType::Swiglu
        ? &doGatedActivationKernel<T, cutlass::epilogue::thread::SiLu<float>>
        : &doGatedActivationKernel<T, cutlass::epilogue::thread::GELU<float>>;
    fn<<<blocks, threads, 0, stream>>>(output, gemm_result, num_valid_tokens_ptr, inter_size);
}

template <typename T>
void run_moe_mlp(
    T* moe_output,
    T* fc1_output,
    T* glu_output,
    const T* input_tokens,
    int64_t* cum_num_tokens_per_expert,
    const T* fc1_expert_weights,
    const T* fc1_expert_biases,
    ActivationType fc1_activation_type,
    const T* fc2_expert_weights,
    const int64_t num_expanded_tokens,
    const int hidden_size,
    const int inter_size,
    const int num_experts,
    cudaStream_t stream)
{
    // FIXME(woosuk): The MoE GEMM runner is created for each call. This is inefficient.
    tensorrt_llm::MoeGemmRunner<T, T> moe_gemm_runner;
    HopperGroupedGemmInput hopper_input;
    // Compute FC1
    if (!tensorrt_llm::isGatedActivation(fc1_activation_type)) {
        moe_gemm_runner.moeGemmBiasAct(
            input_tokens, fc1_expert_weights, nullptr, fc1_expert_biases, fc1_output,
            cum_num_tokens_per_expert, hopper_input, num_expanded_tokens, inter_size, hidden_size, num_experts,
            fc1_activation_type, stream);
    } else {
        const size_t fc1_out_size = inter_size * 2;
        // Run the GEMM with activation function overridden with `Identity`, we do the activation separately
        moe_gemm_runner.moeGemmBiasAct(
            input_tokens, fc1_expert_weights, nullptr, fc1_expert_biases, glu_output,
            cum_num_tokens_per_expert, hopper_input, num_expanded_tokens, fc1_out_size, hidden_size, num_experts,
            ActivationType::Identity, stream);
        doGatedActivation<T>(
            fc1_output, glu_output, nullptr, inter_size, num_expanded_tokens,
            fc1_activation_type, stream);
    }
    // Compute FC2
    moe_gemm_runner.moeGemm(
        fc1_output, fc2_expert_weights, nullptr, moe_output, cum_num_tokens_per_expert,
        hopper_input, num_expanded_tokens, hidden_size, inter_size, num_experts, stream);
}

} // namespace tensorrt_llm

// FIXME(woosuk)
#define LAUNCH_MOE_MLP(scalar_t, nv_t)                                                                    \
    tensorrt_llm::run_moe_mlp<nv_t>(                                                                      \
        (nv_t *) moe_output.data_ptr<scalar_t>(),                                                            \
        (nv_t *) fc1_output.data_ptr<scalar_t>(),                                                            \
        (nv_t *) glu_output.data_ptr<scalar_t>(),                                                            \
        (nv_t *) input_tokens.data_ptr<scalar_t>(),                                                          \
        cum_num_tokens_per_expert.data_ptr<int64_t>(),                                              \
        (nv_t *) fc1_expert_weights.data_ptr<scalar_t>(),                                                    \
        (nv_t *) (fc1_expert_biases.has_value() ? fc1_expert_biases.value().data_ptr<scalar_t>() : nullptr),   \
        fc1_activation_type_enum,                                                                   \
        (nv_t *) fc2_expert_weights.data_ptr<scalar_t>(),                                                    \
        num_expanded_tokens,                                                                        \
        hidden_size,                                                                                \
        inter_size,                                                                                 \
        num_experts,                                                                                \
        stream);

void moe_mlp(
    torch::Tensor& moe_output,                              // [num_tokens * topk, hidden_size]
    torch::Tensor& input_tokens,                            // [num_tokens * topk, hidden_size]
    torch::Tensor& cum_num_tokens_per_expert,               // [num_experts]
    torch::Tensor& fc1_expert_weights,                      // [num_experts, inter_size or 2 * inter_size, hidden_size]
    const c10::optional<torch::Tensor>& fc1_expert_biases,  // [num_experts, inter_size]
    int fc1_activation_type,
    torch::Tensor& fc2_expert_weights)                      // [num_experts, hidden_size, inter_size]
{
    const int64_t num_expanded_tokens = input_tokens.numel() / input_tokens.size(-1);
    const int num_experts = fc2_expert_weights.size(0);
    const int hidden_size = fc2_expert_weights.size(1);
    const int inter_size = fc2_expert_weights.size(2);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input_tokens));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    tensorrt_llm::ActivationType fc1_activation_type_enum = static_cast<tensorrt_llm::ActivationType>(fc1_activation_type);
    torch::Tensor fc1_output = torch::empty({num_expanded_tokens, inter_size}, input_tokens.options());
    const bool is_glu = tensorrt_llm::isGatedActivation(fc1_activation_type_enum);
    const int64_t glu_output_size = is_glu ? num_expanded_tokens * inter_size * 2 : 0;
    torch::Tensor glu_output = torch::empty({glu_output_size}, input_tokens.options());

    auto dtype = input_tokens.dtype();
    if (dtype == at::ScalarType::Float) {
        LAUNCH_MOE_MLP(float, float);
    } else if (dtype == at::ScalarType::Half) {
        LAUNCH_MOE_MLP(at::Half, half);
    } else if (dtype == at::ScalarType::BFloat16) {
        LAUNCH_MOE_MLP(at::BFloat16, __nv_bfloat16);
    } else if (dtype == at::ScalarType::Float8_e4m3fn)
        LAUNCH_MOE_MLP(at::Float8_e4m3fn, __nv_fp8_e4m3);
    } else {
        TORCH_CHECK(false, "Unsupported data type: ", dtype);
    }
}
