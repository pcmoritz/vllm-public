#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "../dispatch_utils.h"

#include <c10/util/BFloat16.h>
#include <c10/cuda/CUDAStream.h>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"

using namespace cute;

namespace vllm {

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;  // <M,N,K> per group
using ElementA = cutlass::float_e4m3_t;                                     // Element type for A matrix operand
using ElementB = cutlass::float_e4m3_t;                                     // Element type for B matrix operand
using ElementC = cutlass::half_t;                                           // Element type for C and D matrix operands

// A matrix configuration
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         LayoutC     = cutlass::layout::ColumnMajor;                   // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

// Core kernel configurations
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm90;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
using TileShape           = Shape<_256,_128,_64>;                           // Threadblock-level tile size
using ClusterShape        = Shape<_2,_2,_1>;                                // Shape of the threadblocks in a cluster
using StageCountType = cutlass::gemm::collective::StageCountAuto;           // Stage count maximized based on the tile size
using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum; // Kernel to launch
using EpilogueSchedule = cutlass::epilogue::PtrArrayNoSmemWarpSpecialized;                     // Epilogue to launch

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC *, AlignmentC,
    ElementC, LayoutC *, AlignmentC,
    EpilogueSchedule
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA *, AlignmentA,
    ElementB, LayoutB *, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

std::vector<typename ProblemShape::UnderlyingProblemShape> MakeProblemSizes(torch::Tensor b, torch::Tensor cum_num_tokens_per_expert) {
  const size_t num_experts = cum_num_tokens_per_expert.size(0);
  const size_t k = b.size(1), n = b.size(2);
  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes(num_experts);
  problem_sizes[0] = {cum_num_tokens_per_expert.data_ptr<int64_t>()[0], n, k};
  for (int i = 1; i < num_experts; ++i) {
    int64_t batch_size = cum_num_tokens_per_expert.data_ptr<int64_t>()[i] - cum_num_tokens_per_expert.data_ptr<int64_t>()[i-1];
    problem_sizes[i] = {batch_size, n, k};
  }
  return problem_sizes;
}

template <typename Gemm>
struct ProblemData {
  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;
  cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;
  cutlass::DeviceAllocation<typename Gemm::ElementA *> ptr_A;
  cutlass::DeviceAllocation<typename Gemm::ElementB *> ptr_B;
  cutlass::DeviceAllocation<typename Gemm::ElementC *> ptr_C;
  cutlass::DeviceAllocation<typename Gemm::GemmKernel::UnderlyingStrideA> stride_A;
  cutlass::DeviceAllocation<typename Gemm::GemmKernel::UnderlyingStrideB> stride_B;
  cutlass::DeviceAllocation<typename Gemm::GemmKernel::UnderlyingStrideC> stride_C;
};

template <typename T>
void CopyDataToDevice(const std::vector<T> &src, cutlass::DeviceAllocation<T> &target) {
  target.reset(src.size());
  target.copy_from_host(src.data());
}

template <typename Gemm>
typename Gemm::Arguments MakeArguments(ProblemData<Gemm>& problem_data,
               torch::Tensor a,
				       torch::Tensor b,
				       torch::Tensor c,
				       torch::Tensor cum_num_tokens_per_expert) {
  problem_data.problem_sizes_host = MakeProblemSizes(b, cum_num_tokens_per_expert);

  // Calculate the number of threadblocks to use and validate the result.
  int64_t num_experts = problem_data.problem_sizes_host.size();

  // Create the host arrays of leading dimension data and pointer data.
  using StrideA = typename Gemm::GemmKernel::UnderlyingStrideA;
  using StrideB = typename Gemm::GemmKernel::UnderlyingStrideB;
  using StrideC = typename Gemm::GemmKernel::UnderlyingStrideC;
  using StrideD = typename Gemm::GemmKernel::UnderlyingStrideD;

  std::vector<int64_t>  offsets_a(num_experts);
  std::vector<int64_t> offsets_b(num_experts);
  std::vector<int64_t> offsets_c(num_experts);
  std::vector<StrideA> stride_a_host;
  std::vector<StrideB> stride_b_host;
  std::vector<StrideC> stride_c_host;
  int64_t elements_a = 0, elements_b = 0, elements_c = 0;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;
  std::vector<ElementA *> ptr_a_host(num_experts);
  std::vector<ElementB *> ptr_b_host(num_experts);
  std::vector<ElementC *> ptr_c_host(num_experts);

  for (int i = 0; i < num_experts; ++i) {
    auto problem = problem_data.problem_sizes_host[i];
    auto M = get<0>(problem);
    auto N = get<1>(problem);
    auto K = get<2>(problem);

    stride_a_host.push_back(cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, Int<1>{})));
    stride_b_host.push_back(cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, Int<1>{})));
    stride_c_host.push_back(cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, Int<1>{})));

    offsets_a[i] = elements_a;
    offsets_b[i] = elements_b;
    offsets_c[i] = elements_c;

    ptr_a_host[i] = (ElementA*)a.data_ptr() + offsets_a[i];
    ptr_b_host[i] = (ElementB*)b.data_ptr() + offsets_b[i];
    ptr_c_host[i] = (ElementC*)c.data_ptr() + offsets_c[i];

    elements_a += M * K;
    elements_b += K * N;
    elements_c += M * N;
  }

  // Copy the problem sizes, pointers and leading dimension data to the device.
  CopyDataToDevice(problem_data.problem_sizes_host, problem_data.problem_sizes);

  CopyDataToDevice(ptr_a_host, problem_data.ptr_A);
  CopyDataToDevice(ptr_b_host, problem_data.ptr_B);
  CopyDataToDevice(ptr_c_host, problem_data.ptr_C);

  CopyDataToDevice(stride_a_host, problem_data.stride_A);
  CopyDataToDevice(stride_b_host, problem_data.stride_B);
  CopyDataToDevice(stride_c_host, problem_data.stride_C);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = b.device().index();
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGrouped,
    {static_cast<int>(num_experts), problem_data.problem_sizes.get(), problem_data.problem_sizes_host.data()},
    {(vllm::ElementA **)problem_data.ptr_A.get(), problem_data.stride_A.get(),
     (vllm::ElementB **)problem_data.ptr_B.get(), problem_data.stride_B.get()},
    {{/*alpha=*/1.0f, /*beta=*/0.0f},
     (vllm::ElementC **)problem_data.ptr_C.get(), problem_data.stride_C.get(),
     (vllm::ElementC **)problem_data.ptr_C.get(), problem_data.stride_C.get()},
    hw_info
  };

  return arguments;
}

void CutlassGroupedGemm(
    torch::Tensor a,
	torch::Tensor b,
	torch::Tensor c,
	torch::Tensor cum_num_tokens_per_expert,
    cudaStream_t stream) {
  Gemm gemm;
  ProblemData<Gemm> problem_data;

  auto arguments = MakeArguments<Gemm>(problem_data, a, b, c, cum_num_tokens_per_expert);
  int64_t workspace_size = gemm.get_workspace_size(arguments);
  auto options = torch::TensorOptions().dtype(torch::kInt8).device(a.device());
  torch::Tensor workspace = torch::empty(workspace_size, options);

  // Check if the problem size is supported or not
  auto status = gemm.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess, cutlass::cutlassGetStatusString(status));

  // Initialize the kernel.
  if(gemm.initialize(arguments, workspace.data_ptr()) != cutlass::Status::kSuccess) {
    TORCH_CHECK(false, "Failed to initialize CUTLASS Grouped GEMM");
  }

  // Execute the kernel.
  if(gemm.run(stream) != cutlass::Status::kSuccess) {
    TORCH_CHECK(false, "Failed to run CUTLASS Grouped GEMM");
  }
}

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
        float gate_value = __bfloat162float(gemm_result[i + inter_size]);
        float gate_act = fn(gate_value);
        output[i] = __float2bfloat16(__bfloat162float(fc1_value) * gate_act);
    }
}

template <class T>
void doGatedActivation(T* output, const T* gemm_result, const int64_t* num_valid_tokens_ptr, int inter_size,
    int num_tokens, cudaStream_t stream)
{
    const int blocks = num_tokens;
    const int threads = std::min(inter_size, 1024);

    // TODO Instead of T use a vectored type if performance would benefit
    // TODO For some reason Volta fails on GELU_taylor here with Warp Illegal Instruction.
    auto* fn = &doGatedActivationKernel<T, cutlass::epilogue::thread::SiLu<float>>;
    fn<<<blocks, threads, 0, stream>>>(output, gemm_result, num_valid_tokens_ptr, inter_size);
}

}  // namespace

void moe_mlp(
    torch::Tensor& moe_output,                              // [num_tokens * topk, hidden_size]
    torch::Tensor& input_tokens,                            // [num_tokens * topk, hidden_size]
    torch::Tensor& cum_num_tokens_per_expert,               // [num_experts]
    torch::Tensor& fc1_expert_weights,                      // [num_experts, 2 * inter_size, hidden_size]
    torch::Tensor& fc2_expert_weights)                      // [num_experts, hidden_size, inter_size]
{
  const int64_t num_expanded_tokens = input_tokens.numel() / input_tokens.size(-1);
  const int num_experts = fc2_expert_weights.size(0);
  const int hidden_size = fc2_expert_weights.size(1);
  const int inter_size = fc2_expert_weights.size(2);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input_tokens));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  torch::Tensor fc1_output = torch::empty({num_expanded_tokens, inter_size}, input_tokens.options());
  torch::Tensor glu_output = torch::empty({num_expanded_tokens * inter_size * 2}, input_tokens.options());

  vllm::CutlassGroupedGemm(input_tokens, fc1_expert_weights, fc1_output, cum_num_tokens_per_expert, stream);

  vllm::doGatedActivation<__nv_bfloat16>(
    (__nv_bfloat16*) fc1_output.data_ptr<at::BFloat16>(),
    (__nv_bfloat16*) glu_output.data_ptr<at::BFloat16>(),
    nullptr, inter_size, num_expanded_tokens, stream);

  vllm::CutlassGroupedGemm(fc1_output, fc2_expert_weights, moe_output, cum_num_tokens_per_expert, stream);
}