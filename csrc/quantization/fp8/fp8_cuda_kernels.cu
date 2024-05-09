#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>

#include "cuda_compat.h"
#include "dispatch_utils.h"

namespace vllm {

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));

    return old;
}

#define FP8_E4M3_MAX std::numeric_limits<c10::Float8_e4m3fn>::max()

template<typename scalar_t>
__device__ __forceinline__ c10::Float8_e4m3fn scaled_fp8_conversion(const scalar_t val, const float scale) {
  float x = static_cast<float>(val) / scale;
  float r = fmax(-FP8_E4M3_MAX, fmin(x, FP8_E4M3_MAX));
  return static_cast<c10::Float8_e4m3fn>(r);
}

// Compute the absolute maximum m of the input tensor and store
// m / float8_e4m3::max() in *scale. Each thread block performs a
// reduction tree and the memory in scale is atomically updated.
// So to get the right answer, *scale needs to be initialized to
// a value <= 0.0 and we need to wait for all thread blocks to
// finish before consuming *scale.
template<typename scalar_t>
__global__ void segmented_max_reduction(
  float* __restrict__ scale,
  const scalar_t* __restrict__ input,
  int64_t num_elems) {
  __shared__ float cache[1024];
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  // First store maximum for all values processes by
  // the current thread in cache[threadIdx.x]
  scalar_t tmp = 0.0;
  while (i < num_elems) {
    float x = static_cast<float>(input[i]);
    tmp = max(tmp, fabs(x));
    i += blockDim.x * gridDim.x;
  }
  cache[threadIdx.x] = tmp;

  __syncthreads();

  // Now perform parallel reduction within the thread block
  int ib = blockDim.x / 2;
  while (ib != 0) {
    if (threadIdx.x < ib && cache[threadIdx.x + ib] > cache[threadIdx.x]) {
        cache[threadIdx.x] = cache[threadIdx.x + ib];
    }
    __syncthreads();
    ib /= 2;
  }
  // Finally, since cache[0] contains the maximum for this thread block,
  // atomically write the max to the target location
  if (threadIdx.x == 0) {
    atomicMaxFloat(scale, cache[0] / std::numeric_limits<c10::Float8_e4m3fn>::max());
  }
}

template<typename scalar_t>
__global__ void scaled_fp8_quant_kernel(
  c10::Float8_e4m3fn* __restrict__ out,
  const scalar_t* __restrict__ input,
  const float* __restrict__ scale,
  int64_t num_elems) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  while (i < num_elems) {
    out[i] = scaled_fp8_conversion(input[i], *scale);
    i += blockDim.x * gridDim.x;
  }
}

__device__ __forceinline__ float silu_kernel(float x) {
  // x * sigmoid(x)
  return x / (1.0f + expf(-x));
}

__global__ void scaled_fp8_silu_and_mul_kernel(
  c10::Float8_e4m3fn* __restrict__ out,
  const float* __restrict__ scale_out,
  const c10::Float8_e4m3fn* __restrict__ input,
  const float* __restrict__ scale_input,
  const int d) {
  const int64_t token_idx = blockIdx.x;
  float s = *scale_input;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const float x = static_cast<float>(input[token_idx * 2 * d + idx]) * s;
    const float y = static_cast<float>(input[token_idx * 2 * d + d + idx]) * s;
    const float r = silu_kernel(x) * y;
    out[token_idx * d + idx] = scaled_fp8_conversion(r, *scale_out);
  }
}

} // namespace vllm

void static_scaled_fp8_quant(
  torch::Tensor& out,      // [..., d]
  torch::Tensor& input,    // [..., d]
  torch::Tensor& scale)    // [1]
{
  int64_t num_tokens = input.numel() / input.size(-1);
  int64_t num_elems = input.numel();
  dim3 grid(num_tokens);
  dim3 block(1024);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    input.scalar_type(),
    "scaled_fp8_quant_kernel",
    [&] {
      vllm::scaled_fp8_quant_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out.data_ptr<c10::Float8_e4m3fn>(),
        input.data_ptr<scalar_t>(),
        scale.data_ptr<float>(),
        num_elems);
      });
}

void dynamic_scaled_fp8_quant(
  torch::Tensor& out,      // [..., d]
  torch::Tensor& input,    // [..., d]
  torch::Tensor& scale)    // [1]
{
  int64_t num_tokens = input.numel() / input.size(-1);
  int64_t num_elems = input.numel();
  dim3 grid(num_tokens);
  dim3 block(1024);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    input.scalar_type(),
    "scaled_fp8_quant_kernel",
    [&] {
      vllm::segmented_max_reduction<scalar_t><<<grid, block, 0, stream>>>(
        scale.data_ptr<float>(),
        input.data_ptr<scalar_t>(),
        num_elems);
      vllm::scaled_fp8_quant_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out.data_ptr<c10::Float8_e4m3fn>(),
        input.data_ptr<scalar_t>(),
        scale.data_ptr<float>(),
        num_elems);
      });
}


void static_scaled_fp8_silu_and_mul(
  torch::Tensor& out,
  torch::Tensor& scale_out,
  torch::Tensor& input,
  torch::Tensor& scale_input) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);
  dim3 block(std::min(d, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  vllm::scaled_fp8_silu_and_mul_kernel<<grid, block, 0, stream>>(
    out.data_ptr<c10::Float8_e4m3fn>(),
    scale_out.data_ptr<float>(),
    input.data_ptr<c10::Float8_e4m3fn>(),
    scale_input.data_ptr<float>(),
    d
  );
}
