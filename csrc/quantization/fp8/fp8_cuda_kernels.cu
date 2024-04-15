#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>

#include "cuda_compat.h"
#include "dispatch_utils.h"

namespace vllm {

#define CEILDIV(x,y) (((x) + (y) - 1) / (y))

template<typename scalar_t>
__global__ void scaled_fp8_quant_kernel(
  c10::Float8_e4m3fn* __restrict__ out,
  const scalar_t* __restrict__ input,
  const c10::Float8_e4m3fn* __restrict__ scales,
  const int d, const int64_t num_tokens) {
  // __shared__ c10::Float8_e4m3fn ss[1024];
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t token_idx = threadIdx.y;
  if (idx < d && token_idx < num_tokens) {
    // if (token_idx == 0) {
    //     ss[threadIdx.x] = scales[idx];
    // }
    // __syncthreads();
    // const scalar_t s = static_cast<scalar_t>(ss[threadIdx.x]);
    const scalar_t s = static_cast<scalar_t>(scales[idx]);
    const scalar_t x = input[token_idx * d + idx];
    const scalar_t r = x / s;
    out[token_idx * d + idx] = static_cast<c10::Float8_e4m3fn>(r);
  }
}

} // namespace vllm

void scaled_fp8_quant(
  torch::Tensor& out,      // [..., d]
  torch::Tensor& input,    // [..., d]
  torch::Tensor& scales)   // [d]
{
  int d = input.size(-1);
  int64_t num_tokens = input.numel() / input.size(-1);
  int64_t num_threads_x = 1024 / num_tokens;
  dim3 grid(CEILDIV(d, num_threads_x));
  dim3 block(num_threads_x, num_tokens);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES_FP8(
    input.scalar_type(),
    "scaled_fp8_quant_kernel",
    [&] {
      vllm::scaled_fp8_quant_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out.data_ptr<c10::Float8_e4m3fn>(),
        input.data_ptr<scalar_t>(),
        scales.data_ptr<c10::Float8_e4m3fn>(),
        d, num_tokens);
      });
}