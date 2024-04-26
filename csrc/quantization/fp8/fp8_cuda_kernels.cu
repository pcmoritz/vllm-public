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
    out[i] = static_cast<c10::Float8_e4m3fn>(input[i] / *scale);
    i += blockDim.x * gridDim.x;
  }
}

// Make it so for batch size 12 and Mixtral-8x22B TP2 the fastpath can still be used
constexpr int64_t MAX_ACTIVATION_SIZE_FOR_FASTPATH = 12 * 8192;

// The case where the activations fit into shared memory
template<typename scalar_t>
__global__ void fast_scaled_fp8_quant_kernel(
  c10::Float8_e4m3fn* __restrict__ out,
  const scalar_t* __restrict__ input,
  const float* __restrict__ scale,
  int64_t num_elems,
) {
  __shared__ float cache[1024];
  __shared__ scalar_t activations[MAX_ACTIVATION_SIZE_FOR_FASTPATH];
  int i = threadIdx.x;

  // First store maximum for all values processes by
  // the current thread in cache[threadIdx.x]
  scalar_t tmp = 0.0;
  while (i < num_elems) {
    activations[i] = input[i];
    float x = static_cast<float>(activations[i]);
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

  // Now the maximum is in cache[0], rescale and write the results
  // back to global memory
  *scale = cache[0];

  i = threadIdx.x;
  while (i < num_elems) {
    out[i] = static_cast<c10::Float8_e4m3fn>(activations[i] / cache[0]);
    i += blockDim.x * gridDim.x;
  }
}

} // namespace vllm

void scaled_fp8_quant(
  torch::Tensor& out,      // [..., d]
  torch::Tensor& input,    // [..., d]
  torch::Tensor& scale)    // [1]
{
  int64_t num_tokens = input.numel() / input.size(-1);
  int64_t num_elems = input.numel();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (num_elems <= MAX_ACTIVATION_SIZE_FOR_FASTPATH) {
    dim3 grid(1);
    dim3 block(1024);
    VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(),
      "fast_scaled_fp8_quant_kernel",
      [&] {
        vllm::fast_scaled_fp8_quant_kernel<scalar_t><<<grid, block, 0, stream>>>(
          out.data_ptr<c10::Float8_e4m3fn>(),
          input.data_ptr<scalar_t>(),
          scale.data_ptr<float>(),
          num_elems);
      });
  } else {
    dim3 grid(num_tokens);
    dim3 block(1024);
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
}

