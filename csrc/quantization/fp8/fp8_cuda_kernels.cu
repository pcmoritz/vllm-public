#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>

#include <cmath>

#include "cuda_compat.h"
#include "dispatch_utils.h"

namespace cg = cooperative_groups;

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
__device__ void segmented_max_reduction(
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
  float* __restrict__ scale,
  const int64_t num_elems) {
  cg::grid_group grid = cg::this_grid();

  segmented_max_reduction(scale, input, num_elems);

  // Synchronize accross the grid
  cg::sync(grid);

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  while (i < num_elems) {
    out[i] = static_cast<c10::Float8_e4m3fn>(input[i] / *scale);
    i += blockDim.x * gridDim.x;
  }
}

// See https://leimao.github.io/blog/CUDA-Shared-Memory-Templated-Kernel/
template <typename T>
struct SharedMemory
{
    __device__ T* get()
    {
        extern __device__ void Error_UnsupportedType(); // Ensure that we won't compile any un-specialized types
        Error_UnsupportedType();
        return (T*)0;
    }
};

template <>
struct SharedMemory <float>
{
    __device__ float* get() { extern __shared__ float float_data[]; return float_data; }
};

template <>
struct SharedMemory <c10::Half>
{
    __device__ c10::Half* get() { extern __shared__ c10::Half half_data[]; return half_data; }
};

template <>
struct SharedMemory <c10::BFloat16>
{
    __device__ c10::BFloat16* get() { extern __shared__ c10::BFloat16 bfloat16_data[]; return bfloat16_data; }
};


template<typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  // x * sigmoid(x)
  return (T) (((float) x) / (1.0f + expf((float) -x)));
}

template<typename scalar_t>
__global__ void fp8_silu_and_mul_kernel(
  c10::Float8_e4m3fn* __restrict__ out,
  const scalar_t* __restrict__ input,
  float* __restrict__ scale,
  const int d,
  const int64_t num_tokens) {
  cg::grid_group grid = cg::this_grid();

  SharedMemory<scalar_t> smem;
  scalar_t* result = smem.get();
  __shared__ float cache[1024];

  for (int64_t token_idx = blockIdx.x; token_idx < num_tokens; token_idx += gridDim.x) {
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const float x = (float) input[token_idx * 2 * d + idx];
      const float y = (float) input[token_idx * 2 * d + d + idx];
      float r = silu_kernel(x) * y;
      result[idx] = static_cast<scalar_t>(r);
      cache[threadIdx.x] = max(cache[threadIdx.x], fabs(r));
    }
  }

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

  // Synchronize accross the grid
  cg::sync(grid);

  // Convert results to FP8 with scaling
  for (int64_t token_idx = blockIdx.x; token_idx < num_tokens; token_idx += gridDim.x) {
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      out[token_idx * d + idx] = static_cast<c10::Float8_e4m3fn>(result[idx] / *scale);
    }
  }
}

} // namespace vllm

void scaled_fp8_quant(
  torch::Tensor& out,      // [..., d]
  torch::Tensor& input,    // [..., d]
  torch::Tensor& scale)   // [1]
{
  int64_t num_elems = input.numel();
  dim3 grid(128);
  dim3 block(1024);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    input.scalar_type(),
    "scaled_fp8_quant_kernel",
    [&] {
      c10::Float8_e4m3fn* out_ptr = out.data_ptr<c10::Float8_e4m3fn>();
      scalar_t* input_ptr = input.data_ptr<scalar_t>();
      float* scale_ptr = scale.data_ptr<float>();
      void *kernelArgs[] = {(void *)&out_ptr, (void *)&input_ptr,
                            (void *)&scale_ptr, (void *)&num_elems};
      AT_CUDA_CHECK(
        cudaLaunchCooperativeKernel((void *)vllm::scaled_fp8_quant_kernel<scalar_t>,
          grid, block, kernelArgs, 0, stream));
      });
}

void fp8_silu_and_mul_kernel(
  torch::Tensor& out,      // [..., d]
  torch::Tensor& input,    // [..., 2 * d]
  torch::Tensor& scale)   // [1]
{
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(128);
  dim3 block(1024);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    input.scalar_type(),
    "fp8_silu_and_mul_kernel_kernel",
    [&] {
      c10::Float8_e4m3fn* out_ptr = out.data_ptr<c10::Float8_e4m3fn>();
      scalar_t* input_ptr = input.data_ptr<scalar_t>();
      float* scale_ptr = scale.data_ptr<float>();
      void *kernelArgs[] = {(void *)&out_ptr, (void *)&input_ptr,
                            (void *)&scale_ptr, (void*)d, (void *)&num_tokens};
      AT_CUDA_CHECK(
        cudaLaunchCooperativeKernel((void *)vllm::fp8_silu_and_mul_kernel<scalar_t>,
          grid, block, kernelArgs, d * sizeof(scalar_t), stream));
      });
}

