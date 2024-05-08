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

void fp8_scaled_gemm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weights, torch::Tensor& workspace) {
  cublasLtHandle_t ltHandle = at::cuda::getCurrentCUDABlasLtHandle();

  int m = input.size(0);
  int n = weights.size(1);
  int k = weights.size(0);

  int lda = k;
  int ldb = k;
  int ldc = m;

  uint8_t* workspace = workspace.data_ptr<uint8_t>();
  size_t workspaceSize = workspace.numel();

  float alpha = 1.0;

  cublasLtMatmulDesc_t operationDesc = NULL;
  cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;
  cublasLtMatmulPreference_t preference = NULL;

  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_N;
  float beta = 0.0;

  int returnedResults                             = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};

  // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
  // set the transforms for A and B
  checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

  // set scaling factors
  // checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));
  // checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)));
  // checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, &c_scale, sizeof(c_scale)));
  // checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_scale, sizeof(d_scale)));
  // checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &amax_d, sizeof(amax_d)));

  // create matrix descriptors, we are good with the details here so no need to set any extra attributes
  // table of supported type combinations can be found in the documentation: https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, m, n, ldc));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_8F_E4M3, m, n, ldc));

  // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
  // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
  // directly come from cudaMalloc)
  checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
  checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

  // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
  // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
  checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1, &heuristicResult, &returnedResults));

  if (returnedResults == 0) {
      checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
  }

  checkCublasStatus(cublasLtMatmul(ltHandle,
                                   operationDesc,
                                   &alpha,
                                   A,
                                   Adesc,
                                   B,
                                   Bdesc,
                                   &beta,
                                   nullptr,
                                   Cdesc,
                                   D,
                                   Ddesc,
                                   &heuristicResult.algo,
                                   workspace,
                                   workspaceSize,
                                   0));

  // descriptors are no longer needed as all GPU work was already enqueued
  if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
  if (Ddesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Ddesc));
  if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
  if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
  if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
  if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}
