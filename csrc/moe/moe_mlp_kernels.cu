#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "../dispatch_utils.h"

#include <c10/util/BFloat16.h>
#include <c10/cuda/CUDAStream.h>

#include "cutlass/bfloat16.h"
#include "cutlass/complex.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"

namespace vllm {

#define CUDA_CALL(code)					                    \
  do {                                              \
    cudaError_t status = code;                      \
    std::string err = cudaGetErrorString(status);   \
    TORCH_CHECK(status == cudaSuccess, err);		    \
  } while (0)

using GroupedGemmKernelNN = typename cutlass::gemm::kernel::DefaultGemmGrouped<
  // Non-transposed A operand.
  ::cutlass::bfloat16_t,
  ::cutlass::layout::RowMajor,
  ::cutlass::ComplexTransform::kNone,
  8,
  // Non-transposed B operand.
  ::cutlass::bfloat16_t,
  ::cutlass::layout::RowMajor,
  ::cutlass::ComplexTransform::kNone,
  8,
  // C operand.
  ::cutlass::bfloat16_t,
  ::cutlass::layout::RowMajor,
  float,
  ::cutlass::arch::OpClassTensorOp,
  ::cutlass::arch::Sm80,
  ::cutlass::gemm::GemmShape<128, 128, 32>,
  ::cutlass::gemm::GemmShape<64, 64, 32>,
  ::cutlass::gemm::GemmShape<16, 8, 16>,
  ::cutlass::epilogue::thread::LinearCombination<::cutlass::bfloat16_t, 8, float, float>,
  // NOTE: Threadblock swizzling is currently not supported by CUTLASS's grouped kernels.
  // This parameter is passed in at present to match the APIs of other kernels. The parameter
  // is unused within the kernel.
  ::cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
  4>::GemmKernel;
using GemmGroupedNN = ::cutlass::gemm::device::GemmGrouped<GroupedGemmKernelNN>;


std::vector<cutlass::gemm::GemmCoord> MakeProblemSizes(torch::Tensor b, torch::Tensor cum_num_tokens_per_expert) {
  const size_t num_experts = cum_num_tokens_per_expert.size(0);
  const size_t k = b.size(1), n = b.size(2);
  std::vector<cutlass::gemm::GemmCoord> problem_sizes(num_experts);
  for (int i = 0; i < num_experts; ++i) {
    int64_t batch_size = cum_num_tokens_per_expert.data_ptr<int64_t>()[i] - (i > 0 ? cum_num_tokens_per_expert.data_ptr<int64_t>()[i-1] : 0);
    problem_sizes[i] = cutlass::gemm::GemmCoord(batch_sizes.data_ptr<int64_t>()[i], n, k);
  }
  return problem_sizes;
}

template <typename T>
torch::Tensor CopyToDevice(const std::vector<T> &x, const torch::Device &device) {
  size_t bytes = x.size() * sizeof(T);
  auto options = torch::TensorOptions().dtype(torch::kInt8).device(device);
  torch::Tensor out = torch::empty(bytes, options);

  CUDA_CALL(cudaMemcpyAsync(out.data_ptr(),
			    x.data(), bytes,
			    cudaMemcpyHostToDevice,
			    c10::cuda::getCurrentCUDAStream()));
  return out;
}

template <typename Gemm>
typename Gemm::Arguments MakeArguments(torch::Tensor a,
				       torch::Tensor b,
				       torch::Tensor c,
				       torch::Tensor batch_sizes) {
  auto problem_sizes_host = MakeProblemSizes(b, batch_sizes);

  // Calculate the number of threadblocks to use and validate the result.
  int64_t num_experts = problem_sizes_host.size();

  // NOTE: This is borrowed from FasterTransformer.
  int threadblock_count = Gemm::sufficient(problem_sizes_host.data(), num_experts);
  if (!threadblock_count) {
    TORCH_CHECK(false, "Grouped GEMM execution not possible with HW");
  }

  // Create the host arrays of leading dimension data and pointer data.
  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;

  std::vector<int64_t> lda_host(num_experts), offsets_a(num_experts);
  std::vector<int64_t> ldb_host(num_experts), offsets_b(num_experts);
  std::vector<int64_t> ldc_host(num_experts), offsets_c(num_experts);
  int64_t elements_a = 0, elements_b = 0, elements_c = 0;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;
  std::vector<ElementA *> ptr_a_host(num_experts);
  std::vector<ElementB *> ptr_b_host(num_experts);
  std::vector<ElementC *> ptr_c_host(num_experts);

  for (int i = 0; i < num_experts; ++i) {
    auto problem = problem_sizes_host[i];
    lda_host[i] = LayoutA::packed({problem.m(), problem.k()}).stride(0);
    ldb_host[i] = LayoutB::packed({problem.k(), problem.n()}).stride(0);
    ldc_host[i] = LayoutC::packed({problem.m(), problem.n()}).stride(0);

    offsets_a[i] = elements_a;
    offsets_b[i] = elements_b;
    offsets_c[i] = elements_c;

    ptr_a_host[i] = (ElementA*)a.data_ptr() + offsets_a[i];
    ptr_b_host[i] = (ElementB*)b.data_ptr() + offsets_b[i];
    ptr_c_host[i] = (ElementC*)c.data_ptr() + offsets_c[i];

    elements_a += problem.m() * problem.k();
    elements_b += problem.k() * problem.n();
    elements_c += problem.m() * problem.n();
  }

  // Copy the problem sizes, pointers and leading dimension data to the device.
  torch::Tensor lda = CopyToDevice(lda_host, a.device());
  torch::Tensor ldb = CopyToDevice(ldb_host, a.device());
  torch::Tensor ldc = CopyToDevice(ldc_host, a.device());
  torch::Tensor ptr_a = CopyToDevice(ptr_a_host, a.device());
  torch::Tensor ptr_b = CopyToDevice(ptr_b_host, a.device());
  torch::Tensor ptr_c = CopyToDevice(ptr_c_host, a.device());
  torch::Tensor problem_sizes = CopyToDevice(problem_sizes_host, a.device());

  typename Gemm::EpilogueOutputOp::Params epilogue_op(/*alpha=*/1.0f, /*beta=*/0.0f);
  typename Gemm::Arguments arguments((cutlass::gemm::GemmCoord*)problem_sizes.data_ptr(),
  				     (int)num_experts,
  				     (int)threadblock_count,
  				     epilogue_op,
  				     (ElementA**)ptr_a.data_ptr(),
  				     (ElementB**)ptr_b.data_ptr(),
  				     (ElementC**)ptr_c.data_ptr(),
  				     (ElementC**)ptr_c.data_ptr(),
  				     /*lda=*/(int64_t*)lda.data_ptr(),
  				     /*ldb=*/(int64_t*)ldb.data_ptr(),
  				     /*ldc=*/(int64_t*)ldc.data_ptr(),
  				     /*ldd=*/(int64_t*)ldc.data_ptr(),
  				     (cutlass::gemm::GemmCoord*)problem_sizes_host.data());
  return arguments;
}

void CutlassGroupedGemm(torch::Tensor a,
				 torch::Tensor b,
				 torch::Tensor c,
				 torch::Tensor cum_num_tokens_per_expert,
         cudaStream_t stream) {
  using Gemm = GemmGroupedNN;
  Gemm gemm;

  auto arguments = MakeArguments<Gemm>(a, b, c, cum_num_tokens_per_expert);
  int64_t workspace_size = gemm.get_workspace_size(arguments);
  auto options = torch::TensorOptions().dtype(torch::kInt8).device(a.device());
  torch::Tensor workspace = torch::empty(workspace_size, options);

  // Initialize the kernel.
  if(gemm.initialize(arguments, workspace.data_ptr()) != cutlass::Status::kSuccess) {
    TORCH_CHECK(false, "Failed to initialize CUTLASS Grouped GEMM");
  }

  // Execute the kernel in the current stream.
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
        float gate_value = gemm_result[i + inter_size];
        T gate_act = fn(gate_value);
        output[i] = fc1_value * gate_act;
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
