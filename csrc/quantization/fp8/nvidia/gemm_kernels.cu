#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "dual_gemm_device.h"
#include "left_silu_and_mul.h"


using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementOutput = cutlass::float_e4m3_t;
using ElementAuxOutput = ElementOutput;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

/*
using ElementOperandA = cutlass::half_t;
using ElementOperandB = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementAccumulator = cutlass::half_t;
using ElementCompute = cutlass::half_t;
*/

using EpilogueOutputOp0 = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
  cutlass::epilogue::thread::Identity<float>,
  ElementOutput,
  ElementAuxOutput,
  128 / cutlass::sizeof_bits<ElementOutput>::value,
  ElementAccumulator,
  ElementAccumulator
>;
using EpilogueOutputOp1 = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
  cutlass::epilogue::thread::Identity<float>,
  ElementOutput,
  ElementAuxOutput,
  128 / cutlass::sizeof_bits<ElementOutput>::value,
  ElementAccumulator,
  ElementAccumulator
>;
using EpilogueOutputOp2 = cutlass::epilogue::thread::LeftSiLUAndMul<
  ElementOutput,
  128 / cutlass::sizeof_bits<ElementOutput>::value,
  ElementOutput,
  ElementCompute
>;

const ElementCompute alpha0 = ElementCompute(1);
const ElementCompute beta0 = ElementCompute(0);
const ElementCompute alpha1 = ElementCompute(1);
const ElementCompute beta1 = ElementCompute(0);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

using DualGemm = cutlass::gemm::device::DualGemm<
  ElementOperandA,
  cutlass::layout::RowMajor,
  ElementOperandB,
  cutlass::layout::ColumnMajor,
  cutlass::layout::ColumnMajor,
  ElementOutput,
  cutlass::layout::RowMajor,
  ElementAccumulator,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp0,
  EpilogueOutputOp1,
  EpilogueOutputOp2,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
  kStages,
  // Do not store the intermediate GEMM outputs
  /*store D0*/false,
  /*store D1*/false,
  kSplitKSerial
>;

void fp8_scaled_gemm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& w0, torch::Tensor& w1, torch::Tensor& workspace) {
  DualGemm gemm;

  int m = input.size(0);
  int n = weights.size(1);
  int k = weights.size(0);
  int l = 1;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, l));
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, l));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, l));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, l));

  cutlass::gemm::GemmCoord problem_size(m, n, k);

  typename Gemm::EpilogueOutputOp::Params::ActivationParams activation0_params{
    alpha0,
    beta0,
  };

  typename EpilogueOutputOp0::Params epilogue0_params{
    activation0_params,
    /* scales */
  }

  typename Gemm::EpilogueOutputOp::Params::ActivationParams activation1_params{
    alpha1,
    beta1,
  };

  typename EpilogueOutputOp0::Params epilogue1_params{
    activation1_params,
    /* scales */
  }

  int64_t stride_A = problem_size.m() * problem_size.k();
  int64_t stride_B0 = problem_size.k() * problem_size.n();
  int64_t stride_B1 = problem_size.k() * problem_size.n();
  int64_t stride_Bias = problem_size.n();
  int64_t stride_D = problem_size.m() * problem_size.n();

  int split_k_slices = 1;
  int batch_count = 1;
  typename DualGemm::Arguments arguments{
      cutlass::gemm::DualGemmMode::kGemm,
      problem_size,
      reinterpret_cast<cutlass::float_e4m3_t*>(input.data_ptr<c10::Float8_e4m3fn>()),
      reinterpret_cast<cutlass::float_e4m3_t*>(w0.data_ptr<c10::Float8_e4m3fn>()),
      /*w0 bias*/nullptr,
      /*D0 intermediate storage*/nullptr,
      reinterpret_cast<cutlass::float_e4m3_t*>(w1.data_ptr<c10::Float8_e4m3fn>()),
      /*w1 bias*/nullptr,
      /*D1 intermediate storage*/nullptr,
      reinterpret_cast<cutlass::float_e4m3_t*>(out.data_ptr<c10::Float8_e4m3fn>()),
      epilogue0_params,
      epilogue1_params,
      {},
      split_k_slices,
      batch_count,
      batch_stride_A,
      batch_stride_B0,
      batch_stride_B1,
      batch_stride_Bias,
      batch_stride_D,
    };

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  TORCH_CHECK(workspace.numel() >= workspace_size);
  TORCH_CHECK(gemm.can_implement(arguments) == cutlass::Status::kSuccess);
  TORCH_CHECK(gemm.initialize(arguments, workspace.data_ptr<uint8_t>()) == cutlass::Status::kSuccess);
  TORCH_CHECK(gemm.run() == cutlass::Status::kSuccess);
}
