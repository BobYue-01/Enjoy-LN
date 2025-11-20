#include <torch/extension.h>

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <rms_norm.h>

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/cpu/mixed_data_type.h>
#include <c10/util/irange.h>
#include <ATen/OpMathType.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/rms_norm.h>
#include <ATen/ops/zeros_like_native.h>
#endif

#include <array>
#include <tuple>
#include <vector>

namespace at::native {

static void rms_norm_with_rstd_out(
    at::Tensor& out,
    at::Tensor& rstd,
    const at::Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& gamma,
    const Tensor& beta,
    double eps,
    int64_t M,
    int64_t N) {
  if (input.scalar_type() == at::ScalarType::Float) {
      RmsNormKernel<float, 0>(input, gamma, beta, M, N, static_cast<float>(eps), &out, &rstd);
  } else if (input.scalar_type() == at::ScalarType::Double) {
      RmsNormKernel<double, 0>(input, gamma, beta, M, N, eps, &out, &rstd);
  } else {
      TORCH_CHECK(false, "Unsupported data type for RMS normalization");
  }
  const auto input_shape = input.sizes();
  const size_t axis = input.dim() - normalized_shape.size();

  DimVector stat_shape;
  for (const auto idx : c10::irange(axis)) {
    stat_shape.emplace_back(input_shape[idx]);
  }
  for ([[maybe_unused]] const auto idx : c10::irange(axis, input.dim())) {
    stat_shape.emplace_back(1);
  }

  rstd = rstd.view(stat_shape);
}

std::tuple<Tensor, Tensor> rms_norm_forward(
    const torch::Tensor input,
    IntArrayRef normalized_shape,
    const std::optional<torch::Tensor> weight_opt /* optional */,
    const std::optional<torch::Tensor> bias_opt /* optional */,
    double eps) {
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  bool mixed_type = is_mixed_type(input, weight, bias);
  if (mixed_type) {
    check_mixed_data_type(input, weight, bias);
  }

  auto M_N = _check_rms_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto N = M_N.second;
  auto X = input.expect_contiguous();
  auto gamma = weight.expect_contiguous();
  auto beta = bias.expect_contiguous();

  Tensor Y = at::native::empty_like(
      *X,
      std::nullopt /* dtype */,
      std::nullopt /* layout */,
      std::nullopt /* device */,
      std::nullopt /* pin_memory */,
      at::MemoryFormat::Contiguous);
  const auto dtype = param_scalar_type(input, mixed_type);
  Tensor rstd = at::empty({M}, X->options().dtype(dtype));

  rms_norm_with_rstd_out(Y, rstd, *X, normalized_shape, *gamma, *beta, eps, M, N);
  return std::make_tuple(std::move(Y), std::move(rstd));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &rms_norm_forward, "RMS Norm forward");
  // m.def("backward", &rms_norm_backward, "RMS Norm backward");
}

} // namespace at::native
