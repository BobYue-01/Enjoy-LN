#include <torch/extension.h>

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <variance_utils.h>

#include <cmath>
#include <tuple>

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/cpu/mixed_data_type.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

namespace at::native {

template <typename T,
          typename std::enable_if_t<!is_reduced_floating_point_v<T>, int> = 0>
void RmsNormKernel(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t M,
    int64_t N,
    T eps,
    Tensor* Y,
    Tensor* rstd) {
  using Vec = vec::Vectorized<T>;
  const T* X_data = X.const_data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.const_data_ptr<T>() : nullptr;
  const T* beta_data = beta.defined() ? beta.const_data_ptr<T>() : nullptr;
  T* Y_data = Y->data_ptr<T>();
  T* rstd_data = rstd ? rstd->data_ptr<T>() : nullptr;

  const bool gamma_null = gamma_data == nullptr;
  const bool beta_null = beta_data == nullptr;
  const bool rstd_null = rstd_data == nullptr;
  at::parallel_for(0, M, 1, [&](int64_t start, int64_t end) {
    for (const auto i : c10::irange(start, end)) {
      const T* X_ptr = X_data + i * N;
      T* Y_ptr = Y_data + i * N;
      auto rstd_val = RowwiseVariance(X_ptr, N);
      const T scale = float(1) / std::sqrt(rstd_val + eps);
      if (gamma_null || beta_null) {
        for (const auto j : c10::irange(N)) {
          const T gamma_v = gamma_null ? T(1) : gamma_data[j];
          const T beta_v = beta_null ? T(0) : beta_data[j];
          Y_ptr[j] = X_ptr[j] * rstd_val * gamma_v + beta_v;
        }
      } else {
        vec::map3<T>(
            [scale](Vec x, Vec gamma, Vec beta) {
              return x * Vec(scale) * gamma + beta;
            },
            Y_ptr,
            X_ptr,
            gamma_data,
            beta_data,
            N);
      }
      if (!rstd_null) {
        rstd_data[i] = rstd_val;
      }
    }
  });
}

template void RmsNormKernel<float, 0>(
  const Tensor& X,
  const Tensor& gamma,
  const Tensor& beta,
  int64_t M,
  int64_t N,
  float eps,
  Tensor* Y,
  Tensor* rstd);

template void RmsNormKernel<double, 0>(
  const Tensor& X,
  const Tensor& gamma,
  const Tensor& beta,
  int64_t M,
  int64_t N,
  double eps,
  Tensor* Y,
  Tensor* rstd);

} // namespace at::native
