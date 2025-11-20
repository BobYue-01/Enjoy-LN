#include <torch/extension.h>

#pragma once
// Please note that this file is
// used across both CPU and GPU.

#include <type_traits>
#include <complex>
#include <c10/macros/Macros.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/NumericUtils.h>
#include <ATen/OpMathType.h>
#if defined(__CUDACC__)
#include <ATen/cuda/DeviceUtils.cuh>
#include <ATen/native/cuda/DeviceSqrt.cuh>
#elif defined(__HIPCC__)
#include <ATen/hip/DeviceUtils.cuh>
#include <ATen/native/hip/DeviceSqrt.cuh>
#endif
#if defined(__CUDACC__) || defined(__HIPCC__)
#include <thrust/pair.h>
#else
#include <cmath>
#define device_sqrt std::sqrt
#endif

namespace at::native {

namespace detail {

#if defined(__CUDACC__) || defined(__HIPCC__)
template <typename T1, typename T2> using pair = thrust::pair<T1, T2>;
#else
template <typename T1, typename T2> using pair = std::pair<T1, T2>;
#endif

} // namespace detail

template <typename scalar_t, typename index_t>
struct VarianceData {
  // scalar_t mean;
  scalar_t m2;
  index_t n;
  scalar_t nf;

  C10_HOST_DEVICE VarianceData() : /* mean(0), */ m2(0), n(0), nf(0) {}

  C10_HOST_DEVICE VarianceData(
      // scalar_t mean,
      scalar_t m2,
      index_t n,
      scalar_t nf)
      : /* mean(mean), */ m2(m2), n(n), nf(nf) {}
};


template <typename scalar_t, typename acc_scalar_t, typename index_t /*, typename res_t */>
struct VarianceOps {
  acc_scalar_t correction;
  bool take_sqrt;
 public:
  using acc_t = VarianceData<acc_scalar_t, index_t>;
  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data, index_t /*idx*/) const {
    // We accumulate n in index_t to avoid cumulative rounding error, but still
    // need nf for use in combine where int32 may overflow.
    index_t new_n = acc.n + 1;
    acc_scalar_t new_nf = static_cast<acc_scalar_t>(new_n);
    // acc_scalar_t delta = data - acc.mean;
    // acc_scalar_t new_mean = acc.mean + delta / new_nf;
    // acc_scalar_t new_delta = data - new_mean;
    return {
      // new_mean,
      acc.m2 + data * data,
      new_n,
      new_nf,
    };
  }
  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    if (a.nf == 0) {
      return b;
    }
    if (b.nf == 0) {
      return a;
    }
    // acc_scalar_t delta = b.mean - a.mean;
    acc_scalar_t new_count = a.nf + b.nf;
    // acc_scalar_t nb_over_n = b.nf / new_count;
    return {
      // a.mean + delta * nb_over_n,
      a.m2 + b.m2 /* + delta * delta * a.nf * nb_over_n */,
      // setting acc.n as -1 since acc.n might not be able to represent the count
      // correctly within its range, setting it to -1 to avoid confusion
      -1,
      new_count
    };
  }
  inline C10_DEVICE acc_scalar_t project(acc_t acc) const __ubsan_ignore_float_divide_by_zero__ {
    // const auto mean = static_cast<scalar_t>(acc.mean);
    const auto divisor = acc.nf > correction ? acc.nf - correction : 0;
    const auto var = acc.m2 / divisor;
    // res_t results(take_sqrt ? device_sqrt(var) : var, mean);
    // return results;
    return take_sqrt ? device_sqrt(var) : var;
  }

  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  inline __device__ acc_t warp_shfl_down(acc_t acc, int offset) const {
    return {
      // WARP_SHFL_DOWN(acc.mean, offset)
      /* , */ WARP_SHFL_DOWN(acc.m2, offset)
      , WARP_SHFL_DOWN(acc.n, offset)
      , WARP_SHFL_DOWN(acc.nf, offset)
    };
  }
#endif
  C10_HOST_DEVICE VarianceOps(acc_scalar_t correction, bool take_sqrt)
      : correction(correction), take_sqrt(take_sqrt) {}
};

} // namespace at::native

#undef MAX
#undef MIN
