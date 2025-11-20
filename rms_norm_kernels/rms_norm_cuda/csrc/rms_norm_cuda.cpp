#include <torch/extension.h>

#include <vector>

#include <c10/util/ArrayRef.h>

namespace at::native {

std::tuple<at::Tensor, at::Tensor> rms_norm_cuda(
  const at::Tensor& input,
  IntArrayRef normalized_shape,
  const std::optional<at::Tensor>& weight_opt /* optional */,
  const std::optional<at::Tensor>& bias_opt /* optional */,
  double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &rms_norm_cuda, "RMS Norm CUDA forward");
  // m.def("backward", &rms_norm_cuda_backward, "RMS Norm CUDA backward");
}

} // namespace at::native
