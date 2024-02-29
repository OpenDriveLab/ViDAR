#include <string>
#include <torch/extension.h>
#include <vector>

/*
 * CUDA forward declarations
 */

std::vector<torch::Tensor> render_cuda_v2(torch::Tensor sigma,
                                          torch::Tensor origin,
                                          torch::Tensor points,
                                          torch::Tensor tindex,
                                          torch::Tensor sigma_regul);

std::vector<torch::Tensor> get_grad_sigma_cuda_v2(torch::Tensor elementwise_mult,
                                                  torch::Tensor indices,
                                                  torch::Tensor tindex,
                                                  torch::Tensor sigma_shape,
                                                  torch::Tensor indicator,
                                                  torch::Tensor grad_ray_pred);

/*
 * C++ interface
 */

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> render_v2(
    torch::Tensor sigma, torch::Tensor origin,
    torch::Tensor points, torch::Tensor tindex,
    torch::Tensor sigma_regul) {
  CHECK_INPUT(sigma);
  CHECK_INPUT(origin);
  CHECK_INPUT(points);
  CHECK_INPUT(tindex);
  CHECK_INPUT(sigma_regul)
  return render_cuda_v2(sigma, origin, points, tindex, sigma_regul);
}

std::vector<torch::Tensor> get_grad_sigma_v2(torch::Tensor elementwise_mult,
                                             torch::Tensor indices,
                                             torch::Tensor tindex,
                                             torch::Tensor sigma_shape,
                                             torch::Tensor indicator,
                                             torch::Tensor grad_ray_pred) {
  CHECK_INPUT(elementwise_mult);
  CHECK_INPUT(indices);
  CHECK_INPUT(tindex);
  CHECK_INPUT(sigma_shape);
  CHECK_INPUT(indicator);
  CHECK_INPUT(grad_ray_pred);
  return get_grad_sigma_cuda_v2(
    elementwise_mult,
    indices,
    tindex,
    sigma_shape,
    indicator,
    grad_ray_pred);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("render_v2", &render_v2, "Render with returned path and indicator.");
  m.def("get_grad_sigma_v2", &get_grad_sigma_v2, "Get Grad Sigma");
}
