#include <string>
#include <torch/extension.h>
#include <vector>

/*
 * CUDA forward declarations
 */

std::vector<torch::Tensor> render_cuda(torch::Tensor sigma,
                                       torch::Tensor origin,
                                       torch::Tensor points,
                                       torch::Tensor tindex);

std::vector<torch::Tensor> get_grad_sigma_cuda(torch::Tensor elementwise_mult,
                                               torch::Tensor indices,
                                               torch::Tensor tindex,
                                               torch::Tensor sigma_shape);

torch::Tensor init_cuda(torch::Tensor points, torch::Tensor tindex,
                        const std::vector<int> grid);

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

std::vector<torch::Tensor> render(torch::Tensor sigma, torch::Tensor origin,
                                  torch::Tensor points, torch::Tensor tindex) {
  CHECK_INPUT(sigma);
  CHECK_INPUT(origin);
  CHECK_INPUT(points);
  CHECK_INPUT(tindex);
  return render_cuda(sigma, origin, points, tindex);
}

std::vector<torch::Tensor> get_grad_sigma(torch::Tensor elementwise_mult,
                                          torch::Tensor indices,
                                          torch::Tensor tindex,
                                          torch::Tensor sigma_shape) {
  CHECK_INPUT(elementwise_mult);
  CHECK_INPUT(indices);
  CHECK_INPUT(tindex);
  CHECK_INPUT(sigma_shape);
  return get_grad_sigma_cuda(elementwise_mult, indices, tindex, sigma_shape);
}

torch::Tensor init(torch::Tensor points, torch::Tensor tindex,
                   const std::vector<int> grid) {
  CHECK_INPUT(points);
  CHECK_INPUT(tindex);
  return init_cuda(points, tindex, grid);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &init, "Initialize");
  m.def("render", &render, "Render");
  m.def("get_grad_sigma", &get_grad_sigma, "Get Grad Sigma");
}
