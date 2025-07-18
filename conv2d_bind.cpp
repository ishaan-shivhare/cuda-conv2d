#include <torch/extension.h>
#include "conv2d_config.h"

// Forward declaration from .cu file
void launch_conv2d_naive(torch::Tensor input, torch::Tensor kernel, torch::Tensor output);
void launch_conv2d_shared(torch::Tensor input, torch::Tensor kernel, torch::Tensor output);
void launch_conv2d_shared_multi_in(torch::Tensor input, torch::Tensor kernel, torch::Tensor output);

PYBIND11_MODULE(my_cuda_conv, m) {
    m.def("conv2d_naive", &launch_conv2d_naive, "Naive 2D convolution kernel (CUDA)");
    m.def("conv2d_shared", &launch_conv2d_shared, "Shared 2D convolution kernel (CUDA)");
    m.def("conv2d_shared_multi_in", &launch_conv2d_shared_multi_in, "Shared 2D conv (multi-in, CUDA)");
}
