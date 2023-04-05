#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include <THC/THC.h>
#include <iostream>

// C++ interface
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> deform_conv_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    int64_t kW, int64_t kH,
    int64_t dW, int64_t dH,
    int64_t padW, int64_t padH, bool is_bias);

std::vector<torch::Tensor> deform_conv_cuda_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gate_weights,
    torch::Tensor weights);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &deform_conv_forward, "Deformable convolution forward (CUDA)");
  m.def("backward", &deform_conv_backward, "Deformable convolution backward (CUDA)");
}