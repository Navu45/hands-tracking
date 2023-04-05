#include <iostream>
#include <torch/extension.h>
#include <vector>

namespace F = torch::nn::functional;

struct Conv2dOptions {
  std::pair<int, int> kernel;
  std::pair<int, int> stride = std::make_pair(1, 1);
  std::pair<int, int> padding = std::make_pair(0, 0);
  int groups = 1;
}

// // s'(z) = (1 - s(z)) * s(z)
// torch::Tensor d_sigmoid(torch::Tensor z) {
//   auto s = torch::sigmoid(z);
//   return (1 - s) * s;
// }

// torch::Tensor interpolate_1D(torch::Tensor a, torch::Tensor b) {
//   return torch::max(0, 1 - torch::abs(a - b));
// }

// torch::Tensor interpolate_2D(std::pair<int, int> q, std::pair<int, int> p) {
//   return interpolate_1D(q[0], p[0]) * interpolate_1D(q[1], p[1]);
// }

// torch::Scalar x_offset(torch::Tensor input, std::pair<int, int> p,
//                        torch::Tensor offsets, torch::Tensor R,
//                        std::pair<int, int> kernel_size) {
//   auto x_off = tensor;
//   auto input_accessor = input.accessor<float, 4>();
//   for (int i = 0; i < input.size(2); i++) {
//     for (int j = 0; j < input.size(3); j++) {
//       auto q = std::make_pair(i, j);
//       x_off += interpolate_2D(q, p) * input_accessor[i][j];
//     }
//   }
//   return x_off;
// }

std::vector<torch::Tensor> apply_conv(torch::Tensor input, torch::Tensor output,
                                      torch::Tensor weights, torch::Tensor bias,
                                      // torch::Tensor offsets, torch::Tensor
                                      // grid, torch::Tensor mod_scalars,
                                      Conv2dOptions opt, vector<int> out_size,
                                      int batch_size) {
  for (int batch = 0; batch < batch_size; batch++) {
    torch::Tensor input_n = input[batch];
    output[batch].add_(bias.mm(ones).reshape(out_size), 1);

    // columns.dim: (inplanes * opt.kernel[1] * opt.kernel[0]) * (outHeight *
    // outWidth)
    columns = torch::im2col(input_n.clone(),
                            /*kernel_size=*/opt.kernel,
                            /*dilation=*/{1, 1},
                            /*padding=*/opt.padding,
                            /*stride=*/opt.stride);

    // weights.dim: outplanes * inplanes * opt.kernel[1] * opt.kernel[0],
    // conv(weights, coloumns)
    output[batch].add_(weights.mm(columns).reshape(out_size), 1);
  }
  return {output};
}

std::vector<torch::Tensor>
deform_conv_forward(torch::Tensor input, torch::Tensor weights,
                    torch::Tensor bias, torch::Tensor offsets, torch::Tensor p0,
                    torch::Tensor mod_scalars, Conv2dOptions opt) {
  // Compute sizes - batch_size, in_channels, in_h, in_w
  int batch_size = input.size(0);
  int in_channels = input.size(1);
  int in_h = input.size(2);
  int in_w = input.size(3);

  // batch_size, out_channels, out_h, out_w
  int out_channels = weights.size(0);
  auto out_h = torch::floor(
      (in_h + 2 * opt.padding[0] - opt.kernel[0]) / opt.stride[0] + 1);
  auto out_w = torch::floor(
      (in_w + 2 * opt.padding[1] - opt.kernel[1]) / opt.stride[1] + 1);

  // Prepare tensors for applying convolution
  torch::Tensor output = torch::zeros({batch_size, out_channels, out_h, out_w});
  torch::Tensor columns = torch::zeros(
      {in_channels * opt.kernel[0] * opt.kernel[1], out_h * out_w});
  torch::Tensor ones = torch::ones({1, out_h * out_w});

  // Apply conv
  weights = weights.reshape(
      {out_channels, in_channels * opt.kernel[0] * opt.kernel[1]});
  bias = bias.reshape({out_channels, 1});
  output =
      apply_conv(input, output, weights, bias, {out_channels, out_h, out_w});

  return {input_gate, output_gate, candidate_cell, X, gate_weights};
}

torch::Tensor backward_gradInput(torch::Tensor input, torch::Tensor out_grad,
                                 torch::Tensor weights, Conv2dOptions opt) {

  int batch_size = input.size(0);
  int in_channels = input.size(1);
  int in_h = input.size(2);
  int in_w = input.size(3);

  int out_channels = out_grad.size(1);
  int out_h = out_grad.size(2);
  int out_w = out_grad.size(3);

  torch::Tensor input_grad =
      torch::zeros({batch_size, in_channels, in_h, in_w}).cuda();
  torch::Tensor grad_cols =
      torch::zeros({in_channels * opt.kernel[0] * opt.kernel[1], out_h * out_w})
          .cuda();

  // weight reshape to (inputPlanes * opt.kernel[1] * kernel_h) * outputplanes

  torch::Tensor weights_ = weights.clone();
  weights =
      weights
          .reshape({out_channels, in_channels * opt.kernel[1] * opt.kernel[0]})
          .t()
          .cuda();

  for (int batch = 0; batch < batch_size; batch++) {
    torch::Tensor gradOutput_n = out_grad[batch];

    // gradOutput_n.dim: out_channels * (out_h*out_w)
    gradOutput_n = gradOutput_n.reshape({out_channels, out_h * out_w}).cuda();
    grad_cols = weights.mm(gradOutput_n).cuda();

    input_grad[batch].add_(torch::col2im(grad_cols.clone(),
                                         /*output_size=*/{in_h, in_w},
                                         /*kernel_size=*/opt.kernel,
                                         /*dilation=*/{1, 1},
                                         /*padding*/ opt.padding,
                                         /*stride=*/opt.stride)
                               .cuda(),
                           1);
  }

  return input_grad;
}

std::vector<torch::Tensor> backward_gradParameters(torch::Tensor input,
                                                   torch::Tensor out_grad,
                                                   torch::Tensor weights,
                                                   Conv2dOptions opt) {

  int batch_size = input.size(0);
  int in_channels = input.size(1);
  int in_h = input.size(2);
  int in_w = input.size(3);

  int out_channels = out_grad.size(1);
  int out_h = out_grad.size(2);
  int out_w = out_grad.size(3);

  torch::Tensor gradWeights = torch::zeros({weights.size(0), weights.size(1),
                                            weights.size(2), weights.size(3)})
                                  .cuda();
  torch::Tensor gradBias = torch::zeros({out_channels}).cuda();
  torch::Tensor ones = torch::ones({out_h * out_w, 1}).cuda();

  torch::Tensor columns =
      torch::zeros({in_channels * opt.kernel[1] * opt.kernel[0], out_h * out_w})
          .cuda();

  for (int batch = 0; batch < batch_size; batch++) {
    torch::Tensor gradOutput_n = out_grad[batch];
    gradOutput_n = gradOutput_n.reshape({out_channels, out_h * out_w}).cuda();

    // columns.dim: (inplanes * opt.kernel[1] * opt.kernel[0]) * (outHeight *
    // outWidth)
    columns = torch::im2col(input[batch].clone(),
                            /*kernel_size=*/{opt.kernel[1], opt.kernel[0]},
                            /*dilation=*/{1, 1},
                            /*padding=*/opt.padding,
                            /*stride=*/opt.stride)
                  .t()
                  .cuda();
    gradWeights.add_(
        gradOutput_n.mm(columns)
            .reshape({out_channels, in_channels, opt.kernel[1], opt.kernel[0]})
            .cuda(),
        1);
    gradBias.add_(gradOutput_n.mm(ones).reshape({out_channels}), 1);
  }
  return {gradWeights, gradBias};
}

std::vector<torch::Tensor> deform_conv_backward(torch::Tensor input,
                                                torch::Tensor out_grad,
                                                torch::Tensor weights,
                                                Conv2dOptions opt) {

  torch::Tensor input_grad = backward_gradInput(input, out_grad, weights, opt);
  std::vector<torch::Tensor> gradParas =
      backward_gradParameters(input, out_grad, weights, opt);

  torch::Tensor gradWeights = gradParas[0];
  torch::Tensor gradBias = gradParas[1];

  return {input_grad, gradWeights, gradBias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &deform_conv_forward, "deform_conv forward");
  m.def("backward", &deform_conv_backward, "deform_conv backward");
}

