#include <iostream>
#include <torch/extension.h>
#include <utility>
#include <vector>

struct Conv2dOptions {
  std::vector<int64_t> kernel;
  std::vector<int64_t> stride = {1, 1};
  std::vector<int64_t> padding = {0, 0};
  std::vector<int64_t> dilation = {1, 1};

  Conv2dOptions(std::vector<int64_t> _kernel, std::vector<int64_t> _stride,
                std::vector<int64_t> _padding, std::vector<int64_t> _dilation) {
    kernel = _kernel;
    stride = _stride;
    padding = _padding;
    dilation = _dilation;
  }
};

torch::Tensor conv_forward(torch::Tensor input,
                                        torch::Tensor weights,
                                        torch::Tensor bias, Conv2dOptions opt) {

  int64_t batch_size = input.size(0);
  int64_t in_channels = input.size(1);
  int64_t in_h = input.size(2);
  int64_t in_w = input.size(3);

  int64_t out_channels = weights.size(0);
  int64_t out_h =
      (in_h + 2 * opt.padding[0] - opt.kernel[0]) / opt.stride[0] + 1;
  int64_t out_w =
      (in_w + 2 * opt.padding[1] - opt.kernel[1]) / opt.stride[1] + 1;

  torch::Tensor output = torch::zeros({batch_size, out_channels, out_h, out_w});
  torch::Tensor columns = torch::zeros(
      {in_channels * opt.kernel[1] * opt.kernel[0], out_h * out_w});
  torch::Tensor ones = torch::ones({1, out_h * out_w});

  // after reshape, weights conv with columns
  weights = weights.reshape(
      {out_channels, in_channels * opt.kernel[1] * opt.kernel[0]});
  bias = bias.reshape({out_channels, 1});

  for (int64_t batch = 0; batch < batch_size; batch++) {
    torch::Tensor input_n = input[batch];
    output[batch].add_(bias.mm(ones).reshape({out_channels, out_h, out_w}),
                         1);

    // columns.dim: (inplanes * opt.kernel[1] * opt.kernel[0]) *
    // (outHeight * outWidth)
    columns = torch::im2col(input_n.clone(),
                            /*kernel_size=*/opt.kernel,
                            /*dilation=*/{1, 1},
                            /*padding=*/opt.padding,
                            /*stride=*/opt.stride);

    // weights.dim: outplanes * inplanes * opt.kernel[1] * opt.kernel[0],
    // conv(weights, coloumns)
    output[batch].add_(
        weights.mm(columns).reshape({out_channels, out_h, out_w}), 1);
  }
  return {output};
}

torch::Tensor backward_gradInput(torch::Tensor input, torch::Tensor gradOutput,
                                 torch::Tensor weights, Conv2dOptions opt) {

  int64_t batch_size = input.size(0);
  int64_t in_channels = input.size(1);
  int64_t in_h = input.size(2);
  int64_t in_w = input.size(3);

  int64_t out_channels = gradOutput.size(1);
  int64_t out_h = gradOutput.size(2);
  int64_t out_w = gradOutput.size(3);

  torch::Tensor gradInput = torch::zeros({batch_size, in_channels, in_h, in_w});
  torch::Tensor gradColumns = torch::zeros(
      {in_channels * opt.kernel[0] * opt.kernel[1], out_h * out_w});

  /* weight reshape to (inputPlanes * opt.kernel[1] * opt.kernel[0]) *
   * outputplanes */
  torch::Tensor weights_ = weights.clone();
  weights =
      weights
          .reshape({out_channels, in_channels * opt.kernel[1] * opt.kernel[0]})
          .t();

  for (int64_t batch = 0; batch < batch_size; batch++) {
    torch::Tensor gradInput_n = gradInput[batch];
    torch::Tensor gradOutput_n = gradOutput[batch];

    // gradOutput_n.dim: out_channels * (out_h*out_w)
    gradOutput_n = gradOutput_n.reshape({out_channels, out_h * out_w});

    gradColumns = weights.mm(gradOutput_n);

    gradInput[batch].add_(torch::col2im(gradColumns.clone(),
                                        /*output_size=*/{in_h, in_w},
                                        /*kernel_size=*/opt.kernel,
                                        /*dilation=*/{1, 1},
                                        /*padding=*/opt.padding,
                                        /*stride=*/opt.stride),
                          1);
  }

  return gradInput;
}

std::vector<torch::Tensor> backward_gradParameters(torch::Tensor input,
                                                   torch::Tensor gradOutput,
                                                   torch::Tensor weights,
                                                   Conv2dOptions opt) {

  int64_t batch_size = input.size(0);
  int64_t in_channels = input.size(1);
  int64_t in_h = input.size(2);
  int64_t in_w = input.size(3);

  int64_t out_channels = gradOutput.size(1);
  int64_t out_h = gradOutput.size(2);
  int64_t out_w = gradOutput.size(3);

  torch::Tensor gradWeights = torch::zeros(
      {weights.size(0), weights.size(1), weights.size(2), weights.size(3)});
  torch::Tensor gradBias = torch::zeros({out_channels});
  torch::Tensor ones = torch::ones({out_h * out_w, 1});

  torch::Tensor columns = torch::zeros(
      {in_channels * opt.kernel[1] * opt.kernel[0], out_h * out_w});

  for (int64_t batch = 0; batch < batch_size; batch++) {
    torch::Tensor gradOutput_n = gradOutput[batch];
    gradOutput_n = gradOutput_n.reshape({out_channels, out_h * out_w});

    // columns.dim: (inplanes * opt.kernel[1] * opt.kernel[0]) *
    // (outHeight * outWidth)
    columns = torch::im2col(input[batch].clone(),
                            /*kernel_size=*/opt.kernel,
                            /*dilation=*/{1, 1},
                            /*padding=*/opt.padding,
                            /*stride=*/opt.stride)
                  .t();
    gradWeights.add_(
        gradOutput_n.mm(columns).reshape(
            {out_channels, in_channels, opt.kernel[1], opt.kernel[0]}),
        1);
    gradBias.add_(gradOutput_n.mm(ones).reshape({out_channels})), 1;
    
  }
  return {gradWeights, gradBias};
}

std::vector<torch::Tensor> conv_backward(torch::Tensor input,
                                         torch::Tensor gradOutput,
                                         torch::Tensor weights,
                                         Conv2dOptions opt) {
  torch::Tensor gradInput = backward_gradInput(input, gradOutput, weights, opt);
  std::vector<torch::Tensor> gradParas =
      backward_gradParameters(input, gradOutput, weights, opt);

  torch::Tensor gradWeights = gradParas[0];
  torch::Tensor gradBias = gradParas[1];

  return {gradInput, gradWeights, gradBias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_forward, "conv forward (CPU)");
  m.def("backward", &conv_backward, "conv backward (CPU)");
  py::class_<Conv2dOptions>(m, "Conv2dOptions")
      .def(py::init<std::vector<int64_t>, std::vector<int64_t>,
                    std::vector<int64_t>, std::vector<int64_t>>());
}