from torch import nn
import torch
import torch.nn.functional as F
import conv
from pyparsing import Iterable

class ConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, params):
        outputs = conv.forward(input, weights, bias, conv.Conv2dOptions(*[[int(x[0]), int(x[1])] for x in params]))
        ctx.save_for_backward(input, weights, bias, params)
        return outputs

    @staticmethod
    def backward(ctx, out_grad):
        _ = torch.autograd.Variable(torch.zeros(4))
        input, weights, bias, params = ctx.saved_tensors

        gradInput, gradWeight, gradBias = conv.backward(input, out_grad, weights, conv.Conv2dOptions(*[[int(x[0]), int(x[1])] for x in params]))
        return gradInput, gradWeight, gradBias, _
    

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, stride=1, dilation=1):
        super(Conv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        
        params = [kernel_size, stride, padding, dilation]
        params = [[x, ] * 2 if not isinstance(x, Iterable) else x for x in params]
        self.params = torch.autograd.Variable(torch.Tensor(params))

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bias, 0)

    def forward(self, input):
        return ConvFunction.apply(input, self.weight, self.bias, self.params)