import torch
from cpu.conv import Conv2d

test_data = torch.ones([1, 1, 3, 3])

conv = Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
conv_torch = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)

print('conv based on PyTorch extension: \n', conv(test_data))
print('conv based on PyTorch: \n', conv_torch(test_data))
