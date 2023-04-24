from torch import nn
import torch
from fastmetro.model import HandReconstructor


class TestBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 128, 3, 2, 1)

    def forward(self, x):
        return [self.conv(x)]


if __name__ == '__main__':
    batch_size = 10
    in_tensor = torch.rand(batch_size, 3, 7, 7)
    print(in_tensor.shape)
    model = HandReconstructor([128],
                              [128],
                              TestBackbone(),
                              num_layers=1)
    print(model)
    results = model(in_tensor)
    for name in results:
        print(name, results[name].shape)