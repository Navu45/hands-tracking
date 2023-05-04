from torch import nn


class MultiScaleFusion(nn.Module):
    def __init__(self,
                 depth,
                 ):
        super().__init__()
