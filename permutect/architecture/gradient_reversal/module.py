from torch import nn

from .functional import revgrad


class GradientReversal(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return revgrad(x, self.alpha)
