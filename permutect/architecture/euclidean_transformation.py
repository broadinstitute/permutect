import torch
from torch import Tensor
from torch import nn
from torch.nn import Parameter
from torch.nn.utils.parametrizations import orthogonal


class EuclideanTransformation(nn.Module):
    """
    Euclidean transformation: translation composed with rotation i.e. distance- and angle-preserving
    affine transformation.
    """
    def __init__(self, dimension: int):
        super(EuclideanTransformation, self).__init__()
        self.translation_e = Parameter(torch.rand(dimension))
        self.rotation_ee = orthogonal(torch.nn.Linear(dimension, dimension, bias=False))

    def transform(self, x_be: Tensor) -> Tensor:
        return self.rotation_ee(x_be + self.translation_e[None, :])

    def forward(self, x):
        pass
