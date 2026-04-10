import math

import torch
from torch import Tensor
from torch import nn
from torch.nn import Parameter
from torch.nn.utils import parametrize

from permutect.architecture.parameterizations import BoundedNumber

SQRTPI = torch.sqrt(torch.tensor(math.pi))
SQRT2 = torch.sqrt(torch.tensor(2.0))

MIN_STDEV = 0.01
MAX_STDEV = 100
STDEV_CONSTRAINT = BoundedNumber(min_val=MIN_STDEV, max_val=MAX_STDEV)

MIN_LAMBDA = 0.01
MAX_LAMBDA = 100
LAMBDA_CONSTRAINT = BoundedNumber(min_val=MIN_LAMBDA, max_val=MAX_LAMBDA)

"""
Pytorch has no built-in logerfc, and erfc(z) decays so fast (faster than exp(-z^2)!) that it underflows to zero
leading to NaNs when its log is taken.  Here for large arguments we use the asymptotic expansion:

erfc(z) ~ [exp(-z^2)/(z sqrt(pi))] * [1 - 1/(2z^2) + 3/(4z^4) - 15/(8z^6) . . .]
"""
def logerfc(z: Tensor) -> Tensor:
    use_asymptotic = z > 5

    z_clip = torch.clip(z, min=2)
    z_squared = z_clip * z_clip
    z_fourth = z_squared * z_squared
    z_sixth = z_squared * z_fourth

    # asymptotic produces NaN when z is <= 2 or so.  Since only z > 5 is ever used, we can cap z at 2
    asymptotic = (
        -z_squared
        - torch.log(z_clip * SQRTPI)
        + torch.log1p(-1 / (2 * z_squared) + 3 / (4 * z_fourth) - 15 / (8 * z_sixth))
    )

    # when z = 5.0, torch.erfc(z) is 1.5375e-12.  since only z < 5 is ever used, we can cap erfc(z) at 1.0 e-12
    built_in = torch.log(torch.clip(torch.erfc(z), min=1.0e-12))

    # asymptotic_is_nan = asymptotic.isnan() | asymptotic.isinf()
    # built_in_is_nan = built_in.isnan() | built_in.isinf()

    # genuine_nan = ((asymptotic_is_nan & use_asymptotic) | (built_in_is_nan & torch.logical_not(use_asymptotic))).any().item()
    # assert not genuine_nan, "Our logerfc implementation has a bug"

    # If one branch of torch.where is a NaN, backpropagation yields a NaN regardless of whether that branch is used!!
    return torch.where(use_asymptotic, asymptotic, built_in)

class ExponentiallyModifiedGaussian(nn.Module):
    """
    an exponentially-modified Gaussian (EMG) distribution is a Gaussian modulated to have an exp(-lambda*x)
    long tail in the positive-x direction:
        P(x) = (lambda / 2) * [1 - erf((mu + lambda*sigma^2 - x)/(sqrt(2)*sigma))] * \
            exp[(lambda/2) * (2*mu + lambda*sigma^2 - 2*x)]

    This module contains K independent EMG distributions and calculates their log likelihoods independently.
    """
    def __init__(self, num_distributions: int):
        super(ExponentiallyModifiedGaussian, self).__init__()

        self.mu_k = Parameter(2 * torch.ones(num_distributions))

        self.sigma_k = Parameter(torch.ones(num_distributions))
        parametrize.register_parametrization(self, "sigma_k", STDEV_CONSTRAINT)

        self.lambda_k = Parameter(torch.ones(num_distributions))
        parametrize.register_parametrization(self, "lambda_k", LAMBDA_CONSTRAINT)

    # This works as long as the last dimension of x -- the 'k' index -- has the same size as the number of
    # distributions.  The batch index 'b' can actually be any number of dimensions of any size thanks to
    # broadcasting.
    def log_likelihood(self, x_bk: Tensor) -> Tensor:
        variance = torch.square(self.sigma_k)

        return (
                torch.log(self.lambda_k / 2)
                + logerfc((self.mu_k + self.lambda_k * variance - x_bk) / (SQRT2 * self.sigma_k))
                + (self.lambda_k / 2) * (2 * self.mu_k + self.lambda_k * variance - 2 * x_bk)
        )

    def forward(self, x):
        pass
