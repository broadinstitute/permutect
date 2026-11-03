import math

import torch
from torch import Tensor
from torch import nn
from torch.distributions import Distribution
from torch.distributions import Exponential
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
from torch.nn import Parameter
from torch.nn.utils import parametrize

from permutect.architecture.parameterizations import BoundedNumber

SQRTPI = math.sqrt(math.pi)
SQRT2 = math.sqrt(2.0)

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


class ExponentiallyModifiedGaussianDistribution(Distribution):
    """
    an exponentially-modified Gaussian (EMG) distribution is that of the sum of independent N(mu, sigma^2) and
    Exponential(lambda) variables, i.e. a Gaussian modulated to have an exp(-lambda*x) long tail in the positive-x
    direction:
        P(x) = (lambda / 2) * [1 - erf((mu + lambda*sigma^2 - x)/(sqrt(2)*sigma))] * \
            exp[(lambda/2) * (2*mu + lambda*sigma^2 - 2*x)]

    mu, sigma, and lambda_ broadcast against each other to give the batch shape of independent distributions.  Events
    are scalars.
    """

    arg_constraints = {"mu": constraints.real, "sigma": constraints.positive, "lambda_": constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, mu, sigma, lambda_, validate_args=None):
        self.mu, self.sigma, self.lambda_ = broadcast_all(mu, sigma, lambda_)
        super().__init__(batch_shape=self.mu.shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ExponentiallyModifiedGaussianDistribution, _instance)
        batch_shape = torch.Size(batch_shape)
        new.mu = self.mu.expand(batch_shape)
        new.sigma = self.sigma.expand(batch_shape)
        new.lambda_ = self.lambda_.expand(batch_shape)
        super(ExponentiallyModifiedGaussianDistribution, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self) -> Tensor:
        return self.mu + 1 / self.lambda_

    @property
    def variance(self) -> Tensor:
        return torch.square(self.sigma) + 1 / torch.square(self.lambda_)

    def rsample(self, sample_shape=torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        gaussian = self.mu + self.sigma * torch.randn(shape, dtype=self.mu.dtype, device=self.mu.device)
        return gaussian + Exponential(self.lambda_, validate_args=False).rsample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        variance = torch.square(self.sigma)

        return (
            torch.log(self.lambda_ / 2)
            + logerfc((self.mu + self.lambda_ * variance - value) / (SQRT2 * self.sigma))
            + (self.lambda_ / 2) * (2 * self.mu + self.lambda_ * variance - 2 * value)
        )


class ExponentiallyModifiedGaussian(nn.Module):
    """
    This module contains the learnable parameters of K independent EMG distributions.
    """

    def __init__(self, num_distributions: int):
        super(ExponentiallyModifiedGaussian, self).__init__()

        self.mu_k = Parameter(2 * torch.ones(num_distributions))

        self.sigma_k = Parameter(torch.ones(num_distributions))
        parametrize.register_parametrization(self, "sigma_k", STDEV_CONSTRAINT)

        self.lambda_k = Parameter(torch.ones(num_distributions))
        parametrize.register_parametrization(self, "lambda_k", LAMBDA_CONSTRAINT)

    # The batch shape is (K,).  Log likelihoods work as long as the last dimension of x -- the 'k' index -- has the same
    # size as the number of distributions, or size 1.  The batch index 'b' can actually be any number of dimensions of
    # any size thanks to broadcasting.
    def distribution(self) -> ExponentiallyModifiedGaussianDistribution:
        # the parametrizations already constrain sigma and lambda, and validating them makes the CPU wait for the GPU
        return ExponentiallyModifiedGaussianDistribution(self.mu_k, self.sigma_k, self.lambda_k, validate_args=False)

    def forward(self, x):
        pass
