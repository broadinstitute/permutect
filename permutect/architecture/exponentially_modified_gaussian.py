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

SQRT2 = math.sqrt(2.0)

MIN_STDEV = 0.01
MAX_STDEV = 100
STDEV_CONSTRAINT = BoundedNumber(min_val=MIN_STDEV, max_val=MAX_STDEV)

MIN_LAMBDA = 0.01
MAX_LAMBDA = 100
LAMBDA_CONSTRAINT = BoundedNumber(min_val=MIN_LAMBDA, max_val=MAX_LAMBDA)


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

        # With a = lambda*sigma/sqrt(2) and b = (mu - x)/(sqrt(2)*sigma), the argument of erf above is z = a + b and the
        # exponent is a^2 + 2ab = z^2 - b^2.  For large z, erfc(z) underflows, and its log cancels against z^2, so for
        # positive z we use the scaled function erfcx(z) = exp(z^2) erfc(z), which decays only like 1/z.  For negative z,
        # erfcx(z) overflows instead, but erfc(z) is between 1 and 2 and the exponent a(a + 2b) is negative.
        a = self.lambda_ * self.sigma / SQRT2
        b = (self.mu - value) / (SQRT2 * self.sigma)
        z = a + b

        # If one branch of torch.where is a NaN, backpropagation yields a NaN regardless of whether that branch is used!!
        # Clipping keeps both branches finite.
        positive_z = torch.log(torch.special.erfcx(torch.clip(z, min=0))) - torch.square(b)
        negative_z = torch.log(torch.erfc(torch.clip(z, max=0))) + a * (a + 2 * b)
        return torch.log(self.lambda_ / 2) + torch.where(z > 0, positive_z, negative_z)


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
