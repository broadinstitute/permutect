import math

import torch
from torch import Tensor
from torch import logsumexp

from permutect.utils.math_utils import log_binomial_coefficient

"""
Note: all functions below operate on inputs of the same or broadcasting-compatible size.  When, for example, count tensors
such as n and k pertain to a batch while parameters such as p, alpha, beta belong to a model, reshaping may be necessary.

Outputs have same shape as the input eg for the log binomial output[i,j. .] = log binomial(n[i,j..], k[i,j..], p[i,j..]).

All distributions are normalized (they contain nCk combinatorial factors etc) so that
sum_k exp (log prob(k | n, params)) = 1
"""


def binomial_log_lk(n, k, p):
    # binomial distribution log likelihood log Binom(k | n, p)
    # = log [nCk * p^k * (1-p)^(n-k)
    # xlogy(x, y) = x log(y), but 0 when x = 0, so that p = 0 or 1 gives 0 * log(0) = 0 rather than NaN
    return log_binomial_coefficient(n, k) + torch.xlogy(k, p) + torch.xlogy(n - k, 1 - p)


def beta_binomial_log_lk(n, k, alpha, beta):
    # beta binomial distribution log likelihood  log P (k | n, alpha, beta)
    #                   = log integral_{0 to 1} Binomial(k | n, p) Beta(p | alpha, beta) dp
    return (
        log_binomial_coefficient(n, k)
        + torch.lgamma(k + alpha)
        + torch.lgamma(n - k + beta)
        + torch.lgamma(alpha + beta)
        - torch.lgamma(n + alpha + beta)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
    )


def uniform_binomial_log_lk(n: Tensor, k: Tensor, x1: Tensor, x2: Tensor):
    assert x1.shape == x2.shape
    interp = torch.arange(start=0.001, end=0.999, step=0.01).to(device=n.device)
    num_mix = len(interp)

    # 'x' subscript denotes something with the shape of x1 plus an extra dimension
    # 'o' subscript denotes the original mutual shape of n, k, x1, and x2
    x1_x = x1.view(*x1.shape, 1)
    x2_x = x2.view(*x1.shape, 1)
    n_x = n.view(*n.shape, 1)
    k_x = k.view(*k.shape, 1)
    interp_x = interp.view(*([1] * x1.dim()), -1)
    p_x = x2_x * interp_x + x1_x * (1 - interp_x)
    binom_log_lks_x = binomial_log_lk(n_x, k_x, p_x)
    binom_log_lks_o = logsumexp(binom_log_lks_x, dim=-1) - math.log(num_mix)
    return binom_log_lks_o
