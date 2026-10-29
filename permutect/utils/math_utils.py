from __future__ import annotations

import math

import torch
from torch import Tensor


def find_factors(n: int):
    result = []
    for m in range(1, int(math.sqrt(n)) + 1):
        if n % m == 0:
            result.append(m)
            if (n // m) > m:
                result.append(n // m)
    result.sort()
    return result


# torch.logit(probs, eps=LOGIT_EPS) clips probabilities to [LOGIT_EPS, 1 - LOGIT_EPS], so that 0 and 1 have finite logits
LOGIT_EPS = 5e-8


def prob_to_logit(prob: float) -> float:
    # the inverse of the sigmoid function.  Convert a probability to a logit.
    return torch.logit(torch.tensor(prob, dtype=torch.float64), eps=LOGIT_EPS).item()


def log_binomial_coefficient(n: Tensor, k: Tensor):
    return (n + 1).lgamma() - (k + 1).lgamma() - ((n - k) + 1).lgamma()


def f_score(precision, recall, beta: float = 1.0):
    """
    The F_beta score, the weighted harmonic mean of precision and recall in which recall is beta times as important as
    precision.  The 0.0001 avoids dividing by zero when both are zero.
    """
    beta_sqr = beta**2
    return (1 + beta_sqr) * recall * precision / (recall + (beta_sqr * precision) + 0.0001)


def subtract_in_log_space(x: Tensor, y: Tensor):
    """
    implement result = log(exp(x) - exp(y)) in a numerically stable way by extracting the elementwise max:
    with M = torch.maximum(x,y); result = log(exp(M)exp(x-M) - exp(M)exp(y-M)) = M + log(exp(x-M) - exp(y-M))

    note: this only works if x > y, elementwise!
    :return:
    """
    m = torch.maximum(x, y)
    return m + torch.log(torch.exp(x - m) - torch.exp(y - m))


def log1mexp(x: Tensor):
    """
    implement result = log(1 - exp(x)) for x <= 0 in a numerically stable way.  Following Machler (2012), use
    log(-expm1(x)) when x is close to zero (exp(x) close to 1) and log1p(-exp(x)) otherwise (exp(x) small).
    """
    return torch.where(x > -math.log(2), torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))
