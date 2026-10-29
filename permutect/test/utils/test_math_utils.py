import math

import pytest
import torch

from permutect.utils.math_utils import LOGIT_EPS
from permutect.utils.math_utils import find_factors
from permutect.utils.math_utils import log1mexp
from permutect.utils.math_utils import prob_to_logit
from permutect.utils.math_utils import subtract_in_log_space


def test_find_factors():
    assert find_factors(1) == [1]
    assert find_factors(2) == [1, 2]
    assert find_factors(4) == [1, 2, 4]
    assert find_factors(7) == [1, 7]
    assert find_factors(24) == [1, 2, 3, 4, 6, 8, 12, 24]


def test_prob_to_logit():
    # the inverse of the sigmoid.  Every probability used to be squeezed toward 1/2, which changed the logit of an error
    # probability of 1e-6 by 5%; now only 0 and 1 are clipped, to LOGIT_EPS and 1 - LOGIT_EPS, so their logits are finite
    for prob in (1e-6, 0.1, 0.5, 0.9, 1 - 1e-6):
        assert prob_to_logit(prob) == pytest.approx(math.log(prob / (1 - prob)), rel=1e-9)
    max_logit = math.log((1 - LOGIT_EPS) / LOGIT_EPS)
    assert prob_to_logit(0.0) == pytest.approx(-max_logit) and prob_to_logit(1.0) == pytest.approx(max_logit)


def test_subtract_in_log_space():
    y = torch.rand((4, 5))
    x = y + torch.rand((4, 5))  # this guarantees y < x, elementwise and both are positive
    logx, logy = torch.log(x), torch.log(y)
    exact = x - y
    calc = torch.exp(subtract_in_log_space(logx, logy))
    assert torch.sum(torch.abs(exact - calc)).item() < 10 ** (-5)


def test_log1mexp():
    # cover both branches: x near zero (exp(x) near 1) and very negative x (exp(x) tiny)
    x = torch.tensor([-1e-10, -1e-4, -0.1, -0.69, -0.7, -1.0, -5.0, -20.0, -50.0], dtype=torch.float64)
    exact = torch.log(1 - torch.exp(x.to(torch.float64)))
    exact[0] = torch.log(torch.tensor(1e-10, dtype=torch.float64))  # 1 - exp(-1e-10) loses precision in float64
    exact[-2:] = -torch.exp(x[-2:])  # log(1 - eps) ~ -eps, and float64 can't resolve 1 - exp(-50)
    calc = log1mexp(x.to(torch.float32)).to(torch.float64)
    assert torch.allclose(calc, exact, rtol=1e-5, atol=0)
    assert log1mexp(torch.tensor([0.0])).item() == float("-inf")
