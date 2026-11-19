"""
A sample's capture bias, step 1 of the read orientation model's generative model (see
permutect.read_orientation.read_orientation_model): the distribution of the fraction of a site's reads that come from the
+ strand of the original DNA, its F1R2 reads.

Hybrid capture with baits of one strand of the reference recovers the two original strands of a fragment unequally.  The
F1R2 fraction is then the same in direction at every site of a library, relative to the reference, but varies in size
from site to site, and from library to library: in SEQC2's FFPE exomes it is 0.15 to 0.46 on target, while in a fresh
exome it is 0.50, as it is at sites more than a fragment's length from any bait.  So a site's number of F1R2 reads n1, of
its d reads, is beta-binomial,

    n1 ~ BetaBinomial(d; f kappa, (1 - f) kappa),

with mean F1R2 fraction f and concentration kappa.  Capture doesn't see alleles, so this is the same whether the site is
a real variant, an artifact of damage to one strand or neither, and it is fit to the F1R2 counts of all the reads, ref
and alt, at the sites of the alt tables, the sites whose number of F1R2 reads Mutect2 records.  The fit maximizes the
likelihood with L-BFGS, starting from the method of moments.
"""

import torch
from torch import Tensor
from torch import nn
from torch.nn.utils import parametrize

from permutect.architecture.parameterizations import BoundedNumber
from permutect.read_orientation.f1r2_counts_reader import CaptureCounts
from permutect.utils.stats_utils import beta_binomial_log_lk

# Bounds of the concentration, which keep the fit well behaved: a narrower distribution is a binomial at any depth
MIN_CONCENTRATION = 5.0
MAX_CONCENTRATION = 10_000.0
# the fraction and concentration start strictly inside their bounds, where their parametrizations are finite
MIN_STARTING_FRACTION = 1e-3
MAX_ITERATIONS = 100


class CaptureBias(nn.Module):
    """the beta-binomial distribution of a site's number of F1R2 reads given its depth"""

    def __init__(self, f1r2_fraction: float, concentration: float):
        super().__init__()
        f1r2_fraction = min(max(f1r2_fraction, MIN_STARTING_FRACTION), 1 - MIN_STARTING_FRACTION)
        margin = 1e-3 * (MAX_CONCENTRATION - MIN_CONCENTRATION)
        concentration = min(max(concentration, MIN_CONCENTRATION + margin), MAX_CONCENTRATION - margin)
        self.f1r2_fraction = nn.Parameter(torch.tensor(f1r2_fraction, dtype=torch.float64))
        parametrize.register_parametrization(self, "f1r2_fraction", BoundedNumber(0, 1))
        self.concentration = nn.Parameter(torch.tensor(concentration, dtype=torch.float64))
        parametrize.register_parametrization(self, "concentration", BoundedNumber(MIN_CONCENTRATION, MAX_CONCENTRATION))

    def pseudocounts(self) -> tuple[float, float]:
        """the beta distribution's (alpha, beta)"""
        f1r2_fraction, concentration = self.f1r2_fraction.item(), self.concentration.item()
        return f1r2_fraction * concentration, (1 - f1r2_fraction) * concentration

    def log_likelihoods(self, depths: Tensor, f1r2_counts: Tensor) -> Tensor:
        alpha = self.f1r2_fraction * self.concentration
        return beta_binomial_log_lk(depths, f1r2_counts, alpha, self.concentration - alpha)


def moment_estimates(counts: CaptureCounts) -> tuple[float, float]:
    """
    The F1R2 fraction f, pooled over the reads, and the concentration kappa from the excess of the variance of the
    sites' fractions over sampling: var(k / n) = f (1 - f) [1 / n + (1 - 1 / n) rho], with rho = 1 / (kappa + 1).
    Without excess variance, or without sites of two or more reads to measure it, the concentration is the maximum.
    """
    depths, f1r2_counts, weights = (x.double() for x in counts.tensors())
    f1r2_fraction = ((weights * f1r2_counts).sum() / (weights * depths).sum()).item()
    variance = f1r2_fraction * (1 - f1r2_fraction)

    def mean(x: Tensor) -> float:
        return ((weights * x).sum() / weights.sum()).item()

    excess = mean((f1r2_counts / depths - f1r2_fraction) ** 2) - variance * mean(1 / depths)
    denominator = variance * mean(1 - 1 / depths)
    rho = excess / denominator if denominator > 0 else 0.0
    return f1r2_fraction, (1 / rho - 1 if rho > 1 / (MAX_CONCENTRATION + 1) else MAX_CONCENTRATION)


def fit_capture_bias(counts: CaptureCounts) -> CaptureBias | None:
    """the maximum likelihood capture bias of the sites' F1R2 counts, or None if there are no sites"""
    if counts.num_sites() == 0:
        return None
    model = CaptureBias(*moment_estimates(counts))
    depths, f1r2_counts, weights = (x.double() for x in counts.tensors())
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=MAX_ITERATIONS, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = -torch.sum(weights * model.log_likelihoods(depths, f1r2_counts)) / weights.sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return model
