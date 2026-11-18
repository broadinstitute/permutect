"""
A sample's baseline orientation: the expected fraction of the reads at a site that are in F1R2 orientation, and how
much that fraction varies from site to site.  GATK's read orientation model gives the alt reads of every non-artifact
site an F1R2 count that is beta-binomial with fixed pseudocounts (10, 10): a fraction of 1/2, varying with a standard
deviation of about 0.11.  But hybrid capture skews the reads of some libraries, eg to a fraction of 0.15 in FFPE exomes,
so that the alt reads of real variants look like F2R1 artifacts, while in others the fraction barely varies beyond
sampling, so that Beta(10, 10) makes artifacts look ordinary.

The ref reads and alt reads of a site are skewed alike, and ref reads aren't artifacts, so the baseline is fit to the
F1R2 counts of the ref reads at the sites of the alt tables: a beta-binomial in each site's ref count, with mean the
baseline F1R2 fraction f and concentration kappa, that is, pseudocounts (f kappa, (1 - f) kappa).  The fit maximizes the
likelihood with L-BFGS, starting from the method of moments.
"""

import torch
from torch import Tensor
from torch import nn
from torch.nn.utils import parametrize

from permutect.architecture.parameterizations import BoundedNumber
from permutect.read_orientation.f1r2_counts_reader import RefOrientations
from permutect.utils.stats_utils import beta_binomial_log_lk

# Bounds of the concentration: a broader distribution of F1R2 fractions would explain artifacts' alt reads too, and a
# narrower one is a binomial at any depth
MIN_CONCENTRATION = 5.0
MAX_CONCENTRATION = 10_000.0
# the fraction and concentration start strictly inside their bounds, where their parametrizations are finite
MIN_STARTING_FRACTION = 1e-3
MAX_ITERATIONS = 100


class BaselineOrientation(nn.Module):
    """the beta-binomial distribution of the F1R2 count of a non-artifact site's reads given their number"""

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


def moment_estimates(orientations: RefOrientations) -> tuple[float, float]:
    """
    The F1R2 fraction f, pooled over the reads, and the concentration kappa from the excess of the variance of the
    sites' fractions over sampling: var(k / n) = f (1 - f) [1 / n + (1 - 1 / n) rho], with rho = 1 / (kappa + 1).
    Without excess variance, or without sites of two or more reads to measure it, the concentration is the maximum.
    """
    depths, f1r2_counts, weights = (x.double() for x in orientations.tensors())
    f1r2_fraction = ((weights * f1r2_counts).sum() / (weights * depths).sum()).item()
    variance = f1r2_fraction * (1 - f1r2_fraction)

    def mean(x: Tensor) -> float:
        return ((weights * x).sum() / weights.sum()).item()

    excess = mean((f1r2_counts / depths - f1r2_fraction) ** 2) - variance * mean(1 / depths)
    denominator = variance * mean(1 - 1 / depths)
    rho = excess / denominator if denominator > 0 else 0.0
    return f1r2_fraction, (1 / rho - 1 if rho > 1 / (MAX_CONCENTRATION + 1) else MAX_CONCENTRATION)


def fit_baseline_orientation(orientations: RefOrientations) -> BaselineOrientation | None:
    """the maximum likelihood baseline orientation of the ref reads' F1R2 counts, or None if there are no ref reads"""
    if orientations.num_sites() == 0:
        return None
    model = BaselineOrientation(*moment_estimates(orientations))
    depths, f1r2_counts, weights = (x.double() for x in orientations.tensors())
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=MAX_ITERATIONS, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = -torch.sum(weights * model.log_likelihoods(depths, f1r2_counts)) / weights.sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return model
