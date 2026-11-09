"""
The emission model of allele fraction segmentation: the likelihood of a site's alt count given its depth, the population
allele frequency q of its alt allele, and the minor allele fraction (MAF) f of its hidden state.

The sample's genotype is hom ref, het or hom alt with the Hardy-Weinberg probabilities (1 - q)^2, 2q(1 - q) and q^2, so
the fraction x of the sample's alleles that are alt is 0, f or 1 - f (each with probability 1/2, because the phase is
unknown), or 1.  The expected fraction of alt reads comes from x in three steps:
    contamination: a fraction c of the reads come from another person, whose expected alt fraction is q, giving
        (1 - c) x + c q
    reference bias: ref reads are captured and mapped more efficiently than alt reads by a factor of the reference bias
    sequencing error: ref reads become alt reads, and vice versa, with the error rate
The alt count is beta-binomial with that mean, with one concentration for hom genotypes and another for hets.  A small
outlier weight mixes in a uniform distribution of alt counts, for sites such as mismapped segmental duplications that
fit no genotype.

The contamination is a fixed input.  The error rate, the two concentrations, the reference bias and the outlier weight
are learned in the M-step.
"""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch import nn
from torch.nn.utils import parametrize

from permutect.architecture.parameterizations import BoundedNumber
from permutect.architecture.parameterizations import PositiveNumber
from permutect.utils.stats_utils import beta_binomial_log_lk


@dataclass
class Pileups:
    """the alt counts, depths (ref plus alt counts) and population alt allele frequencies of sites, as float64 tensors"""

    alt_counts_n: Tensor
    depths_n: Tensor
    allele_frequencies_n: Tensor

    def subset(self, indices) -> "Pileups":
        return Pileups(self.alt_counts_n[indices], self.depths_n[indices], self.allele_frequencies_n[indices])


class EmissionModel(nn.Module):
    def __init__(
        self,
        contamination: float = 0.0,
        error_rate: float = 1e-3,
        hom_concentration: float = 100.0,
        het_concentration: float = 100.0,
        reference_bias: float = 1.0,
        outlier_weight: float = 1e-3,
        max_outlier_weight: float = 0.01,
    ):
        super().__init__()
        assert 0 < outlier_weight < max_outlier_weight
        self.contamination = contamination

        def parameter(value: float) -> nn.Parameter:
            return nn.Parameter(torch.tensor(value, dtype=torch.float64))

        self.error_rate = parameter(error_rate)
        parametrize.register_parametrization(self, "error_rate", BoundedNumber(0, 1))
        self.hom_concentration = parameter(hom_concentration)
        parametrize.register_parametrization(self, "hom_concentration", PositiveNumber())
        self.het_concentration = parameter(het_concentration)
        parametrize.register_parametrization(self, "het_concentration", PositiveNumber())
        self.reference_bias = parameter(reference_bias)
        parametrize.register_parametrization(self, "reference_bias", PositiveNumber())
        # without a cap, the outlier weight could grow to absorb systematic misfit
        self.outlier_weight = parameter(outlier_weight)
        parametrize.register_parametrization(self, "outlier_weight", BoundedNumber(0, max_outlier_weight))

    def alt_fractions(self, sample_alt_fractions, allele_frequencies: Tensor) -> Tensor:
        """
        The expected fraction of alt reads given the fraction of the sample's alleles that are alt, after contamination,
        reference bias and sequencing error
        """
        mixed = (1 - self.contamination) * sample_alt_fractions + self.contamination * allele_frequencies
        biased = mixed / (mixed + (1 - mixed) * self.reference_bias)
        return biased * (1 - self.error_rate) + (1 - biased) * self.error_rate

    def log_likelihoods(self, pileups: Pileups, mafs_k: Tensor) -> Tensor:
        """log P(alt count | depth, allele frequency, MAF) for each site n and MAF k"""
        alt_counts_n1, depths_n1, allele_frequencies_n1 = (
            tensor[:, None] for tensor in (pileups.alt_counts_n, pileups.depths_n, pileups.allele_frequencies_n)
        )

        def log_beta_binomial(sample_alt_fractions, concentration: Tensor) -> Tensor:
            means = self.alt_fractions(sample_alt_fractions, allele_frequencies_n1)
            return beta_binomial_log_lk(depths_n1, alt_counts_n1, means * concentration, (1 - means) * concentration)

        log_af_n1, log_1m_af_n1 = torch.log(allele_frequencies_n1), torch.log1p(-allele_frequencies_n1)
        hom_ref_n1 = 2 * log_1m_af_n1 + log_beta_binomial(0.0, self.hom_concentration)
        hom_alt_n1 = 2 * log_af_n1 + log_beta_binomial(1.0, self.hom_concentration)
        # the Hardy-Weinberg probability 2q(1 - q) of a het, times 1/2 for each phase
        het_nk = (
            log_af_n1
            + log_1m_af_n1
            + torch.logaddexp(
                log_beta_binomial(mafs_k, self.het_concentration), log_beta_binomial(1 - mafs_k, self.het_concentration)
            )
        )
        germline_nk = torch.logsumexp(torch.stack(torch.broadcast_tensors(hom_ref_n1, het_nk, hom_alt_n1)), dim=0)

        # uniform over alt counts from 0 to the depth
        outlier_n1 = -torch.log(depths_n1 + 1)
        return torch.logaddexp(
            torch.log1p(-self.outlier_weight) + germline_nk, torch.log(self.outlier_weight) + outlier_n1
        )

    def fit(
        self,
        pileups: Pileups,
        posteriors_nk: Tensor,
        mafs_k: Tensor,
        max_iterations: int = 20,
        step_size: float = 1.0,
    ):
        """
        The M-step: maximize sum_nk posteriors_nk log P(data of site n | MAF k), the log likelihood weighted by the
        E-step's posteriors, over the learned parameters.  The objective is deterministic and has only five parameters,
        so L-BFGS converges in a few iterations.  A step size above 1 over-relaxes the M-step, multiplying the change
        in the unconstrained parameters.
        """
        previous_parameters = [parameter.detach().clone() for parameter in self.parameters()]
        optimizer = torch.optim.LBFGS(self.parameters(), max_iter=max_iterations, line_search_fn="strong_wolfe")
        num_sites = len(posteriors_nk)

        def closure():
            optimizer.zero_grad()
            loss = -torch.sum(posteriors_nk * self.log_likelihoods(pileups, mafs_k)) / num_sites
            loss.backward()
            return loss

        optimizer.step(closure)
        if step_size != 1:
            with torch.no_grad():
                for parameter, previous in zip(self.parameters(), previous_parameters):
                    parameter.copy_(previous + step_size * (parameter - previous))
