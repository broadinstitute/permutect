"""
Polymerase slippage rates of STR alleles, and their maximum-likelihood fit to a sample's STR slippage counts from GATK
CollectSTRSlippageCounts.

A fragment of an STR allele slips by j units, for each j in SLIPS, with a probability that depends on the allele's repeat
unit length and its length in bases.  Any other change, of 3 or more units or of part of a unit, is an error with
probability OTHER_ERROR_RATE per count bin.  The log odds of each slip, relative to not slipping, rise with the allele's
length and, at a fixed length, fall with its unit length, which MonotoneSlipLogOdds guarantees.

Each STR's germline genotype is hidden: hom ref, a het with the reference on the minor haplotype, a het with the alt
allele on the minor haplotype, or hom alt.  The alt allele is the STR's most common change of whole units in the sample,
and every other change is an error.  The genotypes' prior probabilities are those of Hardy-Weinberg equilibrium with the
alt allele's population allele frequency q: (1 - q)^2, q(1 - q), q(1 - q) and q^2.  Given the genotype, each fragment
comes from the reference haplotype with probability 1, f, 1 - f or 0, where f is the minor allele fraction (MAF) of the
STR's segment, and then slips or not, so that the STR's counts are multinomial.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch import nn
from torch.nn.functional import softplus

from permutect.slippage.slippage_counts import COUNT_COLUMNS
from permutect.slippage.slippage_counts import MAX_BINNED_CHANGE
from permutect.slippage.slippage_counts import NO_CHANGE_BIN
from permutect.slippage.slippage_counts import SlippageCounts

# the changes, in units, of the slips that have their own rates, and of all the outcomes of a fragment, starting with no
# slip, indexed by o
SLIPS = (-2, -1, 1, 2)
OUTCOMES = (0, *SLIPS)

# the grid of slip rates: unit lengths 1 to MAX_UNIT_LENGTH, which is CollectSTRSlippageCounts' longest unit, and allele
# lengths of 1 to MAX_ALLELE_LENGTH bases.  Longer alleles, 0.02% of exome STRs and 0.2% of genome STRs, slip at the
# rates of the longest.
MAX_UNIT_LENGTH = 5
MAX_ALLELE_LENGTH = 40

# the probability that a fragment lands in a count bin that no outcome of OUTCOMES reaches
OTHER_ERROR_RATE = 1e-4

# the population allele frequency of alt alleles that CollectSTRSlippageCounts' germline resource lacks
DEFAULT_ALLELE_FREQUENCY = 1e-7

# L-BFGS iterations of the fit
MAX_ITERATIONS = 1000

# the genotypes, indexed by g
GENOTYPES = ("hom_ref", "ref_minor_het", "alt_minor_het", "hom_alt")


def inverse_softplus(x: float) -> float:
    return math.log(math.expm1(x))


class MonotoneSlipLogOdds(nn.Module):
    """
    The log odds, relative to not slipping, of each slip j in SLIPS on a grid of unit lengths u = 1, ..., MAX_UNIT_LENGTH
    and allele lengths l = 1, ..., MAX_ALLELE_LENGTH bases, parametrized so that they rise with l and fall with u:

        log_odds[j, 1, l] = start[j] + sum over 1 < l' <= l of softplus(rise[j, l'])
        log_odds[j, u + 1, l] = log_odds[j, u, l] - gap[j, u, l], where
        gap[j, u, l] = softplus(last_gap[j, u]) + sum over l < l' <= MAX_ALLELE_LENGTH of softplus(gap_shrink[j, u, l'])

    The gaps are positive and shrink, or stay the same, as l grows, so each unit length's log odds rise at least as fast as
    those of the next shorter unit length.  That matches SEQC2 normals, where homopolymers' slip rates saturate first.
    Log odds without data keep their initial values.
    """

    def __init__(self):
        super().__init__()
        num_slips, num_gaps, num_steps = len(SLIPS), MAX_UNIT_LENGTH - 1, MAX_ALLELE_LENGTH - 1
        # initially a homopolymer's log odds of slipping by one unit are -5 at 8 bases and rise by 1/2 per base up to 16
        # bases, then barely; slips by two units are e^3 times rarer; and the gaps between unit lengths are 2.5 at 10
        # bases and shrink to 1 at the longest alleles
        lengths_l = torch.arange(2, MAX_ALLELE_LENGTH + 1)
        rises_l = torch.where(lengths_l <= 16, inverse_softplus(0.5), inverse_softplus(0.01)).double()
        self.start_j = nn.Parameter(torch.tensor([-8.5 if abs(j) == 1 else -11.5 for j in SLIPS], dtype=torch.float64))
        self.rise_jl = nn.Parameter(rises_l.expand(num_slips, num_steps).clone())
        self.last_gap_ju = nn.Parameter(torch.full((num_slips, num_gaps), inverse_softplus(1.0), dtype=torch.float64))
        shrink = inverse_softplus(0.05)
        self.gap_shrink_jul = nn.Parameter(torch.full((num_slips, num_gaps, num_steps), shrink, dtype=torch.float64))

    def forward(self) -> Tensor:
        """the log odds of each slip j for each unit length u and allele length l, as a [j, u, l] tensor"""
        rises_jl = torch.cumsum(softplus(self.rise_jl), dim=-1)
        homopolymer_jl = self.start_j[:, None] + torch.cat((torch.zeros_like(rises_jl[:, :1]), rises_jl), dim=-1)
        # the sum over l' > l of the gaps' shrinks, by a reversed cumulative sum, and 0 at the longest alleles
        shrinks_jul = softplus(self.gap_shrink_jul)
        later_shrinks_jul = torch.flip(torch.cumsum(torch.flip(shrinks_jul, dims=(-1,)), dim=-1), dims=(-1,))
        later_shrinks_jul = torch.cat((later_shrinks_jul, torch.zeros_like(later_shrinks_jul[..., :1])), dim=-1)
        gaps_jul = softplus(self.last_gap_ju)[..., None] + later_shrinks_jul
        cumulative_gaps_jul = torch.cumsum(gaps_jul, dim=1)
        cumulative_gaps_jul = torch.cat((torch.zeros_like(cumulative_gaps_jul[:, :1]), cumulative_gaps_jul), dim=1)
        return homopolymer_jl[:, None, :] - cumulative_gaps_jul


def outcome_probabilities(log_odds_j: Tensor) -> Tensor:
    """the probabilities of OUTCOMES, ie of no slip and of each slip, along the last dimension, given the slips' log odds"""
    return torch.softmax(torch.cat((torch.zeros_like(log_odds_j[..., :1]), log_odds_j), dim=-1), dim=-1)


def slip_probabilities(log_odds_jul: Tensor) -> Tensor:
    """the probability of each slip, ignoring other errors, from a [j, u, l] tensor of log odds, as a [j, u, l] tensor"""
    return outcome_probabilities(log_odds_jul.movedim(0, -1)).movedim(-1, 0)[1:]


@dataclass
class AlleleBins:
    """
    Where the fragments of one allele of each STR s land: the index of the allele's slip log odds in the flattened grid of
    unit lengths and allele lengths, the count bin of each outcome o of OUTCOMES, and the count bins k that none reach
    """

    grid_indices_s: Tensor
    bins_so: Tensor
    unreached_sk: Tensor

    @classmethod
    def of(cls, unit_lengths_s: Tensor, ref_lengths_s: Tensor, changes_s: Tensor) -> "AlleleBins":
        """the bins of the alleles whose lengths differ from the reference's by changes_s units"""
        lengths_s = torch.clamp(ref_lengths_s + changes_s * unit_lengths_s, 1, MAX_ALLELE_LENGTH)
        grid_indices_s = (unit_lengths_s - 1) * MAX_ALLELE_LENGTH + lengths_s - 1
        changes_so = changes_s[:, None] + torch.tensor(OUTCOMES)
        bins_so = torch.clamp(changes_so, -MAX_BINNED_CHANGE, MAX_BINNED_CHANGE) + NO_CHANGE_BIN
        reached_sk = torch.zeros(len(changes_s), len(COUNT_COLUMNS), dtype=torch.bool).scatter_(1, bins_so, True)
        return cls(grid_indices_s, bins_so, (~reached_sk).double())

    def probabilities_sk(self, log_odds_jul: Tensor) -> Tensor:
        """the probability that a fragment of the allele lands in each count bin"""
        log_odds_sj = log_odds_jul.reshape(len(SLIPS), -1)[:, self.grid_indices_s].T
        reached_s = 1 - OTHER_ERROR_RATE * self.unreached_sk.sum(dim=-1)
        probabilities_so = outcome_probabilities(log_odds_sj) * reached_s[:, None]
        return (OTHER_ERROR_RATE * self.unreached_sk).scatter_add(1, self.bins_so, probabilities_so)


@dataclass
class SlippageData:
    """
    What the likelihood of the slip log odds needs from each STR s, which stays the same during the fit: where the
    fragments of its reference and alt alleles land, the fraction of fragments from the reference haplotype and the log
    prior probability of each genotype g of GENOTYPES, and the counts in each count bin k
    """

    ref: AlleleBins
    alt: AlleleBins
    ref_fractions_sg: Tensor
    log_priors_sg: Tensor
    counts_sk: Tensor

    @classmethod
    def of(cls, counts: SlippageCounts, mafs_s: Tensor, default_allele_frequency: float = DEFAULT_ALLELE_FREQUENCY):
        """
        The data of the STRs of counts, whose MAFs are mafs_s.  Alt alleles without a population allele frequency get
        default_allele_frequency, and STRs without an alt allele, ie without a change of whole units, are hom ref.
        """
        unit_lengths_s, ref_lengths_s = counts.unit_lengths_s, counts.repeat_lengths_s
        if counts.num_strs() > 0 and unit_lengths_s.max().item() > MAX_UNIT_LENGTH:
            raise ValueError(f"The slip rates are of repeat units of at most {MAX_UNIT_LENGTH} bases")
        alt_changes_s = counts.alt_changes_sa[:, 0]
        allele_frequencies_s = counts.alt_allele_frequencies_sa[:, 0].double()
        allele_frequencies_s = torch.where(allele_frequencies_s.isnan(), default_allele_frequency, allele_frequencies_s)
        allele_frequencies_s = torch.where(alt_changes_s == 0, 0.0, allele_frequencies_s)
        log_q_s, log_1mq_s = torch.log(allele_frequencies_s), torch.log1p(-allele_frequencies_s)
        log_priors_sg = torch.stack((2 * log_1mq_s, log_q_s + log_1mq_s, log_q_s + log_1mq_s, 2 * log_q_s), dim=-1)
        mafs_s = mafs_s.double()
        ref_fractions_sg = torch.stack((torch.ones_like(mafs_s), mafs_s, 1 - mafs_s, torch.zeros_like(mafs_s)), dim=-1)
        return cls(
            ref=AlleleBins.of(unit_lengths_s, ref_lengths_s, torch.zeros_like(alt_changes_s)),
            alt=AlleleBins.of(unit_lengths_s, ref_lengths_s, alt_changes_s),
            ref_fractions_sg=ref_fractions_sg,
            log_priors_sg=log_priors_sg,
            counts_sk=counts.counts_sk.double(),
        )

    def log_likelihoods_s(self, log_odds_jul: Tensor) -> Tensor:
        """the log likelihood of each STR's counts given the slip log odds, without the multinomial coefficient"""
        ref_sk, alt_sk = self.ref.probabilities_sk(log_odds_jul), self.alt.probabilities_sk(log_odds_jul)
        ref_fractions_sgk = self.ref_fractions_sg[..., None]
        probabilities_sgk = ref_fractions_sgk * ref_sk[:, None, :] + (1 - ref_fractions_sgk) * alt_sk[:, None, :]
        log_likelihoods_sg = torch.sum(self.counts_sk[:, None, :] * torch.log(probabilities_sgk), dim=-1)
        return torch.logsumexp(self.log_priors_sg + log_likelihoods_sg, dim=-1)


def fit_slip_log_odds(data: SlippageData, max_iterations: int = MAX_ITERATIONS) -> tuple[Tensor, float, int]:
    """
    The slip log odds that maximize the likelihood of the data, by L-BFGS from MonotoneSlipLogOdds' initial values; the
    log likelihood per fragment; and the number of L-BFGS iterations
    """
    model = MonotoneSlipLogOdds()
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        max_iter=max_iterations,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
    )
    num_fragments = max(data.counts_sk.sum().item(), 1)

    def closure():
        optimizer.zero_grad()
        loss = -torch.sum(data.log_likelihoods_s(model())) / num_fragments
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        log_odds_jul = model()
        log_likelihood = torch.sum(data.log_likelihoods_s(log_odds_jul)).item() / num_fragments
    # L-BFGS keeps its state, including the iteration count, with its first parameter
    return log_odds_jul, log_likelihood, optimizer.state_dict()["state"][0]["n_iter"]
