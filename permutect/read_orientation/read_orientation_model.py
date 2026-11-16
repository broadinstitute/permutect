"""
GATK's read orientation artifact model (LearnReadOrientationModel), which learns, for each 3-base reference context, the
prior probabilities of 12 states of a site:

- 8 artifact states: F1R2_A, F1R2_C, F1R2_G, F1R2_T, F2R1_A, F2R1_C, F2R1_G, F2R1_T, eg F1R2_T for alt T reads that
  are mostly in F1R2 orientation, as from damage before amplification of one strand.  The two whose alt base is the
  context's ref base are impossible.
- 4 non-artifact states: HOM_REF, GERMLINE_HET, SOMATIC_HET and HOM_VAR.

Given its state, a site's alt count is beta-binomial in its depth, and its count of alt reads in F1R2 orientation is
beta-binomial in its alt count, with fixed pseudocounts.  An artifact state also fixes the alt base.  A context and
its reverse complement share their priors, with each F1R2 artifact state of one equivalent to the F2R1 state of the
other with the complementary alt base, eg F1R2_T of ACG to F2R1_A of CGT.  The data of each pair go to the
lexicographically smaller, canonical context.

Since the likelihoods are fixed, learning the priors, which expectation maximization (EM) does as in GATK, is a mixture
weights problem.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from permutect.read_orientation.f1r2_counts_reader import BASES
from permutect.read_orientation.f1r2_counts_reader import CONTEXTS
from permutect.read_orientation.f1r2_counts_reader import ORIENTATIONS
from permutect.read_orientation.f1r2_counts_reader import F1R2Counts
from permutect.read_orientation.f1r2_counts_reader import Sites
from permutect.utils.stats_utils import beta_binomial_log_lk

# the states in the order of GATK's ArtifactState, the columns of its prior tables
ARTIFACT_STATES = tuple(f"{orientation}_{base}" for orientation in ORIENTATIONS for base in BASES)
STATES = ARTIFACT_STATES + ("HOM_REF", "GERMLINE_HET", "SOMATIC_HET", "HOM_VAR")
NUM_ARTIFACT_STATES, NUM_STATES = len(ARTIFACT_STATES), len(STATES)
HOM_REF = STATES.index("HOM_REF")
# the alt base (an index into BASES) of each artifact state
ARTIFACT_ALT_BASES = torch.arange(len(BASES)).repeat(len(ORIENTATIONS))
# The state of the reverse complement of a context that is equivalent to each state of the context.  Since the
# complement of base b is 3 - b, it maps artifact state 4 * orientation + b to 4 * (1 - orientation) + 3 - b, that is,
# reverses the artifact states.
REVERSE_COMPLEMENT_STATES_K = torch.cat(
    [torch.arange(NUM_ARTIFACT_STATES).flip(0), torch.arange(NUM_ARTIFACT_STATES, NUM_STATES)]
)

# GATK's fixed pseudocounts (alpha, beta) of the beta distributions of each state's allele fraction and fraction of alt
# reads in F1R2 orientation.  Artifacts have allele fractions around 0.1 and their alt reads almost all in one
# orientation.  Hom ref sites have rare alt reads (Q35), somatic hets low allele fractions and germline hets allele
# fractions broadly around 1/2 for copy number variation.
ALLELE_FRACTION_PSEUDOCOUNTS_K2 = torch.tensor(
    # the artifact states, then HOM_REF, GERMLINE_HET, SOMATIC_HET and HOM_VAR
    [(1.0, 9.0)] * NUM_ARTIFACT_STATES + [(3.0, 10000.0), (5.0, 5.0), (2.0, 5.0), (10000.0, 3.0)],
    dtype=torch.float64,
)
F1R2_FRACTION_PSEUDOCOUNTS_K2 = torch.tensor(
    # the F1R2 artifact states, the F2R1 artifact states, then the non-artifact states
    [(100.0, 1.0)] * 4 + [(1.0, 100.0)] * 4 + [(10.0, 10.0)] * 4,
    dtype=torch.float64,
)


def reverse_complement(sequence: str) -> str:
    return sequence[::-1].translate(str.maketrans(BASES, BASES[::-1]))


# for each pair of a context and its reverse complement, the lexicographically smaller one
CANONICAL_CONTEXTS = tuple(context for context in CONTEXTS if context < reverse_complement(context))


def possible_states(ref_base: int) -> Tensor:
    """
    Whether each state is possible in a context with the given ref base for a site of each alt base, in a
    (len(BASES), NUM_STATES) tensor: the non-artifact states, and the artifact states of the site's alt base unless it
    is the ref base, as it is for a site without alt reads
    """
    alt_bases_a1 = torch.arange(len(BASES)).unsqueeze(-1)
    artifacts_ak = (ARTIFACT_ALT_BASES == alt_bases_a1) & (alt_bases_a1 != ref_base)
    non_artifacts_ak = torch.ones(len(BASES), NUM_STATES - NUM_ARTIFACT_STATES, dtype=torch.bool)
    return torch.cat([artifacts_ak, non_artifacts_ak], dim=-1)


def flat_prior(ref_base: int) -> Tensor:
    """the prior that is uniform over the states that are possible in a context with the given ref base"""
    possible_k = possible_states(ref_base).any(dim=0).double()
    return possible_k / possible_k.sum()


def log_likelihoods(
    sites: Sites,
    ref_base: int,
    allele_fraction_pseudocounts_k2: Tensor = ALLELE_FRACTION_PSEUDOCOUNTS_K2,
    f1r2_fraction_pseudocounts_k2: Tensor = F1R2_FRACTION_PSEUDOCOUNTS_K2,
) -> Tensor:
    """
    The log likelihood of each of the sites' alt count and alt F1R2 count given each state, in an (N, NUM_STATES)
    tensor, -inf for impossible states
    """
    alt_counts_n1, alt_f1r2_counts_n1, depths_n1 = (
        x.double().unsqueeze(-1) for x in (sites.alt_counts_n, sites.alt_f1r2_counts_n, sites.depths_n)
    )
    allele_fraction_nk = beta_binomial_log_lk(depths_n1, alt_counts_n1, *allele_fraction_pseudocounts_k2.unbind(-1))
    f1r2_fraction_nk = beta_binomial_log_lk(
        alt_counts_n1, alt_f1r2_counts_n1, *f1r2_fraction_pseudocounts_k2.unbind(-1)
    )
    impossible_nk = ~possible_states(ref_base)[sites.alt_bases_n]
    return (allele_fraction_nk + f1r2_fraction_nk).masked_fill(impossible_nk, float("-inf"))


def posteriors(log_likelihoods_nk: Tensor, prior_k: Tensor) -> Tensor:
    """the posterior probabilities of each site's states, given their log likelihoods and the prior"""
    return torch.softmax(log_likelihoods_nk + prior_k.log(), dim=-1)


@dataclass
class EMResult:
    prior_k: Tensor
    num_iterations: int
    converged: bool


def learn_prior(sites: Sites, ref_base: int, convergence_threshold: float = 1e-4, max_iterations: int = 20) -> EMResult:
    """
    The maximum a posteriori prior of the states of sites of a context with the given ref base, by EM as in GATK: the
    prior starts flat and is updated until it changes by at most the convergence threshold (Euclidean distance), or for
    at most max_iterations.  Each update adds the flat prior to the expected numbers of sites in each state as
    pseudocounts.
    """
    log_likelihoods_nk = log_likelihoods(sites, ref_base)
    pseudocounts_k = flat_prior(ref_base)
    prior_k = pseudocounts_k
    for iteration in range(1, max_iterations + 1):
        new_prior_k = sites.counts_n.double() @ posteriors(log_likelihoods_nk, prior_k) + pseudocounts_k
        new_prior_k /= new_prior_k.sum()
        distance = torch.dist(new_prior_k, prior_k).item()
        prior_k = new_prior_k
        if distance <= convergence_threshold:
            return EMResult(prior_k, iteration, converged=True)
    return EMResult(prior_k, max_iterations, converged=False)


@dataclass
class ContextPrior:
    """
    The prior probabilities of the states of a reference context, by index, and the number of sites, and of sites with
    alt reads, from which they were learned, both 0 for the flat prior of a context without enough data
    """

    context: str
    prior_k: Tensor
    num_sites: int
    num_alt_sites: int

    def reverse_complement(self) -> "ContextPrior":
        prior_k = self.prior_k[REVERSE_COMPLEMENT_STATES_K]
        return ContextPrior(reverse_complement(self.context), prior_k, self.num_sites, self.num_alt_sites)


def canonical_sites(counts: F1R2Counts, context: str) -> Sites:
    """the sites of a canonical context and of its reverse complement, in the orientation of the canonical context"""
    return counts.sites[context] + counts.sites[reverse_complement(context)].reverse_complement()


def learn_priors(
    counts: F1R2Counts, convergence_threshold: float = 1e-4, max_iterations: int = 20
) -> dict[str, ContextPrior]:
    """
    The prior of every context of a sample, by context.  As in GATK, a canonical context and its reverse complement keep
    the flat prior unless they have sites without alt reads and sites with two or more, which are in GATK's alt tables.
    """
    result = {}
    for context in CANONICAL_CONTEXTS:
        sites, ref_base = canonical_sites(counts, context), BASES.index(context[1])
        num_ref_sites = sites.num_sites(sites.alt_counts_n == 0)
        num_alt_sites = sites.num_sites() - num_ref_sites
        description = f"Sample {counts.sample}, context {context}: {num_ref_sites} ref and {num_alt_sites} alt sites"
        if num_ref_sites == 0 or sites.num_sites(sites.alt_counts_n >= 2) == 0:
            print(f"{description}, too few to learn from.  Using the flat prior.")
            prior = ContextPrior(context, flat_prior(ref_base), 0, 0)
        else:
            em = learn_prior(sites, ref_base, convergence_threshold, max_iterations)
            outcome = f"converged in {em.num_iterations}" if em.converged else f"did not converge in {max_iterations}"
            print(f"{description}, EM {outcome} iterations")
            prior = ContextPrior(context, em.prior_k, sites.num_sites(), num_alt_sites)
        result[context] = prior
        result[reverse_complement(context)] = prior.reverse_complement()
    return result
