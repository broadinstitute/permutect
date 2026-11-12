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

import numpy as np
import torch
from torch import Tensor

from permutect.read_orientation.picard_histogram_reader import BASES
from permutect.read_orientation.picard_histogram_reader import CONTEXTS
from permutect.read_orientation.picard_histogram_reader import ORIENTATIONS
from permutect.read_orientation.picard_histogram_reader import AltSites
from permutect.read_orientation.picard_histogram_reader import F1R2Counts
from permutect.utils.stats_utils import beta_binomial_log_lk

# in the order of GATK's ArtifactState, the columns of its prior tables
STATES = (
    "F1R2_A", "F1R2_C", "F1R2_G", "F1R2_T", "F2R1_A", "F2R1_C", "F2R1_G", "F2R1_T",
    "HOM_REF", "GERMLINE_HET", "SOMATIC_HET", "HOM_VAR",
)  # fmt: skip
NUM_STATES = len(STATES)
NUM_ARTIFACT_STATES = 8
HOM_REF = STATES.index("HOM_REF")
# the alt base (an index into BASES) of each artifact state, and -1 for the non-artifact states
STATE_ALT_BASES_K = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, -1, -1, -1, -1])

COMPLEMENTS = {"A": "T", "C": "G", "G": "C", "T": "A"}


def reverse_complement(sequence: str) -> str:
    return "".join(COMPLEMENTS[base] for base in reversed(sequence))


# for each pair of a context and its reverse complement, the lexicographically smaller one
CANONICAL_CONTEXTS = tuple(context for context in CONTEXTS if context < reverse_complement(context))


def _reverse_complement_state(state: str) -> str:
    if state.startswith("F1R2_") or state.startswith("F2R1_"):
        orientation, alt_base = state.split("_")
        other_orientation = ORIENTATIONS[1 - ORIENTATIONS.index(orientation)]
        return other_orientation + "_" + COMPLEMENTS[alt_base]
    return state


# the state of the reverse complement context that is equivalent to each state, by index
REVERSE_COMPLEMENT_STATES_K = torch.tensor([STATES.index(_reverse_complement_state(state)) for state in STATES])

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


def impossible_states(ref_base: int, alt_bases_n: Tensor) -> Tensor:
    """
    Whether each state is impossible for sites with the given alt bases (indices into BASES): an artifact state whose
    alt base is the ref base, or another alt base.  Sites without alt reads have the ref base as their alt base, so that
    every artifact state is impossible for them.
    """
    state_alt_bases_k = STATE_ALT_BASES_K.to(alt_bases_n.device)
    is_artifact_k = state_alt_bases_k >= 0
    wrong_alt_base_nk = state_alt_bases_k != alt_bases_n.unsqueeze(-1)
    return is_artifact_k & (wrong_alt_base_nk | (state_alt_bases_k == ref_base))


def log_likelihoods(
    ref_base: int,
    alt_bases_n: Tensor,
    alt_counts_n: Tensor,
    alt_f1r2_counts_n: Tensor,
    depths_n: Tensor,
    allele_fraction_pseudocounts_k2: Tensor = ALLELE_FRACTION_PSEUDOCOUNTS_K2,
    f1r2_fraction_pseudocounts_k2: Tensor = F1R2_FRACTION_PSEUDOCOUNTS_K2,
) -> Tensor:
    """
    The log likelihood of each site's alt count and alt F1R2 count given each state, in an (N, NUM_STATES) tensor,
    -inf for impossible states
    """
    alt_counts_n1, alt_f1r2_counts_n1, depths_n1 = (
        x.to(torch.float64).unsqueeze(-1) for x in (alt_counts_n, alt_f1r2_counts_n, depths_n)
    )
    allele_fraction_nk = beta_binomial_log_lk(
        depths_n1, alt_counts_n1, allele_fraction_pseudocounts_k2[:, 0], allele_fraction_pseudocounts_k2[:, 1]
    )
    f1r2_fraction_nk = beta_binomial_log_lk(
        alt_counts_n1, alt_f1r2_counts_n1, f1r2_fraction_pseudocounts_k2[:, 0], f1r2_fraction_pseudocounts_k2[:, 1]
    )
    result_nk = allele_fraction_nk + f1r2_fraction_nk
    return result_nk.masked_fill(impossible_states(ref_base, alt_bases_n), float("-inf"))


def responsibilities(log_likelihoods_nk: Tensor, prior_k: Tensor, given_not_hom_ref: bool = False) -> Tensor:
    """the posterior probabilities of each site's states, given their log likelihoods and the prior"""
    log_posteriors_nk = log_likelihoods_nk + torch.log(prior_k)
    if given_not_hom_ref:
        log_posteriors_nk[..., HOM_REF] = float("-inf")
    return torch.softmax(log_posteriors_nk, dim=-1)


def flat_prior(ref_base: int) -> Tensor:
    """the prior that is uniform over the possible states of a context with the given ref base"""
    possible_k = STATE_ALT_BASES_K != ref_base
    return possible_k.to(torch.float64) / possible_k.sum()


@dataclass
class ContextData:
    """
    The sites of a canonical reference context and its reverse complement, as distinct combinations of alt base (an
    index into BASES), alt count, count of alt reads in F1R2 orientation and depth, with the number of sites of each.
    Sites of the reverse complement are in the orientation of the canonical context, with their alt bases complemented
    and their F1R2 and F2R1 alt reads swapped.  Sites without alt reads have the ref base as their alt base.
    """

    context: str
    alt_bases_n: Tensor
    alt_counts_n: Tensor
    alt_f1r2_counts_n: Tensor
    depths_n: Tensor
    num_sites_n: Tensor
    num_ref_sites: int
    # sites with one alt read, from the histograms, and with two or more, from the alt tables
    num_alt_depth_one_sites: int
    num_multiple_alt_sites: int

    def ref_base(self) -> int:
        return BASES.index(self.context[1])

    def num_alt_sites(self) -> int:
        return self.num_alt_depth_one_sites + self.num_multiple_alt_sites


def _depth_histogram_sites(alt_base: int, alt_count: int, alt_f1r2_count: int, histogram_b: np.ndarray) -> AltSites:
    """the sites of a depth histogram, which all have the given alt base and counts"""
    depths_b = np.arange(1, len(histogram_b) + 1)
    constant = (np.full(len(histogram_b), value) for value in (alt_base, alt_count, alt_f1r2_count))
    return AltSites(*constant, depths_b, histogram_b)


def _reverse_complement_sites(sites: AltSites) -> AltSites:
    """
    sites of the reverse complement of a context in the orientation of the context: the F1R2 reads of one are the F2R1
    reads of the other, and the alt bases are complementary
    """
    complement_n = np.array([BASES.index(COMPLEMENTS[base]) for base in BASES])[sites.alt_bases_n]
    return AltSites(
        complement_n,
        sites.alt_counts_n,
        sites.alt_counts_n - sites.alt_f1r2_counts_n,
        sites.depths_n,
        sites.num_sites_n,
    )


def context_data(counts: F1R2Counts, context: str) -> ContextData:
    """the sites of a canonical context and its reverse complement in a sample's F1R2 counts"""
    assert context in CANONICAL_CONTEXTS
    ref_base = BASES.index(context[1])
    pieces, num_sites = [], {"ref": 0, "depth_one": 0, "multiple": 0}
    for this_context, is_reverse_complement in ((context, False), (reverse_complement(context), True)):
        context_pieces = [
            _depth_histogram_sites(BASES.index(this_context[1]), 0, 0, counts.ref_histograms[this_context])
        ]
        num_sites["ref"] += int(counts.ref_histograms[this_context].sum())
        for alt_base in BASES:
            if alt_base == this_context[1]:
                continue
            for orientation in ORIENTATIONS:
                histogram_b = counts.alt_depth_one_histograms[(this_context, alt_base, orientation)]
                alt_f1r2_count = 1 if orientation == "F1R2" else 0
                context_pieces.append(_depth_histogram_sites(BASES.index(alt_base), 1, alt_f1r2_count, histogram_b))
                num_sites["depth_one"] += int(histogram_b.sum())
        context_pieces.append(counts.alt_sites[this_context])
        num_sites["multiple"] += counts.alt_sites[this_context].num_sites()
        pieces += (
            [_reverse_complement_sites(piece) for piece in context_pieces] if is_reverse_complement else context_pieces
        )

    combined = AltSites.collapse(*(np.concatenate(columns) for columns in zip(*(piece.columns() for piece in pieces))))
    has_sites = combined.num_sites_n > 0
    tensors = (torch.from_numpy(column[has_sites]) for column in combined.columns()[:4])
    alt_bases_n, alt_counts_n, alt_f1r2_counts_n, depths_n = tensors
    assert torch.all(alt_bases_n[alt_counts_n == 0] == ref_base)
    num_sites_n = torch.from_numpy(combined.num_sites_n[has_sites]).to(torch.float64)
    return ContextData(
        context, alt_bases_n, alt_counts_n, alt_f1r2_counts_n, depths_n, num_sites_n,
        num_sites["ref"], num_sites["depth_one"], num_sites["multiple"],
    )  # fmt: skip


@dataclass
class ContextPrior:
    """
    The prior probabilities of the states of a reference context, by index, and the number of sites, and of alt sites,
    from which they were learned, both 0 for the flat prior of a context without data
    """

    context: str
    prior_k: Tensor
    num_sites: int
    num_alt_sites: int

    def reverse_complement(self) -> "ContextPrior":
        prior_k = self.prior_k[REVERSE_COMPLEMENT_STATES_K]
        return ContextPrior(reverse_complement(self.context), prior_k, self.num_sites, self.num_alt_sites)


@dataclass
class EMResult:
    prior_k: Tensor
    num_iterations: int
    converged: bool


def learn_prior(data: ContextData, convergence_threshold: float = 1e-4, max_iterations: int = 20) -> EMResult:
    """
    The maximum a posteriori prior of a context's states, by EM as in GATK: the prior starts flat and is updated
    until it changes by at most the convergence threshold (Euclidean distance), or for at most max_iterations.  Each
    update adds the flat prior to the expected numbers of sites in each state as pseudocounts.
    """
    ref_base = data.ref_base()
    log_likelihoods_nk = log_likelihoods(
        ref_base, data.alt_bases_n, data.alt_counts_n, data.alt_f1r2_counts_n, data.depths_n
    )
    pseudocounts_k = flat_prior(ref_base)
    prior_k = pseudocounts_k
    for iteration in range(1, max_iterations + 1):
        expected_counts_k = data.num_sites_n @ responsibilities(log_likelihoods_nk, prior_k)
        new_prior_k = (expected_counts_k + pseudocounts_k) / (expected_counts_k + pseudocounts_k).sum()
        distance = torch.linalg.vector_norm(new_prior_k - prior_k).item()
        prior_k = new_prior_k
        if distance <= convergence_threshold:
            return EMResult(prior_k, iteration, converged=True)
    return EMResult(prior_k, max_iterations, converged=False)


def learn_priors(
    counts: F1R2Counts, convergence_threshold: float = 1e-4, max_iterations: int = 20
) -> dict[str, ContextPrior]:
    """
    The prior of every context of a sample, by context.  As in GATK, a canonical context and its reverse complement keep
    the flat prior if they have no sites without alt reads or no sites with two or more alt reads.
    """
    result = {}
    for context in CANONICAL_CONTEXTS:
        data = context_data(counts, context)
        description = (
            f"Sample {counts.sample}, context {context}: {data.num_ref_sites} ref and {data.num_alt_sites()} alt sites"
        )
        if data.num_ref_sites == 0 or data.num_multiple_alt_sites == 0:
            print(f"{description}, too few to learn from.  Using the flat prior.")
            prior = ContextPrior(context, flat_prior(data.ref_base()), 0, 0)
        else:
            em = learn_prior(data, convergence_threshold, max_iterations)
            outcome = f"converged in {em.num_iterations}" if em.converged else f"did not converge in {max_iterations}"
            print(f"{description}, EM {outcome} iterations")
            prior = ContextPrior(context, em.prior_k, data.num_ref_sites + data.num_alt_sites(), data.num_alt_sites())
        result[context] = prior
        result[reverse_complement(context)] = prior.reverse_complement()
    return result
