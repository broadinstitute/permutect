"""
Permutect's read orientation model: a generative model of the reads at a site, in which damage to one strand of the
DNA before sequencing, such as deamination of cytosine in formalin-fixed (FFPE) samples or oxidation of guanine, gives
alt reads that come from that strand alone.  learn_read_orientation_model learns its parameters for a sample from the
F1R2 counts that Mutect2 collects, and filter_variants uses them in the likelihood that a variant's alt reads are
errors rather than a real variant.

READ ORIENTATION

A read pair whose read 1 maps to the forward strand of the reference is F1R2, and one whose read 1 maps to the reverse
strand is F2R1.  Read 1 starts at the 5' end of the original strand of DNA that the pair descends from, so F1R2 pairs
come from the + strand (the reference's) of the original double-stranded fragment and F2R1 pairs from its - strand.
The reads of a site therefore fall into two classes, by original strand:

    n1 reads in F1R2 orientation, a1 of them with the alt base
    n2 reads in F2R1 orientation, a2 of them with the alt base

with depth d = n1 + n2 and alt count a = a1 + a2.

THE GENERATIVE MODEL

Take a site whose + strand reads the 3-base reference context c, eg ACG, and the alt base x.

1. Capture.  Hybrid capture with baits of one strand of the reference recovers the two original strands unequally, by
   an amount that varies from site to site and library to library: in SEQC2's FFPE exomes the F1R2 fraction is 0.15 to
   0.46 on target, while in a fresh exome it is 0.50, as it is at sites more than a fragment's length from any bait.
   The site's number of F1R2 reads is

       n1 ~ BetaBinomial(d; alpha, beta),

   the sample's capture bias (see permutect.read_orientation.capture_bias), in the frame of the reference.  Capture
   doesn't see alleles, so this is the same whatever the site's genotype, and whatever damage gives some of its reads
   the alt base.

2. Damage.  Every strand of every site is damaged, at some rate.  Damage is a chemical event on one strand.  That strand
   reads its own sequence, 5' to 3', and knows nothing of which strand the reference calls forward, so a physical
   damage process is named by the damaged strand's own context and the base the damage makes it read: c -> x, eg
   ACG -> ATG, the deamination of the C of ACG on whichever strand reads ACG.  At a site of c the + strand, which reads
   c, can be damaged by c -> x, and the - strand, which reads rc(c), the reverse complement of c, by rc(c) -> rc(x),
   which would make the + strand read x.  For example, at a site of ACG with alt T these are ACG -> ATG, deamination of
   the + strand's C, and CGT -> CAT, a G -> A of the - strand, while at a site of CGT with alt A, the - strand's
   ACG -> ATG is the same physical process as the + strand's at ACG.

   Each of the 192 physical processes (64 contexts, 3 bases each) has its own distribution of the rate r at which a
   site's molecules of that strand are damaged, independent of every other's:

       r ~ Beta(mean mu, concentration kappa), independently at each site, for each strand.

   In particular, ACG -> ATG and its reverse complement CGT -> CAT are different chemistry.  The rate is per molecule:
   a site's reads come from many molecules, mostly from different cells, so its rate is an average over them, which
   varies continuously from site to site, rather than "damaged" or "not".  Misreading that depends on a strand's
   sequence is in the rate too: these counts can't tell it from damage, and needn't.

3. Genotype.  The site is HOM_REF, GERMLINE_HET, SOMATIC_HET or HOM_VAR, with prior probabilities learned for the
   sample.  A context and its reverse complement share these priors, since a real variant is the same event seen from
   either strand.

4. Alt reads.

   - At a HOM_REF site, the alt reads are damage, on each strand at its own rate:

         a1 ~ Binomial(n1, r of c -> x) and a2 ~ Binomial(n2, r of rc(c) -> rc(x)),

     that is, a1 ~ BetaBinomial(n1; c -> x) and a2 ~ BetaBinomial(n2; rc(c) -> rc(x)).  A read orientation artifact
     is a HOM_REF site whose alt reads came from damage at a high rate on one strand.

   - At a real variant, one allele fraction f, drawn for its genotype, applies to the reads of both strands:

         a1 ~ Binomial(n1, f) and a2 ~ Binomial(n2, f).

     f is Beta(2, 5) for SOMATIC_HET and Beta(10000, 3) for HOM_VAR, as in GATK.  For GERMLINE_HET it is that of the
     alt allele's haplotype, which is the minor or the major one with probability 1/2: given segment_pileups'
     GermlineModel, its emission model at the minor allele fraction (MAF) of the site's segment, and otherwise
     Beta(5, 5).  The two strands of a site have the same haplotypes, so the choice of haplotype is one for both
     classes of reads.  Because f is shared, the likelihood factors as

         P(a1, a2 | n1, n2, genotype) = P(a | d, genotype) x Hypergeometric(a1; n1, n2, a),

     whose second factor, how the alt reads split between the classes given the split of all the reads, is the same
     for every real variant.  Damage to a real variant's ref molecules, small next to f, is left out.

   - Reads of any other base, at any site, are damage too.

WHAT IS OBSERVED

Mutect2's alt tables give (a1, a2, n1, n2) of each site with two or more reads of its most common alt base, but not
the counts of other bases.  Its histograms give the depth of each site without alt reads, and of each site with one alt
read, that read's base and orientation, but not n1, so their likelihoods are marginals over n1 ~ BetaBinomial(d; alpha,
beta).  For a real variant this marginal is P(a | d, genotype) x BetaBinomial(a1; a, alpha, beta).

The data of a site split by base: the reads of its alt base, whose likelihood depends on its genotype, and, at the
histogram sites, the absence of reads of every other base, which is damage's whatever the genotype.  Each is
marginalized over n1 on its own, which ignores that they share n1: this matters little when a site has at most one alt
read.

Where n1 is known, P(n1 | d) multiplies every genotype's likelihood and drops out of the posteriors, so the capture bias
matters only for the histograms' sites, most of the sites.

LEARNING

The capture bias is fit first (see capture_bias), then the genotype priors and damage distributions by maximum a
posteriori expectation maximization (EM), all contexts at once:

- E step: the posterior probability of each site's genotypes, given the reads of its alt base.
- M step for the genotype priors: for each pair of a context and its reverse complement, the expected numbers of its
  sites with each genotype, plus GATK's flat pseudocounts, 1/4 each.
- M step for the damage distributions: each process's Beta maximizes the expected log likelihood of the reads it
  explains plus its log prior: alt reads at HOM_REF sites, weighted by their posterior probability, at sites of c on
  the + strand and sites of rc(c) on the - strand, and the absent reads of other bases at the histogram sites.  The
  prior is hierarchical, in the unconstrained coordinates
      u = (logit(mu / MAX_DAMAGE_MEAN), log(kappa - MIN_DAMAGE_CONCENTRATION)),
  u ~ Normal(m, diag(s^2)), so that processes with few damaged reads borrow strength from the others.  After each M
  step, m is the mean of the processes' u, and each component of s^2 has a weak inverse-gamma hyperprior and is
  updated to its posterior mode given the processes' expected squared deviations, which include their Laplace
  (posterior) variances, so that processes with little data don't collapse s^2 toward 0.  A site's reads of one base
  depend only on the damage of its two strands, so the M step's objective is a sum over the pairs of a process and its
  other strand's process, and Newton's method maximizes each pair's in its 4 coordinates.  The M step is exact: an
  approximate one, such as a few iterations of a general optimizer, makes successive EM iterations differ so little
  that EM seems to converge far from its fixed point.
- The bounds keep genotypes identifiable: a damage distribution's mean is at most MAX_DAMAGE_MEAN, so that damage
  can't imitate a het, and its concentration at least MIN_DAMAGE_CONCENTRATION, so that its density is finite near 1.

EM converges slowly, and SQUAREM (Varadhan and Roland 2008) accelerates it by extrapolating along its path.  As in GATK,
a pair of contexts without both sites without alt reads and sites with two or more keeps the flat prior, and EM stops
when no context's prior changes by more than a threshold (Euclidean distance) and either no process's u changes by
more than DAMAGE_TOLERANCE or the log posterior by more than SQUAREM_OBJECTIVE_TOLERANCE, or after a maximum number of
iterations.  Processes of such pairs, and processes that can't happen (to the base a context already has), get the
damage distribution at the mean m.

FILTERING

In filter_variants, the likelihood of an SNV's alt reads from damage at a HOM_REF site, relative to the hypergeometric
factor that real variants share, is mixed into the SEQ_ERROR call's likelihood with Mutect2's sequencing error
likelihood: the alt reads of a site without a variant come from one mechanism or the other.

Mutect2's VCF gives the orientations of only some of a variant's reads, those of base quality 20 or more, about two
thirds of them in SEQC2's exomes, while the other calls' likelihoods are of all its reads.  Of its d reads and a alt
reads, n1 and n2 have known orientations, a1 and a2 of them alt, and d_u = d - n1 - n2 don't, a_u = a - a1 - a2 of
them alt.  Damage's likelihood is then that of the oriented reads, as above, times Binomial(a_u; d_u, p), where p, the
probability that an unoriented read has damage, is q r1 + (1 - q) r2, with q the expected fraction of the site's reads
that are F1R2 given its oriented reads and the capture bias, and r1 and r2 the expected rates of damage to the + and -
strands given their oriented reads.  (This approximates the exact marginal by the rates' posterior means.)  A real
variant's shared factor is then the multivariate hypergeometric probability that its a alt reads split into a1, a2 and
a_u among its n1, n2 and d_u reads.

GATK's LearnReadOrientationModel instead makes an artifact a state of a site, with a prior for each context and
orientation, an allele fraction Beta(1, 9) of all of the site's reads, and alt reads oriented as Beta(100, 1); sites
without alt reads can't be artifacts.  Learned with damage as a state that sites without alt reads can be in too, every
process's distribution collapses toward the error rate, with a large prior: the data want damage at a low rate at most
sites, which is what the rates here say.  A non-artifact's alt reads there have orientation Beta(10, 10) regardless of
the site's other reads.

A ReadOrientationModel holds what is learned for a sample.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch import nn

from permutect.read_orientation.f1r2_counts_reader import BASES
from permutect.read_orientation.f1r2_counts_reader import CONTEXTS
from permutect.read_orientation.f1r2_counts_reader import UNKNOWN_REF_F1R2
from permutect.read_orientation.f1r2_counts_reader import F1R2Counts
from permutect.read_orientation.f1r2_counts_reader import Sites
from permutect.read_orientation.maf_mixtures import MAF_GRID
from permutect.segmentation.emissions import EmissionModel
from permutect.utils.math_utils import log_binomial_coefficient
from permutect.utils.stats_utils import beta_binomial_log_lk
from permutect.utils.stats_utils import binomial_log_lk

NUM_BASES, NUM_CONTEXTS = len(BASES), len(CONTEXTS)

GENOTYPES = ("HOM_REF", "GERMLINE_HET", "SOMATIC_HET", "HOM_VAR")
NUM_GENOTYPES = len(GENOTYPES)
HOM_REF, GERMLINE_HET, SOMATIC_HET, HOM_VAR = range(NUM_GENOTYPES)
# GATK's pseudocounts (alpha, beta) of the allele fraction of the real variants, GERMLINE_HET, SOMATIC_HET and HOM_VAR:
# germline hets have allele fractions broadly around 1/2 for copy number variation (without a germline model), somatic
# hets low allele fractions and hom var sites rare ref reads
VARIANT_ALLELE_FRACTION_PSEUDOCOUNTS_V2 = torch.tensor([(5.0, 5.0), (2.0, 5.0), (10000.0, 3.0)], dtype=torch.float64)
# the capture bias of a sample without the F1R2 counts of reads at sites with alt reads to fit it to: GATK's
# distribution of the F1R2 fraction of a non-artifact's alt reads
DEFAULT_CAPTURE_PSEUDOCOUNTS = (10.0, 10.0)
# GATK's flat prior, as pseudocounts of 1 / NUM_GENOTYPES of each genotype
FLAT_PRIOR = 1 / NUM_GENOTYPES

# Bounds and starting values of the damage distributions, and the weak inverse-gamma hyperprior, shape and scale, of
# each component of the variance s^2 of the hierarchical prior of their unconstrained coordinates
MAX_DAMAGE_MEAN = 0.25
MIN_DAMAGE_CONCENTRATION = 2.0
INITIAL_DAMAGE_MEAN = 1e-3
INITIAL_DAMAGE_CONCENTRATION = 100.0
INITIAL_DAMAGE_VARIANCE = 1.0
DAMAGE_VARIANCE_SHAPE = 1.0
DAMAGE_VARIANCE_SCALE = 0.5
# EM's tolerance of changes in the damage distributions' unconstrained coordinates
DAMAGE_TOLERANCE = 1e-3
# Newton's method in each M step of the damage distributions: its maximum number of iterations; its tolerance of the
# largest change in a coordinate of a step; the largest change in a coordinate that one step makes, a trust region; the
# largest change of the steps taken without a line search, whose comparisons of objectives, sums over millions of sites,
# rounding makes unreliable near the optimum; and the Armijo constant and maximum number of halvings of the line search of
# the other steps
DAMAGE_M_STEP_ITERATIONS = 20
NEWTON_TOLERANCE = 1e-4
MAX_NEWTON_STEP = 1.0
NEWTON_REGION = 0.1
ARMIJO_CONSTANT = 1e-4
MAX_STEP_HALVINGS = 10
# SQUAREM's tolerance of a loss of EM's objective, in nats; the factor by which its maximum extrapolation grows after an
# extrapolation at the maximum and shrinks after one that fails; and the change in the objective of a cycle that has
# converged
SQUAREM_TOLERANCE = 1.0
SQUAREM_STEP_FACTOR = 4.0
SQUAREM_OBJECTIVE_TOLERANCE = 0.1
# the number of a site's F1R2 reads where it isn't known, as for the sites of the histograms
UNKNOWN_F1R2_DEPTH = -1


def reverse_complement(sequence: str) -> str:
    return sequence[::-1].translate(str.maketrans(BASES, BASES[::-1]))


def context_indices(left_bases, ref_bases, right_bases):
    """the index into CONTEXTS of the contexts with the given bases, indices into BASES, which may be tensors"""
    return NUM_BASES * NUM_BASES * left_bases + NUM_BASES * ref_bases + right_bases


CONTEXT_INDICES = {context: n for n, context in enumerate(CONTEXTS)}
REF_BASES_C = torch.tensor([BASES.index(context[1]) for context in CONTEXTS])
REVERSE_COMPLEMENTS_C = torch.tensor([CONTEXT_INDICES[reverse_complement(context)] for context in CONTEXTS])
# One context of each pair of a context and its reverse complement, which are never the same, to name the pair, and the
# pair of each context
PAIR_CONTEXTS = tuple(context for context in CONTEXTS if context < reverse_complement(context))
NUM_PAIRS = len(PAIR_CONTEXTS)
PAIRS_C = torch.tensor([PAIR_CONTEXTS.index(min(context, reverse_complement(context))) for context in CONTEXTS])

# A physical damage process c -> x, by the damaged strand's context c and the base x it comes to read, has the index
# NUM_BASES * c + x, and can't happen if x is c's ref base.  Since the complement of the base with index b has index
# 3 - b, the process of the other strand at the same site, rc(c) -> rc(x), has the index NUM_BASES * rc(c) + 3 - x: the
# other strand's process of each process.
NUM_PROCESSES = NUM_CONTEXTS * NUM_BASES
POSSIBLE_PROCESSES_P = (torch.arange(NUM_BASES) != REF_BASES_C.unsqueeze(-1)).flatten()
OTHER_STRAND_PROCESSES_P = (
    NUM_BASES * REVERSE_COMPLEMENTS_C.unsqueeze(-1) + (NUM_BASES - 1 - torch.arange(NUM_BASES))
).flatten()


def process_name(process: int) -> str:
    """eg ACG -> ATG"""
    context, base = CONTEXTS[process // NUM_BASES], BASES[process % NUM_BASES]
    return f"{context} -> {context[0]}{base}{context[2]}"


def flat_priors_ws() -> Tensor:
    """GATK's flat prior of the genotypes of each pair of contexts"""
    return torch.full((NUM_PAIRS, NUM_GENOTYPES), FLAT_PRIOR, dtype=torch.float64)


# ---------------------------------------------------------------------------------------------------------------------
# the sites


@dataclass(frozen=True)
class SiteTable:
    """
    The sites of all contexts of a sample, as the distinct combinations of the columns of Sites, with each one's context
    (an index into CONTEXTS) and number n1 of F1R2 reads, ref and alt: the ref F1R2 count plus the alt F1R2 count, or
    UNKNOWN_F1R2_DEPTH for the sites of the histograms.  A site without alt reads has the context's ref base as its alt
    base.
    """

    contexts_n: Tensor
    alt_bases_n: Tensor
    alt_counts_n: Tensor
    alt_f1r2_counts_n: Tensor
    depths_n: Tensor
    maf_mixtures_n: Tensor
    f1r2_depths_n: Tensor
    counts_n: Tensor

    @staticmethod
    def of(sites_by_context: dict[str, Sites]) -> "SiteTable":
        columns = []
        for context, sites in sites_by_context.items():
            known_n = sites.ref_f1r2_counts_n != UNKNOWN_REF_F1R2
            f1r2_depths_n = torch.where(known_n, sites.ref_f1r2_counts_n + sites.alt_f1r2_counts_n, UNKNOWN_F1R2_DEPTH)
            contexts_n = torch.full_like(sites.counts_n, CONTEXT_INDICES[context])
            columns.append(
                torch.column_stack([contexts_n, sites.keys_n4, sites.maf_mixtures_n, f1r2_depths_n, sites.counts_n])
            )
        table = torch.cat(columns) if columns else torch.zeros(0, 8, dtype=torch.int64)
        return SiteTable(*table.T)

    def select(self, selected_n: Tensor) -> "SiteTable":
        return SiteTable(*(getattr(self, name)[selected_n] for name in self.__dataclass_fields__))

    def __len__(self) -> int:
        return len(self.counts_n)

    @property
    def known_n(self) -> Tensor:
        """whether each site's number of F1R2 reads is known"""
        return self.f1r2_depths_n != UNKNOWN_F1R2_DEPTH

    @property
    def alt_f2r1_counts_n(self) -> Tensor:
        return self.alt_counts_n - self.alt_f1r2_counts_n

    @property
    def f2r1_depths_n(self) -> Tensor:
        return self.depths_n - self.f1r2_depths_n

    @property
    def processes_n(self) -> Tensor:
        """the physical process of damage to each site's + strand that would give its alt base"""
        return NUM_BASES * self.contexts_n + self.alt_bases_n


# ---------------------------------------------------------------------------------------------------------------------
# likelihoods


def log_hypergeometric(alt_f1r2_counts, alt_f2r1_counts, f1r2_depths, f2r1_depths):
    """
    The log probability that a1 = alt_f1r2_counts of a = a1 + a2 alt reads are F1R2, drawn without replacement from
    n1 = f1r2_depths F1R2 and n2 = f2r1_depths F2R1 reads: the factor of a real variant's likelihood for how its alt
    reads split between the classes, given the split of all its reads
    """
    return (
        log_binomial_coefficient(f1r2_depths, alt_f1r2_counts)
        + log_binomial_coefficient(f2r1_depths, alt_f2r1_counts)
        - log_binomial_coefficient(f1r2_depths + f2r1_depths, alt_f1r2_counts + alt_f2r1_counts)
    )


class GermlineHetLikelihoods:
    """
    The log likelihoods of germline hets' alt counts from segment_pileups' emission model of the sample, which has its
    reference bias, contamination, error rate and concentration of hets, at the MAFs of the sites' MAF mixtures.  The
    alt allele is on the minor haplotype, whose fraction of the alleles is the MAF, or on the major haplotype, each with
    probability 1/2, the same haplotype for the reads of both strands.  The sites' population allele frequencies, which
    the contaminant's expected alt fraction is, are taken to be 0.
    """

    def __init__(self, emission_model: EmissionModel, maf_mixtures_mg: Tensor):
        self.emission_model = emission_model
        self.log_weights_mg = maf_mixtures_mg.log()

    def __call__(self, alt_counts_n: Tensor, depths_n: Tensor, maf_mixtures_n: Tensor) -> Tensor:
        mafs_1g = torch.tensor(MAF_GRID, dtype=torch.float64).unsqueeze(0)
        alt_counts_n1, depths_n1 = (x.double().unsqueeze(-1) for x in (alt_counts_n, depths_n))
        with torch.no_grad():
            _, minor_ng, major_ng, _ = self.emission_model.phased_genotype_log_likelihoods(
                alt_counts_n1, depths_n1, torch.zeros((), dtype=torch.float64), mafs_1g
            )
        het_ng = torch.logaddexp(minor_ng, major_ng) - math.log(2)
        return torch.logsumexp(het_ng + self.log_weights_mg[maf_mixtures_n], dim=-1)


def variant_log_likelihoods_nv(
    sites: SiteTable, capture_pseudocounts: tuple[float, float], germline_het: GermlineHetLikelihoods = None
) -> Tensor:
    """
    The log likelihood of each site's alt reads given each real variant's genotype, GERMLINE_HET, SOMATIC_HET and
    HOM_VAR, an (N, 3) tensor: P(a | d, genotype) times the hypergeometric probability of the split of the alt reads
    given n1, or, where n1 is unknown, its marginal under the capture bias, BetaBinomial(a1; a, alpha, beta).  Given
    germline_het, germline hets' allele fractions are its.
    """
    alt_counts_n, depths_n = sites.alt_counts_n.double(), sites.depths_n.double()
    alpha_v, beta_v = VARIANT_ALLELE_FRACTION_PSEUDOCOUNTS_V2.unbind(-1)
    result_nv = beta_binomial_log_lk(depths_n.unsqueeze(-1), alt_counts_n.unsqueeze(-1), alpha_v, beta_v)
    if germline_het is not None:
        result_nv[:, GERMLINE_HET - 1] = germline_het(sites.alt_counts_n, sites.depths_n, sites.maf_mixtures_n)
    capture_alpha, capture_beta = (torch.tensor(x, dtype=torch.float64) for x in capture_pseudocounts)
    split_n = beta_binomial_log_lk(alt_counts_n, sites.alt_f1r2_counts_n.double(), capture_alpha, capture_beta)
    known = sites.known_n
    counts = (sites.alt_f1r2_counts_n, sites.alt_f2r1_counts_n, sites.f1r2_depths_n, sites.f2r1_depths_n)
    split_n[known] = log_hypergeometric(*(x[known].double() for x in counts))
    return result_nv + split_n.unsqueeze(-1)


def damage_pseudocounts(unconstrained_q2: Tensor) -> tuple[Tensor, Tensor]:
    """the pseudocounts (alpha, beta) of damage distributions, given their unconstrained coordinates u"""
    mean_q = MAX_DAMAGE_MEAN * torch.sigmoid(unconstrained_q2[:, 0])
    concentration_q = MIN_DAMAGE_CONCENTRATION + torch.exp(unconstrained_q2[:, 1])
    return mean_q * concentration_q, (1 - mean_q) * concentration_q


def unconstrained_damage(mean: float, concentration: float) -> Tensor:
    """the unconstrained coordinates u of a damage distribution of the given mean and concentration"""
    return torch.tensor(
        [math.log(mean / (MAX_DAMAGE_MEAN - mean)), math.log(concentration - MIN_DAMAGE_CONCENTRATION)],
        dtype=torch.float64,
    )


def known_split_damage_log_likelihoods(
    alt_f1r2_counts, f1r2_depths, alt_f2r1_counts, f2r1_depths, plus_alpha, plus_beta, minus_alpha, minus_beta
):
    """
    The log likelihood of a site's alt reads from damage given its numbers of reads in each orientation: beta-binomial
    in the + strand's damage distribution for the F1R2 reads and in the - strand's for the F2R1 reads
    """
    return beta_binomial_log_lk(f1r2_depths, alt_f1r2_counts, plus_alpha, plus_beta) + beta_binomial_log_lk(
        f2r1_depths, alt_f2r1_counts, minus_alpha, minus_beta
    )


def beta_binomial_probabilities(depths: Tensor, alt_count: int, alpha: Tensor, beta: Tensor) -> Tensor:
    """BetaBinomial(alt_count; depth, alpha, beta), 0 where the depth is less than alt_count, finite everywhere"""
    within = depths >= alt_count
    log_lk = beta_binomial_log_lk(torch.clamp(depths, min=alt_count), torch.tensor(float(alt_count)), alpha, beta)
    return log_lk.exp() * within


class HistogramMarginals:
    """
    The likelihoods of damage's reads of one base at the sites of the histograms, which have at most one alt read and
    unknown n1:

        P(k1, k2 | d) = sum over n1 of P(n1 | d) BetaBinomial(k1; n1, + strand) BetaBinomial(k2; d - n1, - strand),

    with n1 ~ BetaBinomial(d; alpha, beta) of the capture bias, for the patterns (k1, k2) of no reads of the base, one
    F1R2 read and one F2R1 read, and the processes of the + and - strands of a site.
    """

    # (k1, k2): none, one F1R2 read, one F2R1 read
    PATTERNS = ((0, 0), (1, 0), (0, 1))

    def __init__(self, depths_d: Tensor, capture_pseudocounts: tuple[float, float]):
        max_depth = int(depths_d.max()) if len(depths_d) else 0
        self.m_m = torch.arange(max_depth + 1, dtype=torch.float64)
        d_d1 = depths_d.double().unsqueeze(-1)
        # P(n1 = m | d), 0 where m > d
        capture_alpha, capture_beta = (torch.tensor(x, dtype=torch.float64) for x in capture_pseudocounts)
        within_dm = self.m_m <= d_d1
        self.capture_dm = (
            beta_binomial_log_lk(d_d1, torch.minimum(self.m_m, d_d1), capture_alpha, capture_beta).exp() * within_dm
        )
        # the index of n2 = d - m among the m_m, where it is one
        self.other_depths_dm = (d_d1 - self.m_m).clamp_min(0).long()

    def log_likelihoods_tqd(
        self, plus_alpha_q: Tensor, plus_beta_q: Tensor, minus_alpha_q: Tensor, minus_beta_q: Tensor
    ) -> Tensor:
        """
        The log likelihoods of the patterns given the damage distributions of the + and - strands of Q kinds of site, at
        the histograms' depths, by pattern, kind of site and depth, a (len(PATTERNS), Q, D) tensor
        """
        plus_kqm, minus_kqm = (
            torch.stack(
                [beta_binomial_probabilities(self.m_m, k, alpha.unsqueeze(-1), beta.unsqueeze(-1)) for k in (0, 1)]
            )
            for alpha, beta in ((plus_alpha_q, plus_beta_q), (minus_alpha_q, minus_beta_q))
        )
        result = []
        for k1, k2 in self.PATTERNS:
            # the - strand's probability of k2 reads among its d - m, by kind, depth and m
            minus_qdm = minus_kqm[k2][:, self.other_depths_dm]
            result.append(torch.sum(self.capture_dm * plus_kqm[k1].unsqueeze(1) * minus_qdm, dim=-1))
        return torch.stack(result).clamp_min(1e-300).log()


# ---------------------------------------------------------------------------------------------------------------------
# learning


@dataclass
class EMResult:
    priors_ws: Tensor
    damage_unconstrained_p2: Tensor
    hyper_mean_2: Tensor
    hyper_variance_2: Tensor
    num_iterations: int
    converged: bool


@dataclass(frozen=True)
class EMState:
    """
    What an iteration of EM updates: the genotype priors of each pair of contexts, the unconstrained damage coordinates
    of the learned processes, and the mean and variance of their hierarchical prior
    """

    priors_ws: Tensor
    unconstrained_q2: Tensor
    hyper_mean_2: Tensor
    hyper_variance_2: Tensor


class OrientationEM:
    """
    The data and the steps of EM for the sites of the pairs of contexts with enough data to learn from (see the module
    docstring).  The learned processes are numbered 0, 1, . . . Q - 1.  A site's reads of one base are explained by the
    damage of a + strand process, which with its other strand's process gives the site's "kind", numbered by its +
    strand process.
    """

    def __init__(
        self,
        sites: SiteTable,
        learned_pairs_w: Tensor,
        capture_pseudocounts: tuple[float, float],
        germline_het: GermlineHetLikelihoods = None,
    ):
        sites = sites.select(learned_pairs_w[PAIRS_C[sites.contexts_n]])
        learned_p = POSSIBLE_PROCESSES_P & learned_pairs_w[PAIRS_C].repeat_interleave(NUM_BASES)
        self.learned_processes_q = learned_p.nonzero().flatten()
        self.numbers_p = torch.full((NUM_PROCESSES,), -1, dtype=torch.int64)
        self.numbers_p[self.learned_processes_q] = torch.arange(len(self.learned_processes_q))
        # the other strand's process of each, which is learned too, since it's of the same pair of contexts
        self.other_strand_q = self.numbers_p[OTHER_STRAND_PROCESSES_P[self.learned_processes_q]]
        # The objective of the damage M step is a sum over pairs of a process and its other strand's process, numbered
        # 0, 1, . . . W - 1: each pair's first process, of the lower number, and its second, and each process's pair
        self.firsts_w = (torch.arange(len(self.other_strand_q)) < self.other_strand_q).nonzero().flatten()
        self.seconds_w = self.other_strand_q[self.firsts_w]
        self.pairs_q = torch.zeros(len(self.other_strand_q), dtype=torch.int64)
        self.pairs_q[self.firsts_w] = torch.arange(len(self.firsts_w))
        self.pairs_q[self.seconds_w] = torch.arange(len(self.firsts_w))
        # the pairs of contexts whose genotype priors are learned
        self.learned_w = torch.zeros(NUM_PAIRS, dtype=torch.bool)
        self.learned_w[PAIRS_C[self.learned_processes_q // NUM_BASES]] = True

        self.histogram, self.alt_table = sites.select(~sites.known_n), sites.select(sites.known_n)
        self.has_alt_reads_n = self.histogram.alt_counts_n > 0
        self.variant_nv = [
            variant_log_likelihoods_nv(part, capture_pseudocounts, germline_het)
            for part in (self.histogram, self.alt_table)
        ]

        # The histogram sites' damage likelihoods come from a table by pattern, kind and depth.  A site with one alt read
        # has the pattern of its orientation and the kind of its alt base.  Reads of other bases, which aren't there,
        # have pattern 0 at the kinds of every other base, whatever the site's genotype, so their weights are fixed.
        h = self.histogram
        depths_d, depth_indices_n = torch.unique(h.depths_n, return_inverse=True)
        self.marginals = HistogramMarginals(depths_d, capture_pseudocounts)
        self.num_depths, num_kinds = len(depths_d), len(self.learned_processes_q)
        patterns_n = torch.where(h.alt_f1r2_counts_n == 1, 1, 2)
        kinds_n = self.numbers_p[h.processes_n].clamp_min(0)
        self.histogram_kinds_n = kinds_n
        self.alt_cells_n = (patterns_n * num_kinds + kinds_n) * self.num_depths + depth_indices_n
        bases_1b = torch.arange(NUM_BASES).unsqueeze(0)
        absent_nb = (bases_1b != REF_BASES_C[h.contexts_n].unsqueeze(-1)) & (
            ~self.has_alt_reads_n.unsqueeze(-1) | (bases_1b != h.alt_bases_n.unsqueeze(-1))
        )
        absent_kinds_nb = self.numbers_p[NUM_BASES * h.contexts_n.unsqueeze(-1) + bases_1b]
        absent_cells_nb = absent_kinds_nb.clamp_min(0) * self.num_depths + depth_indices_n.unsqueeze(-1)
        num_cells = len(HistogramMarginals.PATTERNS) * num_kinds * self.num_depths
        self.absent_cell_weights = torch.zeros(num_cells, dtype=torch.float64).index_add_(
            0, absent_cells_nb[absent_nb], h.counts_n.double().unsqueeze(-1).expand_as(absent_nb)[absent_nb]
        )

        # an alt table site's alt reads, of known orientation, are damage of its kind given HOM_REF
        a = self.alt_table
        self.alt_kinds_n = self.numbers_p[a.processes_n]
        self.alt_counts_4n = torch.stack(
            [x.double() for x in (a.alt_f1r2_counts_n, a.f1r2_depths_n, a.alt_f2r1_counts_n, a.f2r1_depths_n)]
        )

    def damage_log_likelihoods(self, unconstrained_q2: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        The log likelihoods of damage, given the damage distributions' unconstrained coordinates: the flattened table of
        the histogram sites' patterns, and the log likelihoods of the alt reads of each histogram site, 0 for those
        without alt reads, and of each alt table site given HOM_REF
        """
        alpha_q, beta_q = damage_pseudocounts(unconstrained_q2)
        other_q = self.other_strand_q
        table = self.marginals.log_likelihoods_tqd(alpha_q, beta_q, alpha_q[other_q], beta_q[other_q]).flatten()
        histogram_n = torch.where(self.has_alt_reads_n, table[self.alt_cells_n], 0)
        kinds_n = self.alt_kinds_n
        alt_n = known_split_damage_log_likelihoods(
            *self.alt_counts_4n,
            alpha_q[kinds_n],
            beta_q[kinds_n],
            alpha_q[other_q[kinds_n]],
            beta_q[other_q[kinds_n]],
        )
        return table, histogram_n, alt_n

    def e_step(self, priors_ws: Tensor, unconstrained_q2: Tensor) -> tuple[Tensor, Tensor]:
        """the posterior probabilities of the genotypes of the histogram and alt table sites"""
        with torch.no_grad():
            _, histogram_n, alt_n = self.damage_log_likelihoods(unconstrained_q2)
        result = []
        for part, hom_ref_n, variant_nv in zip((self.histogram, self.alt_table), (histogram_n, alt_n), self.variant_nv):
            log_lks_ng = torch.cat([hom_ref_n.unsqueeze(-1), variant_nv], dim=-1)
            result.append(torch.softmax(log_lks_ng + priors_ws[PAIRS_C[part.contexts_n]].log(), dim=-1))
        return tuple(result)

    def priors_m_step(self, posteriors) -> Tensor:
        """the genotype prior of each pair: its expected counts of sites plus the flat pseudocounts, normalized"""
        counts_ws = flat_priors_ws()
        for part, posteriors_ng in zip((self.histogram, self.alt_table), posteriors):
            counts_ws.index_add_(0, PAIRS_C[part.contexts_n], part.counts_n.double().unsqueeze(-1) * posteriors_ng)
        return counts_ws / counts_ws.sum(dim=-1, keepdim=True)

    def pair_objectives(self, unconstrained_q2: Tensor, posteriors, hyper_mean_2, hyper_variance_2) -> Tensor:
        """
        The objective of the damage M step of each pair of a process and its other strand's process: the negative of
        the expected log likelihood of the reads that damage explains, plus the log prior of the damage distributions,
        up to a constant: alt reads at HOM_REF sites, weighted by their posterior probability, and the absent reads of
        other bases at the histogram sites.  A site's reads of one base depend on the damage distributions of its kind
        and of its kind's other strand's process, so the objective is the sum of the pairs'.
        """
        table, histogram_n, alt_n = self.damage_log_likelihoods(unconstrained_q2)
        histogram_hom_ref_n, alt_hom_ref_n = (posteriors_ng[:, HOM_REF] for posteriors_ng in posteriors)
        num_kinds = len(unconstrained_q2)
        absent_tqd = (self.absent_cell_weights * table).reshape(len(HistogramMarginals.PATTERNS), num_kinds, -1)
        log_lk_q = absent_tqd.sum(dim=(0, 2))
        log_lk_q = log_lk_q.index_add(
            0, self.histogram_kinds_n, self.histogram.counts_n.double() * histogram_hom_ref_n * histogram_n
        )
        log_lk_q = log_lk_q.index_add(0, self.alt_kinds_n, self.alt_table.counts_n.double() * alt_hom_ref_n * alt_n)
        log_prior_q = -torch.sum((unconstrained_q2 - hyper_mean_2) ** 2 / (2 * hyper_variance_2), dim=-1)
        objectives_q = -(log_lk_q + log_prior_q)
        return torch.zeros(len(self.firsts_w), dtype=objectives_q.dtype).index_add(0, self.pairs_q, objectives_q)

    def objectives_gradient_and_hessian_blocks(self, unconstrained_q2, posteriors, hyper_mean_2, hyper_variance_2):
        """
        The pairs' objectives, the gradient of the objective, and the 4 x 4 block of its Hessian of each pair, at the
        given unconstrained coordinates, a (W, 4, 4) tensor in the order of the first process's coordinates, then the
        second's.  The objective couples each process only with its other strand's process, so its Hessian is block
        diagonal.
        """
        u = unconstrained_q2.detach().requires_grad_(True)
        objectives_w = self.pair_objectives(u, posteriors, hyper_mean_2, hyper_variance_2)
        (gradient,) = torch.autograd.grad(objectives_w.sum(), u, create_graph=True)
        # The Hessian times a vector of 1s in component j of every pair's first (or second) process gives column j of the
        # first (second) process's block in every pair's rows.
        first_q = torch.zeros(len(u), dtype=torch.bool)
        first_q[self.firsts_w] = True
        own_q22 = torch.zeros(len(u), 2, 2, dtype=torch.float64)  # [q, i, j]: d2 / du_qi du_qj
        cross_q22 = torch.zeros_like(own_q22)  # [q, i, j]: d2 / du_qi du_oj, o the other strand's process of q
        for role_q in (first_q, ~first_q):
            for j in range(2):
                vector_q2 = torch.zeros_like(u)
                vector_q2[role_q, j] = 1
                (column_q2,) = torch.autograd.grad(gradient, u, grad_outputs=vector_q2, retain_graph=True)
                own_q22[role_q, :, j] = column_q2[role_q].detach()
                cross_q22[~role_q, :, j] = column_q2[~role_q].detach()
        # each pair's 4 x 4 block [[A, B], [B^T, C]], with A the first process's
        a, b, c = own_q22[self.firsts_w], cross_q22[self.firsts_w], own_q22[self.seconds_w]
        block_w44 = torch.cat([torch.cat([a, b], -1), torch.cat([b.transpose(1, 2), c], -1)], 1)
        return objectives_w.detach(), gradient.detach(), block_w44

    def damage_m_step(self, unconstrained_q2, posteriors, hyper_mean_2, hyper_variance_2) -> tuple[Tensor, Tensor]:
        """
        The maximum a posteriori damage distributions' unconstrained coordinates, by Newton's method from the current
        ones, and their Laplace approximations' variances, the diagonals of the inverse of the Hessian of the objective.
        The objective is a sum over pairs of a process and its other strand's process (see pair_objectives), so each pair
        takes its own Newton step, in its own 4 coordinates, with its 4 x 4 block of the Hessian, whose eigenvalues are
        replaced by their absolute values so that the step descends where the objective isn't convex, and its own
        backtracking line search.
        """
        firsts_w, seconds_w = self.firsts_w, self.seconds_w
        u = unconstrained_q2.detach().clone()
        for _ in range(DAMAGE_M_STEP_ITERATIONS):
            objectives_w, gradient_q2, block_w44 = self.objectives_gradient_and_hessian_blocks(
                u, posteriors, hyper_mean_2, hyper_variance_2
            )
            gradient_w4 = torch.cat([gradient_q2[firsts_w], gradient_q2[seconds_w]], dim=-1)
            eigenvalues_w4, vectors_w44 = torch.linalg.eigh(block_w44)
            floor_w1 = 1e-10 * eigenvalues_w4.abs().amax(dim=-1, keepdim=True).clamp_min(1e-300)
            magnitudes_w4 = torch.maximum(eigenvalues_w4.abs(), floor_w1)
            projections_w4 = (vectors_w44.transpose(1, 2) @ gradient_w4.unsqueeze(-1)).squeeze(-1)
            step_w4 = -(vectors_w44 @ (projections_w4 / magnitudes_w4).unsqueeze(-1)).squeeze(-1)
            largest_w = step_w4.abs().amax(dim=-1)
            if largest_w.max() <= NEWTON_TOLERANCE:
                break
            # a trust region: in nearly flat directions, eg a concentration that the data hardly constrain, Newton's step
            # can be huge, so it is scaled to change no coordinate by more than MAX_NEWTON_STEP
            step_w4 = step_w4 * (MAX_NEWTON_STEP / largest_w).clamp(max=1).unsqueeze(-1)

            # pairs near their optimum take the full step; the others halve theirs until their objective decreases by
            # the Armijo condition
            trusted_w = (eigenvalues_w4[:, 0] > 0) & (largest_w <= NEWTON_REGION)
            slopes_w = (gradient_w4 * step_w4).sum(dim=-1)
            fractions_w = torch.ones(len(firsts_w), dtype=torch.float64)
            new_u = u.clone()
            new_u[firsts_w[trusted_w]] += step_w4[trusted_w, :2]
            new_u[seconds_w[trusted_w]] += step_w4[trusted_w, 2:]
            pending_w = ~trusted_w
            for _ in range(MAX_STEP_HALVINGS if pending_w.any() else 0):
                trial_w4 = fractions_w.unsqueeze(-1) * step_w4
                trial = u.clone()
                trial[firsts_w] += trial_w4[:, :2]
                trial[seconds_w] += trial_w4[:, 2:]
                with torch.no_grad():
                    trial_objectives_w = self.pair_objectives(trial, posteriors, hyper_mean_2, hyper_variance_2)
                accepted_w = pending_w & (trial_objectives_w <= objectives_w + ARMIJO_CONSTANT * fractions_w * slopes_w)
                new_u[firsts_w[accepted_w]] = trial[firsts_w[accepted_w]]
                new_u[seconds_w[accepted_w]] = trial[seconds_w[accepted_w]]
                pending_w &= ~accepted_w
                if not pending_w.any():
                    break
                fractions_w = torch.where(pending_w, fractions_w / 2, fractions_w)
            if torch.equal(new_u, u):
                break  # no pair can improve, which happens only where rounding hides the decrease
            u = new_u
        else:
            # the loop ran out of iterations after a step, so the Hessian for the variances is at the final coordinates
            _, _, block_w44 = self.objectives_gradient_and_hessian_blocks(u, posteriors, hyper_mean_2, hyper_variance_2)

        valid_w = torch.linalg.eigvalsh(block_w44)[:, 0] > 0
        diagonal_w4 = torch.diagonal(torch.linalg.inv(block_w44[valid_w]), dim1=-2, dim2=-1)
        variances_q2 = hyper_variance_2.expand_as(u).clone()
        variances_q2[firsts_w[valid_w]] = diagonal_w4[:, :2]
        variances_q2[seconds_w[valid_w]] = diagonal_w4[:, 2:]
        return u, variances_q2

    @staticmethod
    def hyperparameters(unconstrained_q2: Tensor, variances_q2: Tensor) -> tuple[Tensor, Tensor]:
        """
        The mean of the hierarchical prior, the mean of the processes' coordinates, and its variance, the posterior mode
        of each component under its inverse-gamma hyperprior, given the processes' expected squared deviations
        """
        mean_2 = unconstrained_q2.mean(dim=0)
        squares_2 = ((unconstrained_q2 - mean_2) ** 2 + variances_q2).sum(dim=0)
        num = len(unconstrained_q2)
        return mean_2, (squares_2 + 2 * DAMAGE_VARIANCE_SCALE) / (num + 2 * DAMAGE_VARIANCE_SHAPE + 2)

    def initial_state(self) -> EMState:
        """EM's starting point: flat priors, and every process's damage distribution at the hierarchical prior's mean"""
        hyper_mean_2 = unconstrained_damage(INITIAL_DAMAGE_MEAN, INITIAL_DAMAGE_CONCENTRATION)
        hyper_variance_2 = torch.full((2,), INITIAL_DAMAGE_VARIANCE, dtype=torch.float64)
        u_q2 = hyper_mean_2.expand(len(self.learned_processes_q), 2).clone()
        return EMState(flat_priors_ws(), u_q2, hyper_mean_2, hyper_variance_2)

    def em_map(self, state: EMState) -> EMState:
        """one iteration of EM"""
        posteriors = self.e_step(state.priors_ws, state.unconstrained_q2)
        new_priors_ws = self.priors_m_step(posteriors)
        new_u_q2, variances_q2 = self.damage_m_step(
            state.unconstrained_q2, posteriors, state.hyper_mean_2, state.hyper_variance_2
        )
        return EMState(new_priors_ws, new_u_q2, *self.hyperparameters(new_u_q2, variances_q2))

    def log_posterior(self, state: EMState) -> float:
        """
        What EM increases, up to a constant: the log likelihood of the sites, each a sum over its genotypes, and of the
        absent reads, plus the log priors of the damage coordinates under the hierarchical prior, of the hierarchical
        prior's variances under their inverse-gamma hyperprior, and of the learned genotype priors under the flat
        pseudocounts
        """
        u_q2, mean_2, variance_2 = state.unconstrained_q2, state.hyper_mean_2, state.hyper_variance_2
        with torch.no_grad():
            table, histogram_n, alt_n = self.damage_log_likelihoods(u_q2)
            log_lk = torch.sum(self.absent_cell_weights * table)
            for part, hom_ref_n, variant_nv in zip(
                (self.histogram, self.alt_table), (histogram_n, alt_n), self.variant_nv
            ):
                log_priors_ng = state.priors_ws[PAIRS_C[part.contexts_n]].log()
                log_lks_ng = torch.cat([hom_ref_n.unsqueeze(-1), variant_nv], dim=-1) + log_priors_ng
                log_lk = log_lk + torch.sum(part.counts_n.double() * torch.logsumexp(log_lks_ng, dim=-1))
            log_prior = torch.sum(
                -((u_q2 - mean_2) ** 2) / (2 * variance_2) - 0.5 * torch.log(2 * math.pi * variance_2)
            )
            log_prior = log_prior + torch.sum(
                -(DAMAGE_VARIANCE_SHAPE + 1) * variance_2.log() - DAMAGE_VARIANCE_SCALE / variance_2
            )
            log_prior = log_prior + FLAT_PRIOR * state.priors_ws[self.learned_w].log().sum()
        return (log_lk + log_prior).item()

    @staticmethod
    def to_vector(state: EMState) -> Tensor:
        """the state in unconstrained coordinates, in which SQUAREM extrapolates: log priors and log variances"""
        log_priors = state.priors_ws.log().flatten()
        return torch.cat(
            [log_priors, state.unconstrained_q2.flatten(), state.hyper_mean_2, state.hyper_variance_2.log()]
        )

    def from_vector(self, vector: Tensor) -> EMState:
        """the state of a vector of to_vector's unconstrained coordinates"""
        num_priors, num_u = NUM_PAIRS * NUM_GENOTYPES, 2 * len(self.learned_processes_q)
        priors_ws = torch.softmax(vector[:num_priors].reshape(NUM_PAIRS, NUM_GENOTYPES), dim=-1)
        u_q2 = vector[num_priors : num_priors + num_u].reshape(-1, 2)
        mean_2, log_variance_2 = vector[num_priors + num_u : num_priors + num_u + 2], vector[num_priors + num_u + 2 :]
        return EMState(priors_ws, u_q2, mean_2, log_variance_2.exp())

    def run(self, convergence_threshold: float, max_iterations: int) -> EMResult:
        """
        EM accelerated by SQUAREM (Varadhan and Roland 2008, scheme S3).  From the state x0, two iterations of EM give x1
        and x2, and with r = x1 - x0 and v = x2 - 2 x1 + x0, the extrapolation x0 + 2 a r + a^2 v, with a = |r| / |v|
        between 1 and a maximum, is followed by an iteration of EM that stabilizes it.  If that loses more than
        SQUAREM_TOLERANCE of the log posterior, the next state is x2 instead and the maximum shrinks; an extrapolation at
        the maximum raises it, as in the R package SQUAREM.  max_iterations counts iterations of EM.  EM has converged
        when no learned pair's prior changes by more than convergence_threshold in a cycle and either no damage coordinate
        changes by more than DAMAGE_TOLERANCE or the log posterior by more than SQUAREM_OBJECTIVE_TOLERANCE: extrapolation
        keeps moving the coordinates that hardly change the log posterior a little.
        """
        state = self.initial_state()
        num_iterations, converged = 0, len(self.learned_processes_q) == 0
        step_max = 1.0
        log_posterior = 0.0 if converged else self.log_posterior(state)
        while not converged and num_iterations < max_iterations:
            x0 = self.to_vector(state)
            state1 = self.em_map(state)
            state2 = self.em_map(state1)
            num_iterations += 2
            r, v = self.to_vector(state1) - x0, self.to_vector(state2) - 2 * self.to_vector(state1) + x0
            alpha = (r.norm() / v.norm().clamp_min(1e-300)).clamp(min=1.0, max=step_max).item()
            if alpha > 1.0 and num_iterations < max_iterations:
                new_state = self.em_map(self.from_vector(x0 + 2 * alpha * r + alpha**2 * v))
                num_iterations += 1
            else:
                new_state = state2  # the extrapolation with a = 1
            new_log_posterior = self.log_posterior(new_state)
            if not (math.isfinite(new_log_posterior) and new_log_posterior >= log_posterior - SQUAREM_TOLERANCE):
                new_state, new_log_posterior = state2, self.log_posterior(state2)
                if alpha == step_max:
                    step_max = max(1.0, step_max / SQUAREM_STEP_FACTOR)
                alpha = 1.0
            if alpha == step_max:
                step_max *= SQUAREM_STEP_FACTOR

            prior_change = torch.linalg.vector_norm(new_state.priors_ws - state.priors_ws, dim=-1)[self.learned_w].max()
            damage_change = (new_state.unconstrained_q2 - state.unconstrained_q2).abs().max()
            log_posterior_change = abs(new_log_posterior - log_posterior)
            state, log_posterior = new_state, new_log_posterior
            converged = prior_change.item() <= convergence_threshold and (
                damage_change.item() <= DAMAGE_TOLERANCE or log_posterior_change <= SQUAREM_OBJECTIVE_TOLERANCE
            )
        u_p2 = state.hyper_mean_2.expand(NUM_PROCESSES, 2).clone()
        u_p2[self.learned_processes_q] = state.unconstrained_q2
        return EMResult(state.priors_ws, u_p2, state.hyper_mean_2, state.hyper_variance_2, num_iterations, converged)


def learned_pairs_w(sites: SiteTable) -> Tensor:
    """
    Whether each pair of a context and its reverse complement has enough data to learn from: as in GATK, sites without
    alt reads and sites with two or more, which are in the alt tables
    """
    result = []
    for selected_n in (sites.alt_counts_n == 0, sites.alt_counts_n >= 2):
        has_w = torch.zeros(NUM_PAIRS, dtype=torch.int64).index_add_(0, PAIRS_C[sites.contexts_n], selected_n.long())
        result.append(has_w > 0)
    return result[0] & result[1]


def fit_read_orientation_model(
    counts: F1R2Counts,
    capture_pseudocounts: tuple[float, float] = DEFAULT_CAPTURE_PSEUDOCOUNTS,
    emission_model: EmissionModel = None,
    convergence_threshold: float = 1e-4,
    max_iterations: int = 20,
) -> "ReadOrientationModel":
    """
    The read orientation model of a sample, learned from its F1R2 counts by EM given its capture bias (see the module
    docstring).  Given segment_pileups' emission model of the sample, germline hets have its likelihoods at the MAFs of
    the counts' MAF mixtures.
    """
    sites = SiteTable.of(counts.sites)
    learned_w = learned_pairs_w(sites)
    num_sites_c, num_alt_sites_c = (
        torch.zeros(NUM_CONTEXTS, dtype=torch.int64).index_add_(0, sites.contexts_n, sites.counts_n * selected_n)
        for selected_n in (torch.ones_like(sites.counts_n), (sites.alt_counts_n > 0).long())
    )
    for w, pair_context in enumerate(PAIR_CONTEXTS):
        c = CONTEXT_INDICES[pair_context]
        contexts = [c, REVERSE_COMPLEMENTS_C[c].item()]
        num_sites, num_alt_sites = (int(x[contexts].sum()) for x in (num_sites_c, num_alt_sites_c))
        description = (
            f"Sample {counts.sample}, contexts {pair_context} and {CONTEXTS[contexts[1]]}: "
            f"{num_sites - num_alt_sites} ref and {num_alt_sites} alt sites"
        )
        print(description if learned_w[w] else f"{description}, too few to learn from.  Using the flat prior.")

    germline_het = None if emission_model is None else GermlineHetLikelihoods(emission_model, counts.maf_mixtures_mg)
    result = OrientationEM(sites, learned_w, capture_pseudocounts, germline_het).run(
        convergence_threshold, max_iterations
    )
    outcome = f"converged in {result.num_iterations}" if result.converged else f"did not converge in {max_iterations}"
    print(f"EM {outcome} iterations")

    learned_c = learned_w[PAIRS_C]
    priors_cs = torch.where(learned_c.unsqueeze(-1), result.priors_ws[PAIRS_C], flat_priors_ws()[PAIRS_C])
    alpha_p, beta_p = damage_pseudocounts(result.damage_unconstrained_p2)
    model = ReadOrientationModel(
        counts.sample,
        torch.stack([alpha_p, beta_p], dim=-1).reshape(NUM_CONTEXTS, NUM_BASES, 2),
        priors_cs,
        torch.tensor(capture_pseudocounts, dtype=torch.float64),
        num_sites_c * learned_c,
        num_alt_sites_c * learned_c,
    )
    mean_alpha, mean_beta = damage_pseudocounts(result.hyper_mean_2.unsqueeze(0))
    print(
        f"Damage distributions' prior: mean {(mean_alpha / (mean_alpha + mean_beta)).item():.2e} and concentration "
        f"{(mean_alpha + mean_beta).item():.1f} at its center, unconstrained standard deviations "
        f"{', '.join(f'{x:.2f}' for x in result.hyper_variance_2.sqrt().tolist())}"
    )
    print(model.summary())
    return model


# ---------------------------------------------------------------------------------------------------------------------
# the model


class ReadOrientationModel(nn.Module):
    """
    A sample's read orientation model, which learn_read_orientation_model learns (see the module docstring): the
    pseudocounts of each physical damage process's distribution of rates, the genotype priors of each context, the
    pseudocounts of the capture bias, and the numbers of sites and of sites with alt reads in each context that they were
    learned from, both 0 in contexts with too few to learn from.

    The methods are batched over variants.  Bases are indices into BASES, so that A, C, G, T are 0 to 3, as in
    Permutect's haplotypes, where 4 is a deletion.
    """

    def __init__(
        self,
        sample: str,
        damage_pseudocounts_ca2: Tensor,
        genotype_priors_cs: Tensor = None,
        capture_pseudocounts_2: Tensor = None,
        num_sites_c: Tensor = None,
        num_alt_sites_c: Tensor = None,
    ):
        super().__init__()
        if tuple(damage_pseudocounts_ca2.shape) != (NUM_CONTEXTS, NUM_BASES, 2):
            raise ValueError(
                f"A read orientation model's damage pseudocounts have shape {(NUM_CONTEXTS, NUM_BASES, 2)}, "
                f"not {tuple(damage_pseudocounts_ca2.shape)}"
            )
        self.sample = sample
        if genotype_priors_cs is None:
            genotype_priors_cs = flat_priors_ws()[PAIRS_C]
        if capture_pseudocounts_2 is None:
            capture_pseudocounts_2 = torch.tensor(DEFAULT_CAPTURE_PSEUDOCOUNTS)
        zeros_c = torch.zeros(NUM_CONTEXTS, dtype=torch.int64)
        # buffers rather than parameters, since nothing that uses the model changes them
        self.register_buffer("damage_pseudocounts_ca2", damage_pseudocounts_ca2.double().clone())
        self.register_buffer("genotype_priors_cs", genotype_priors_cs.double().clone())
        self.register_buffer("capture_pseudocounts_2", capture_pseudocounts_2.double().clone())
        self.register_buffer("num_sites_c", zeros_c.clone() if num_sites_c is None else num_sites_c.long().clone())
        self.register_buffer(
            "num_alt_sites_c", zeros_c.clone() if num_alt_sites_c is None else num_alt_sites_c.long().clone()
        )

    @staticmethod
    def processes_bo(left_bases_b: Tensor, ref_bases_b: Tensor, right_bases_b: Tensor, alt_bases_b: Tensor):
        """
        The physical processes of damage to the + strand and to the - strand that would give each variant's alt base, a
        (B, 2) tensor, and whether each variant is an SNV, which only damage can give.  Other variants get some process.
        """
        bases = (left_bases_b, ref_bases_b, right_bases_b, alt_bases_b)
        snvs_b = (torch.stack(bases) < NUM_BASES).all(dim=0) & (alt_bases_b != ref_bases_b)
        # a deletion's index, NUM_BASES, is past the last base
        left_b, ref_b, right_b, alt_b = (bases_b.long().clamp(max=NUM_BASES - 1) for bases_b in bases)
        plus_b = NUM_BASES * context_indices(left_b, ref_b, right_b) + alt_b
        return torch.stack([plus_b, OTHER_STRAND_PROCESSES_P.to(plus_b.device)[plus_b]], dim=-1), snvs_b

    def damage_log_likelihoods_b(
        self,
        left_bases_b: Tensor,
        ref_bases_b: Tensor,
        right_bases_b: Tensor,
        alt_bases_b: Tensor,
        alt_f1r2_counts_b: Tensor,
        alt_f2r1_counts_b: Tensor,
        f1r2_depths_b: Tensor,
        f2r1_depths_b: Tensor,
        depths_b: Tensor = None,
        alt_counts_b: Tensor = None,
    ) -> tuple[Tensor, Tensor]:
        """
        The log likelihood of each variant's alt reads from damage at a HOM_REF site, given its numbers of reads in each
        orientation, n1 and n2, and, if given, its total depth and alt count, which include reads of unknown orientation,
        relative to the factor that real variants share, and whether each variant is an SNV, which only damage can give
        (see the module docstring)
        """
        processes_bo, snvs_b = self.processes_bo(left_bases_b, ref_bases_b, right_bases_b, alt_bases_b)
        alpha_bo, beta_bo = self.damage_pseudocounts_ca2.flatten(0, 1)[processes_bo].unbind(-1)
        counts = (alt_f1r2_counts_b, f1r2_depths_b, alt_f2r1_counts_b, f2r1_depths_b)
        damage_b = known_split_damage_log_likelihoods(
            *counts, alpha_bo[:, 0], beta_bo[:, 0], alpha_bo[:, 1], beta_bo[:, 1]
        )
        split_b = log_hypergeometric(alt_f1r2_counts_b, alt_f2r1_counts_b, f1r2_depths_b, f2r1_depths_b)
        if depths_b is None:
            return damage_b - split_b, snvs_b

        # reads of unknown orientation, each with damage at the expected rate of the strands' mixture
        oriented_depths_b = f1r2_depths_b + f2r1_depths_b
        oriented_alt_b = alt_f1r2_counts_b + alt_f2r1_counts_b
        unoriented_depths_b = torch.clamp(depths_b - oriented_depths_b, min=0)
        unoriented_alt_b = torch.minimum(torch.clamp(alt_counts_b - oriented_alt_b, min=0), unoriented_depths_b)
        capture_alpha, capture_beta = self.capture_pseudocounts_2.unbind(-1)
        f1r2_fraction_b = (f1r2_depths_b + capture_alpha) / (oriented_depths_b + capture_alpha + capture_beta)
        plus_rate_b = (alpha_bo[:, 0] + alt_f1r2_counts_b) / (alpha_bo[:, 0] + beta_bo[:, 0] + f1r2_depths_b)
        minus_rate_b = (alpha_bo[:, 1] + alt_f2r1_counts_b) / (alpha_bo[:, 1] + beta_bo[:, 1] + f2r1_depths_b)
        rate_b = f1r2_fraction_b * plus_rate_b + (1 - f1r2_fraction_b) * minus_rate_b
        unoriented_b = binomial_log_lk(unoriented_depths_b, unoriented_alt_b, rate_b)
        # the multivariate hypergeometric: the oriented split times how the alt reads split between oriented and not
        oriented_split_b = (
            log_binomial_coefficient(oriented_depths_b, oriented_alt_b)
            + log_binomial_coefficient(unoriented_depths_b, unoriented_alt_b)
            - log_binomial_coefficient(oriented_depths_b + unoriented_depths_b, oriented_alt_b + unoriented_alt_b)
        )
        return damage_b + unoriented_b - split_b - oriented_split_b, snvs_b

    def summary(self, num_processes: int = 12) -> str:
        """the physical processes of the highest mean damage rate, with their concentrations, and the capture bias"""
        alpha_p, beta_p = self.damage_pseudocounts_ca2.flatten(0, 1).unbind(-1)
        means_p = torch.where(POSSIBLE_PROCESSES_P, alpha_p / (alpha_p + beta_p), 0)
        lines = [f"Sample {self.sample}'s physical damage processes of highest mean rate: mean, concentration"]
        for p in means_p.argsort(descending=True)[:num_processes].tolist():
            lines.append(f"    {process_name(p)}: {means_p[p].item():.2e}, {(alpha_p[p] + beta_p[p]).item():.1f}")
        alpha, beta = self.capture_pseudocounts_2.tolist()
        lines.append(f"Capture bias: F1R2 fraction {alpha / (alpha + beta):.3f}, concentration {alpha + beta:.1f}")
        return "\n".join(lines)

    def save(self, path):
        """save everything needed to reconstruct the model, as tensors and a string, which torch.load can read safely"""
        state = {name: buffer.cpu() for name, buffer in self.named_buffers()}
        torch.save(state | {"sample": self.sample}, path)

    @classmethod
    def load(cls, path) -> "ReadOrientationModel":
        state = torch.load(path, map_location="cpu", weights_only=True)
        sample = state.pop("sample")
        return cls(sample, **state)
