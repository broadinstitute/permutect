"""
The contamination of a sample by reads from other people, by the method of GATK CalculateContamination, but with the
minor allele fraction (MAF) segments of segment_pileups in place of GATK's own.

1. The sites are those of the sample's GetPileupSummaries table, optionally only those in a range of population allele
   frequencies (AF), whose total count, including other alt alleles, is above MIN_COVERAGE, above LOW_COVERAGE_RATIO
   times the median and below HIGH_COVERAGE_RATIO times the mean.
2. A genotype model of the matched normal's sites if there is one, and otherwise of the sample's.  The genotypes are hom
   ref, alt minor het, alt major het and hom alt, with Hardy-Weinberg priors and binomial alt counts.  The alt fraction
   of a het is its segment's MAF, contamination dilutes each genotype's alt fraction toward the population AF, and
   sequencing errors give hom sites some reads of the other allele.  The model's contamination maximizes the likelihood
   of the sites outside segments of loss of heterozygosity (LoH).
3. The sample's contamination comes from its reads at sites that the model genotypes as hom alt: the fraction of reads
   with the ref allele, after subtracting expected sequencing errors, divided by the ref allele's population frequency,
   which is the probability that a contaminant's read has it.  If this isn't precise enough, it tries again in segments
   of lower MAF, then with hom refs instead of hom alts, and finally with hom refs chosen by their alt fractions, as
   small gene panels need.
"""

import math

import numpy as np
import torch
from torch import Tensor

from permutect.segmentation.pileup_summaries import PileupSummaries
from permutect.segmentation.segmenter import Segment
from permutect.utils.stats_utils import binomial_log_lk

MIN_COVERAGE = 10
LOW_COVERAGE_RATIO = 0.5
HIGH_COVERAGE_RATIO = 3.0

# the indices g of the genotypes of the genotype model: hom ref, alt minor het, alt major het and hom alt
HOM_REF, HOM_ALT = 0, 3

# The model's contamination comes from segments with MAF above the first of these thresholds that leaves more than
# MIN_NON_LOH_FRACTION of the sites, or from every segment if none does.
NON_LOH_MAF_THRESHOLDS = np.linspace(0.4, 0.04, 10)
MIN_NON_LOH_FRACTION = 0.25
# the model's contamination is the best of a coarse grid of contaminations, refined by a fine grid around it
MAX_CONTAMINATION = 0.5
COARSE_GRID_SIZE, FINE_GRID_SIZE, FINE_GRID_HALF_WIDTH = 51, 41, 0.01

# The sample's contamination comes from hom alts in segments with MAF above each of these thresholds in turn, until its
# error is below MAX_RELATIVE_ERROR times the estimate plus MAX_ABSOLUTE_ERROR, then from hom refs.
HOM_ALT_MAF_THRESHOLDS = (0.4, 0.36, 0.32, 0.28)
HOM_REF_MAF_THRESHOLD = 0.24
MAX_RELATIVE_ERROR = 0.2
MAX_ABSOLUTE_ERROR = 0.001
# Failing that, the hom refs are the sites with alt fraction below HOM_REF_MAX_ALT_FRACTION, except for those above the
# HOM_REF_QUANTILE quantile and HOM_REF_MIN_CUTOFF, which might be hets in LoH.
HOM_REF_MAX_ALT_FRACTION = 0.15
HOM_REF_QUANTILE = 0.9
HOM_REF_MIN_CUTOFF = 0.1

CONTAMINATION_TABLE_COLUMNS = ("sample", "contamination", "error")


def calculate_contamination(
    pileups: PileupSummaries,
    segments: list[Segment],
    normal_pileups: PileupSummaries = None,
    normal_segments: list[Segment] = None,
    min_allele_frequency: float = None,
    max_allele_frequency: float = None,
) -> tuple[float, float]:
    """
    The contamination of a sample and its standard error, given its pileup summaries and segments and, optionally,
    those of a matched normal, which then genotypes the sites.  Sites whose population allele frequency isn't strictly
    between min_allele_frequency and max_allele_frequency, if given, are left out.
    """
    sites = filter_by_coverage(in_allele_frequency_range(pileups, min_allele_frequency, max_allele_frequency))
    if normal_pileups is None:
        model = GenotypeModel(sites, segments)
    else:
        normal_sites = in_allele_frequency_range(normal_pileups, min_allele_frequency, max_allele_frequency)
        model = GenotypeModel(filter_by_coverage(normal_sites), normal_segments)
    return model.contamination_of(sites)


def write_contamination_table(table, sample: str, contamination: float, error: float):
    """the contamination table of GATK CalculateContamination"""
    with open(table, "w") as file:
        file.write("\t".join(CONTAMINATION_TABLE_COLUMNS) + "\n")
        file.write(f"{sample}\t{contamination}\t{error}\n")


def in_allele_frequency_range(sites: PileupSummaries, min_allele_frequency, max_allele_frequency) -> PileupSummaries:
    in_range_n = np.ones(len(sites), dtype=bool)
    if min_allele_frequency is not None:
        in_range_n &= min_allele_frequency < sites.allele_frequencies_n
    if max_allele_frequency is not None:
        in_range_n &= sites.allele_frequencies_n < max_allele_frequency
    return sites.subset(in_range_n)


def filter_by_coverage(sites: PileupSummaries) -> PileupSummaries:
    # sites without coverage are removed first so that they don't throw off the median
    covered = sites.subset(sites.total_counts() > MIN_COVERAGE)
    if len(covered) == 0:
        return covered
    coverage_n = covered.total_counts()
    low, high = LOW_COVERAGE_RATIO * np.median(coverage_n), HIGH_COVERAGE_RATIO * np.mean(coverage_n)
    return covered.subset((low < coverage_n) & (coverage_n < high))


def sequencing_error_rate(sites: PileupSummaries) -> float:
    """
    The rate of sequencing errors, from the reads with other alt alleles, assuming that errors are equally likely to give
    each of the three other bases, two of which are other alt alleles
    """
    total_count = np.sum(sites.total_counts())
    return 1.5 * np.sum(sites.other_alt_counts_n) / total_count if total_count > 0 else 0.0


def segment_mafs(sites: PileupSummaries, segments: list[Segment]) -> np.ndarray:
    """the MAF of the segment of each site, or NaN for a site in no segment"""
    mafs_n = np.full(len(sites), np.nan)
    segments_by_contig = {}
    for segment in segments:
        segments_by_contig.setdefault(segment.contig, []).append(segment)
    for contig, contig_segments in segments_by_contig.items():
        starts, ends, mafs = (np.array(values) for values in zip(*((s.start, s.end, s.maf) for s in contig_segments)))
        site_indices = np.flatnonzero(sites.contigs_n == contig)
        positions = sites.positions_n[site_indices]
        # the last segment starting at or before each site, which contains it if it doesn't end before it
        segment_indices = np.maximum(np.searchsorted(starts, positions, side="right") - 1, 0)
        in_segment = (starts[segment_indices] <= positions) & (positions <= ends[segment_indices])
        mafs_n[site_indices[in_segment]] = mafs[segment_indices[in_segment]]
    return mafs_n


def non_loh_threshold(mafs_n: Tensor) -> float:
    """the MAF above which segments aren't LoH in fitting the model's contamination"""
    for threshold in NON_LOH_MAF_THRESHOLDS:
        if torch.mean((mafs_n > threshold).double()) > MIN_NON_LOH_FRACTION:
            return threshold
    return -math.inf


class GenotypeModel:
    """
    The genotypes hom ref, alt minor het, alt major het and hom alt of sites given the MAFs of their segments, and the
    contamination that best explains them.  Sites in no segment are left out.
    """

    def __init__(self, sites: PileupSummaries, segments: list[Segment]):
        error_rate = sequencing_error_rate(sites)
        mafs_n = segment_mafs(sites, segments)
        in_segment_n = ~np.isnan(mafs_n)
        self.sites = sites.subset(in_segment_n)
        self.mafs_n = torch.from_numpy(mafs_n[in_segment_n])
        self.alt_counts_n = torch.from_numpy(self.sites.alt_counts_n).double()
        self.depths_n = torch.from_numpy(self.sites.ref_counts_n + self.sites.alt_counts_n).double()
        self.allele_frequencies_n = torch.from_numpy(self.sites.allele_frequencies_n)

        # Hardy-Weinberg priors of the genotypes, and their alt fractions without contamination
        f_n, maf_n, error_rates_n = self.allele_frequencies_n, self.mafs_n, torch.full_like(self.mafs_n, error_rate)
        self.log_priors_ng = torch.log(torch.stack(((1 - f_n) ** 2, f_n * (1 - f_n), f_n * (1 - f_n), f_n**2), dim=1))
        self.alt_fractions_ng = torch.stack((error_rates_n / 3, maf_n, 1 - maf_n, 1 - error_rates_n), dim=1)
        self.contamination = self.fit_contamination(self.mafs_n > non_loh_threshold(self.mafs_n))

    def log_likelihoods(self, contamination: float, sites_n: Tensor) -> Tensor:
        """the log prior plus log likelihood of each genotype g of the sites in sites_n"""
        f_n1 = self.allele_frequencies_n[sites_n, None]
        alt_fractions_ng = (1 - contamination) * self.alt_fractions_ng[sites_n] + contamination * f_n1
        depths_n1, alt_counts_n1 = self.depths_n[sites_n, None], self.alt_counts_n[sites_n, None]
        return self.log_priors_ng[sites_n] + binomial_log_lk(depths_n1, alt_counts_n1, alt_fractions_ng)

    def fit_contamination(self, sites_n: Tensor) -> float:
        """the contamination that maximizes the likelihood of the sites in sites_n"""

        def best(contaminations: np.ndarray) -> float:
            log_likelihoods = [torch.logsumexp(self.log_likelihoods(c, sites_n), dim=1).sum() for c in contaminations]
            return float(contaminations[np.argmax(log_likelihoods)])

        coarse = best(np.linspace(0, MAX_CONTAMINATION, COARSE_GRID_SIZE))
        low, high = max(coarse - FINE_GRID_HALF_WIDTH, 0), min(coarse + FINE_GRID_HALF_WIDTH, MAX_CONTAMINATION)
        return best(np.linspace(low, high, FINE_GRID_SIZE))

    def genotyped_sites(self, genotype: int, min_maf: float) -> PileupSummaries:
        """the sites in segments with MAF above min_maf whose posterior probability of a genotype is above 1/2"""
        candidates_n = self.mafs_n > min_maf
        posteriors_n = torch.softmax(self.log_likelihoods(self.contamination, candidates_n), dim=1)[:, genotype]
        genotyped_n = torch.zeros_like(candidates_n)
        genotyped_n[candidates_n] = posteriors_n > 0.5
        return self.sites.subset(genotyped_n.numpy())

    def contamination_of(self, sites: PileupSummaries) -> tuple[float, float]:
        """
        The contamination of a sample of the same person, and its standard error, given its sites, from the most
        reliable hom sites that give a precise enough estimate
        """
        error_rate = sequencing_error_rate(sites)
        attempts = [(HOM_ALT, threshold) for threshold in HOM_ALT_MAF_THRESHOLDS] + [(HOM_REF, HOM_REF_MAF_THRESHOLD)]
        for genotype, min_maf in attempts:
            homs = at_loci(sites, self.genotyped_sites(genotype, min_maf))
            contamination, error = estimate_from_homs(homs, genotype == HOM_ALT, error_rate)
            if error < MAX_RELATIVE_ERROR * contamination + MAX_ABSOLUTE_ERROR:
                return contamination, error
        contamination, error = estimate_from_homs(hom_refs_by_alt_fraction(sites), False, error_rate)
        return (0.0, 1.0) if math.isnan(contamination) else (contamination, error)


def at_loci(sites: PileupSummaries, other: PileupSummaries) -> PileupSummaries:
    """the sites at the contigs and positions of the sites of other"""
    loci = set(zip(other.contigs_n, other.positions_n))
    return sites.subset(np.array([locus in loci for locus in zip(sites.contigs_n, sites.positions_n)], dtype=bool))


def hom_refs_by_alt_fraction(sites: PileupSummaries) -> PileupSummaries:
    """sites that look hom ref by their alt fractions alone"""
    alt_fractions_n = sites.alt_counts_n / np.maximum(sites.total_counts(), 1)
    candidates_n = alt_fractions_n < HOM_REF_MAX_ALT_FRACTION
    if not np.any(candidates_n):
        return sites.subset(candidates_n)
    cutoff = max(HOM_REF_MIN_CUTOFF, np.quantile(alt_fractions_n[candidates_n], HOM_REF_QUANTILE))
    return sites.subset(candidates_n & (alt_fractions_n <= cutoff))


def estimate_from_homs(homs: PileupSummaries, hom_alt: bool, error_rate: float) -> tuple[float, float]:
    """
    The contamination and its standard error from the reads of the opposite allele at hom sites, or NaNs if there are
    none.  A contaminant's read has the opposite allele with the probability f of its population frequency, so for a
    site of depth d we expect c f d such reads, plus sequencing errors.
    """
    depths_n = homs.total_counts().astype(np.float64)
    opposite_counts_n = homs.ref_counts_n if hom_alt else homs.alt_counts_n
    frequencies_n = 1 - homs.allele_frequencies_n if hom_alt else homs.allele_frequencies_n
    weighted_depth = np.sum(frequencies_n * depths_n)
    if weighted_depth == 0:
        return math.nan, math.nan
    estimate = max(np.sum(opposite_counts_n) - error_rate / 3 * np.sum(depths_n), 0) / weighted_depth

    # Given contamination c, the estimate's variance is c (1 - c) / W + c^2 v, where W = sum f d, from binomial sampling
    # of reads, and v = sum f (1 - f) d^2 / W^2, from each site's number of contaminant alleles.  Rather than use this
    # variance at c = estimate, which is 0 at an estimate of 0, the error is the larger distance from the estimate to the
    # contaminations a standard deviation away from it, the roots of the quadratic (c - estimate)^2 = variance(c).
    v = np.sum(frequencies_n * (1 - frequencies_n) * depths_n**2) / weighted_depth**2
    quadratic, linear, constant = 1 + 1 / weighted_depth - v, -(2 * estimate + 1 / weighted_depth), estimate**2
    discriminant = linear**2 - 4 * quadratic * constant
    if quadratic > 0 and discriminant >= 0:
        lower = (-linear - math.sqrt(discriminant)) / (2 * quadratic)
        upper = (-linear + math.sqrt(discriminant)) / (2 * quadratic)
        error = max(upper - estimate, estimate - lower)
    else:
        # with very few sites there may be no such contaminations, and the error is the standard deviation at the estimate
        error = math.sqrt(max(estimate * (1 - estimate) / weighted_depth + estimate**2 * v, 0))
    return min(estimate, 1.0), error
