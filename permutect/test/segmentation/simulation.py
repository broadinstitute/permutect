"""
Simulated pileups at common germline SNPs of a tumor sample with copy number alterations, and of its matched normal,
for testing segmentation and contamination.  The data come from a more realistic process than the emission model: each
contaminant has a genotype at each site, drawn from the population allele frequency with Hardy-Weinberg proportions,
rather than the population allele frequency on average, and hom and het sites have the same overdispersion.
"""

from dataclasses import dataclass

import numpy as np
import torch

from permutect.segmentation.emissions import Pileups

# allele-specific copy numbers (minor, major) of altered tumor cells, with total copy numbers 0 through 4
COPY_NUMBERS = [(minor, total - minor) for total in range(5) for minor in range(total // 2 + 1)]

DEPTH = 60
REFERENCE_BIAS = 1.1
# the probability of a read having the other allele of the site.  As if errors were equally likely to give each of the
# three other bases, reads have one of the two bases that are neither allele with twice this probability.
ERROR_RATE = 0.002
CONCENTRATION = 200.0
OUTLIER_RATE = 0.002


@dataclass
class SimulatedSample:
    contigs_c: list[str]
    lengths_c: np.ndarray
    positions_n: np.ndarray
    # the depths are ref plus alt counts
    pileups: Pileups
    # reads with neither allele, which GATK's GetPileupSummaries counts as other alt
    other_alt_counts_n: np.ndarray
    true_mafs_n: np.ndarray
    # the number of alt alleles of the germline genotype
    genotypes_n: np.ndarray
    # sites whose alt counts are uniformly random, as for a mapping artifact, which a matched normal shares
    outliers_n: np.ndarray


def maf_and_copy_ratio(purity: float, cell_fraction: float, minor: int, major: int) -> tuple[float, float]:
    """
    The MAF and the copy number relative to 2 of a sample in which a fraction cell_fraction of the tumor cells have
    minor and major copies of the two haplotypes and every other cell has one copy of each
    """
    altered = purity * cell_fraction
    minor_copies, major_copies = 1 - altered + altered * minor, 1 - altered + altered * major
    return minor_copies / (minor_copies + major_copies), (minor_copies + major_copies) / 2


def simulate_sample(
    rng: np.random.Generator,
    purity: float,
    cell_fractions: list[float],
    contaminations: list[float],
    num_contigs: int = 3,
    segments_per_contig: tuple[int, int] = (2, 5),
    sites_per_segment: tuple[int, int] = (1000, 4000),
    min_maf_difference: float = 0.05,
) -> SimulatedSample:
    """
    Each contig has a random number of segments, each with a random number of sites.  Each segment has a random cell
    fraction and allele-specific copy number, redrawn until its MAF either equals that of the previous segment, which
    makes them one segment as far as allele fractions can tell, or differs by at least min_maf_difference, so that the
    true segments are unambiguous.  Each element of contaminations is the fraction of reads from one contaminant.
    """
    lengths_c, true_mafs, copy_ratios = [], [], []
    for _ in range(num_contigs):
        lengths_c.append(0)
        previous_maf = None
        for _ in range(rng.integers(segments_per_contig[0], segments_per_contig[1] + 1)):
            while True:
                minor, major = COPY_NUMBERS[rng.integers(len(COPY_NUMBERS))]
                maf, copy_ratio = maf_and_copy_ratio(purity, rng.choice(cell_fractions), minor, major)
                if previous_maf is None or not 1e-9 < abs(maf - previous_maf) < min_maf_difference:
                    break
            # equal MAFs from different copy numbers can differ by rounding error
            maf = previous_maf if previous_maf is not None and abs(maf - previous_maf) <= 1e-9 else maf
            length = rng.integers(sites_per_segment[0], sites_per_segment[1] + 1)
            lengths_c[-1] += length
            true_mafs.append(np.full(length, maf))
            copy_ratios.append(np.full(length, copy_ratio))
            previous_maf = maf
    lengths_c = np.array(lengths_c)
    true_mafs_n, copy_ratios_n = np.concatenate(true_mafs), np.concatenate(copy_ratios)
    num_sites = len(true_mafs_n)
    positions_n = np.concatenate([1000 + np.cumsum(rng.integers(1, 2000, length)) for length in lengths_c])

    allele_frequencies_n = rng.uniform(0.05, 0.95, num_sites)
    # the number of alt alleles, 0, 1 or 2 with the Hardy-Weinberg probabilities
    genotypes_n = rng.binomial(2, allele_frequencies_n)
    het_alt_fractions_n = np.where(rng.random(num_sites) < 0.5, true_mafs_n, 1 - true_mafs_n)
    sample_alt_fractions_n = np.where(genotypes_n == 1, het_alt_fractions_n, genotypes_n / 2)
    pileups, other_alt_counts_n, outliers_n = simulate_reads(
        rng, sample_alt_fractions_n, allele_frequencies_n, contaminations, DEPTH * copy_ratios_n
    )
    contigs_c = [f"chr{c + 1}" for c in range(num_contigs)]
    args = (pileups, other_alt_counts_n, true_mafs_n, genotypes_n, outliers_n)
    return SimulatedSample(contigs_c, lengths_c, positions_n, *args)


def simulate_matched_normal(
    rng: np.random.Generator, tumor: SimulatedSample, contaminations: list[float]
) -> SimulatedSample:
    """a normal sample of the same individual as a simulated tumor, with the same sites and no copy number alterations"""
    allele_frequencies_n = tumor.pileups.allele_frequencies_n.numpy()
    num_sites = len(allele_frequencies_n)
    depths_n = np.full(num_sites, float(DEPTH))
    args = (tumor.genotypes_n / 2, allele_frequencies_n, contaminations, depths_n, tumor.outliers_n)
    pileups, other_alt_counts_n, _ = simulate_reads(rng, *args)
    args = (pileups, other_alt_counts_n, np.full(num_sites, 0.5), tumor.genotypes_n, tumor.outliers_n)
    return SimulatedSample(tumor.contigs_c, tumor.lengths_c, tumor.positions_n, *args)


def simulate_reads(
    rng: np.random.Generator,
    sample_alt_fractions_n: np.ndarray,
    allele_frequencies_n: np.ndarray,
    contaminations: list[float],
    mean_depths_n: np.ndarray,
    outliers_n: np.ndarray = None,
) -> tuple[Pileups, np.ndarray, np.ndarray]:
    """
    The pileups, other alt counts and outlier sites, random unless given, of a sample whose own reads have alt
    fractions sample_alt_fractions_n
    """
    num_sites = len(sample_alt_fractions_n)
    mixed_n = (1 - sum(contaminations)) * sample_alt_fractions_n
    for contamination in contaminations:
        mixed_n += contamination * rng.binomial(2, allele_frequencies_n) / 2
    biased_n = mixed_n / (mixed_n + (1 - mixed_n) * REFERENCE_BIAS)
    means_n = biased_n * (1 - ERROR_RATE) + (1 - biased_n) * ERROR_RATE

    depths_n = rng.poisson(mean_depths_n)
    alt_counts_n = rng.binomial(depths_n, rng.beta(means_n * CONCENTRATION, (1 - means_n) * CONCENTRATION))
    outliers_n = rng.random(num_sites) < OUTLIER_RATE if outliers_n is None else outliers_n
    alt_counts_n[outliers_n] = rng.integers(0, depths_n[outliers_n] + 1)
    other_alt_counts_n = rng.binomial(depths_n, 2 * ERROR_RATE)

    arrays = (alt_counts_n, depths_n, allele_frequencies_n)
    return Pileups(*(torch.tensor(array, dtype=torch.float64) for array in arrays)), other_alt_counts_n, outliers_n
