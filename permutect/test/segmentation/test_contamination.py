import math
import os

import numpy as np
import pytest
import torch

from permutect.segmentation.contamination import MAX_ABSOLUTE_ERROR
from permutect.segmentation.contamination import GenotypeModel
from permutect.segmentation.contamination import calculate_contamination
from permutect.segmentation.contamination import estimate_from_homs
from permutect.segmentation.contamination import filter_by_coverage
from permutect.segmentation.contamination import hom_refs_by_alt_fraction
from permutect.segmentation.contamination import non_loh_threshold
from permutect.segmentation.contamination import segment_mafs
from permutect.segmentation.contamination import write_contamination_table
from permutect.segmentation.pileup_summaries import PileupSummaries
from permutect.segmentation.pileup_summaries import read_pileup_summaries
from permutect.segmentation.segmenter import Segment
from permutect.test.test_file_names import CONTAMINATION_TEST_DIR
from permutect.test.test_file_names import GATK_CONTAMINATION_SEGMENTS_DIR

NA12891_1_PCT = "NA12891_0.01_NA12892_0.99.table"
NA12891_8_PCT = "NA12891_0.08_NA12892_0.92.table"
ALLELE_FREQUENCY_RANGE = (0.07, 0.15)

# the contamination and error from GATK CalculateContamination (GATK commit 8caefa10d) for GATK's test data, with an
# optional matched normal and range of population allele frequencies
GATK_RESULTS = [
    (NA12891_1_PCT, None, None, 0.013148854899098974, 7.185562603803028e-4),
    ("NA12891_0.03_NA12892_0.97.table", None, None, 0.03356118262185688, 0.001332270918388459),
    ("NA12891_0.05_NA12892_0.95.table", None, None, 0.05284385677567531, 0.0018377200826548562),
    (NA12891_8_PCT, None, None, 0.08026059610209388, 0.002561663559993571),
    (NA12891_8_PCT, NA12891_1_PCT, None, 0.08026062411631564, 0.002431847972524201),
    (NA12891_8_PCT, None, ALLELE_FREQUENCY_RANGE, 0.08712101100677862, 0.003918086900676729),
    (NA12891_8_PCT, NA12891_1_PCT, ALLELE_FREQUENCY_RANGE, 0.0854959548589443, 0.004505034108329878),
    ("small-panel-pileups1.tsv", None, None, 0.0, 4.76837158203125e-7),
    ("small-panel-pileups2.tsv", None, None, 0.0, 4.76837158203125e-7),
    ("small-panel-pileups3.tsv", None, None, 0.0, 4.76837158203125e-7),
    ("small_gene_panel.pileups", None, None, 0.0064799769205902006, 0.00511125923586071),
]


def gatk_segments(pileups_file: str, allele_frequency_range) -> list[Segment]:
    """the segments and MAFs that GATK CalculateContamination found for a table of its test data"""
    name = os.path.splitext(pileups_file)[0]
    if allele_frequency_range is not None:
        name += "_af_{}_{}".format(*allele_frequency_range)
    with open(os.path.join(GATK_CONTAMINATION_SEGMENTS_DIR, name + ".segments.table")) as file:
        rows = [line.split() for line in file if not line.startswith("#")][1:]
    return [Segment(contig, int(start), int(end), float(maf), float(maf)) for contig, start, end, maf in rows]


@pytest.mark.parametrize("tumor, normal, allele_frequency_range, expected_contamination, expected_error", GATK_RESULTS)
def test_gatk_segments_give_gatk_results(tumor, normal, allele_frequency_range, expected_contamination, expected_error):
    tumor_pileups = read_pileup_summaries(os.path.join(CONTAMINATION_TEST_DIR, tumor))
    normal_arguments = (None, None)
    if normal is not None:
        normal_pileups = read_pileup_summaries(os.path.join(CONTAMINATION_TEST_DIR, normal))
        normal_arguments = (normal_pileups, gatk_segments(normal, allele_frequency_range))
    min_allele_frequency, max_allele_frequency = allele_frequency_range or (None, None)
    contamination, error = calculate_contamination(
        tumor_pileups,
        gatk_segments(tumor, allele_frequency_range),
        *normal_arguments,
        min_allele_frequency=min_allele_frequency,
        max_allele_frequency=max_allele_frequency,
    )

    # given GATK's own segments and MAFs, the results are GATK's, up to the resolution of the model's contamination and
    # the interpolation of the quantile of alt fractions of small panels
    assert contamination == pytest.approx(expected_contamination, abs=2e-4)
    if expected_contamination > 0:
        assert error == pytest.approx(expected_error, rel=0.03)
    else:
        # GATK's error for an estimate of 0 is the precision of its bisection, which finds the root 0 of
        # c = standard error(c) rather than the positive one
        assert 0 < error < MAX_ABSOLUTE_ERROR


def test_allele_frequency_range():
    # restricting the range of allele frequencies is the same as leaving the other sites out of the pileups
    tumor = read_pileup_summaries(os.path.join(CONTAMINATION_TEST_DIR, NA12891_8_PCT))
    normal = read_pileup_summaries(os.path.join(CONTAMINATION_TEST_DIR, NA12891_1_PCT))
    tumor_segments = gatk_segments(NA12891_8_PCT, ALLELE_FREQUENCY_RANGE)
    normal_segments = gatk_segments(NA12891_1_PCT, ALLELE_FREQUENCY_RANGE)
    restricted = calculate_contamination(tumor, tumor_segments, normal, normal_segments, *ALLELE_FREQUENCY_RANGE)

    def in_range(pileups: PileupSummaries) -> PileupSummaries:
        frequencies_n = pileups.allele_frequencies_n
        in_range_n = (ALLELE_FREQUENCY_RANGE[0] < frequencies_n) & (frequencies_n < ALLELE_FREQUENCY_RANGE[1])
        assert 0 < np.count_nonzero(in_range_n) < len(pileups)
        return pileups.subset(in_range_n)

    assert calculate_contamination(in_range(tumor), tumor_segments, in_range(normal), normal_segments) == restricted


def pileups(contigs, positions, ref_counts, alt_counts, other_alt_counts=None, allele_frequencies=None):
    num_sites = len(positions)
    other_alt_counts = [0] * num_sites if other_alt_counts is None else other_alt_counts
    allele_frequencies = [0.2] * num_sites if allele_frequencies is None else allele_frequencies
    arrays = (positions, ref_counts, alt_counts, other_alt_counts)
    return PileupSummaries(
        "sample",
        np.array(contigs, dtype=object),
        *(np.array(array, dtype=np.int64) for array in arrays),
        np.array(allele_frequencies, dtype=np.float64),
    )


def test_filter_by_coverage():
    # total counts, including other alts, of 10 (not above the minimum), 20, 40, 41, 42 and 200.  The median of those
    # above the minimum is 41 and the mean is 68.6, so sites need more than 20.5 and fewer than 205.8.
    sites = pileups(
        ["1"] * 6, [1, 2, 3, 4, 5, 6], [5, 10, 20, 20, 20, 100], [5, 10, 20, 20, 20, 99], [0, 0, 0, 1, 2, 1]
    )
    assert list(filter_by_coverage(sites).positions_n) == [3, 4, 5, 6]


def test_segment_mafs():
    sites = pileups(["1", "1", "1", "2", "2", "1", "1"], [100, 200, 300, 50, 70, 150, 400], [10] * 7, [10] * 7)
    segments = [
        Segment("1", 100, 150, 0.5, 0.5),
        # a gap segment between sites
        Segment("1", 151, 199, 0.5, 0.3),
        Segment("1", 200, 300, 0.3, 0.3),
        # a segment of a contig without sites
        Segment("3", 1, 1000, 0.1, 0.1),
        # a segment that leaves out the last site of its contig
        Segment("2", 50, 60, 0.4, 0.4),
    ]
    assert np.array_equal(segment_mafs(sites, segments), [0.5, 0.3, 0.3, 0.4, np.nan, 0.5, np.nan], equal_nan=True)


def test_non_loh_threshold():
    # 30% of sites are in segments with MAF above 0.4
    assert non_loh_threshold(torch.tensor([0.45] * 30 + [0.3] * 10 + [0.1] * 60)) == 0.4
    # otherwise the threshold falls by 0.04 until more than a quarter of the sites are above it
    assert non_loh_threshold(torch.tensor([0.25] * 30 + [0.33] * 10 + [0.1] * 60)) == pytest.approx(0.24)
    # if that never happens, every segment
    assert non_loh_threshold(torch.tensor([0.0] * 100)) == -math.inf


def test_estimate_from_homs():
    # hom alts whose ref alleles have population frequencies 0.2 and 0.5, with depths of 100 and 200.  Of their 11 ref
    # reads, an expected 0.003 / 3 * 300 = 0.3 are sequencing errors.
    homs = pileups(["1", "1"], [1, 2], [3, 8], [97, 192], allele_frequencies=[0.8, 0.5])
    estimate, error = estimate_from_homs(homs, hom_alt=True, error_rate=0.003)
    weighted_depth = 0.2 * 100 + 0.5 * 200
    assert estimate == pytest.approx((11 - 0.3) / weighted_depth)

    # the error is the largest distance from the estimate to a contamination that's within a standard deviation of it
    def variance(c):
        return c * (1 - c) / weighted_depth + c**2 * (0.2 * 0.8 * 100**2 + 0.5 * 0.5 * 200**2) / weighted_depth**2

    contaminations = np.linspace(0, 1, 1_000_001)
    consistent = (contaminations - estimate) ** 2 <= variance(contaminations)
    assert error == pytest.approx(np.max(np.abs(contaminations[consistent] - estimate)), abs=1e-5)

    # no opposite reads beyond the expected errors give an estimate of 0, but not an error of 0
    no_contamination_homs = pileups(["1", "1"], [1, 2], [0, 0], [100, 200], allele_frequencies=[0.8, 0.5])
    estimate, error = estimate_from_homs(no_contamination_homs, hom_alt=True, error_rate=0.003)
    assert estimate == 0 and error > 0
    # no hom sites give no estimate
    assert all(math.isnan(x) for x in estimate_from_homs(homs.subset([]), hom_alt=True, error_rate=0.003))


def test_hom_refs_by_alt_fraction():
    # alt fractions of 0 to 0.19 in steps of 0.01.  Those below 0.15 are candidates, and the highest tenth of them, with
    # alt fractions above the 0.9 quantile, 0.126, are left out.
    alt_counts = list(range(20))
    sites = pileups(["1"] * 20, list(range(1, 21)), [100 - k for k in alt_counts], alt_counts)
    assert list(hom_refs_by_alt_fraction(sites).alt_counts_n) == list(range(13))
    # but alt fractions up to 0.1 are kept
    low_sites = sites.subset(sites.alt_counts_n < 11)
    assert list(hom_refs_by_alt_fraction(low_sites).alt_counts_n) == list(range(11))


def test_genotype_model_contamination():
    # sites drawn from the genotype model, without sequencing errors, with a contamination of 0.05 and depth 100
    rng = np.random.default_rng(0)
    num_sites, contamination, depth = 5000, 0.05, 100
    allele_frequencies_n = rng.uniform(0.05, 0.5, num_sites)
    sample_alt_fractions_n = rng.binomial(2, allele_frequencies_n) / 2
    alt_fractions_n = (1 - contamination) * sample_alt_fractions_n + contamination * allele_frequencies_n
    alt_counts_n = rng.binomial(depth, alt_fractions_n)
    positions_n = np.arange(1, num_sites + 1)
    sites = pileups(["1"] * num_sites, positions_n, depth - alt_counts_n, alt_counts_n, None, allele_frequencies_n)
    model = GenotypeModel(sites, [Segment("1", 1, num_sites, 0.5, 0.5)])
    assert model.contamination == pytest.approx(contamination, abs=0.005)


def test_write_contamination_table(tmp_path):
    table = tmp_path / "contamination.table"
    write_contamination_table(table, "tumor", 0.0125, 0.002)
    assert table.read_text() == "sample\tcontamination\terror\ntumor\t0.0125\t0.002\n"
