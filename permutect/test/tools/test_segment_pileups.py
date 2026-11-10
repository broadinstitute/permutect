import math
import os
from dataclasses import dataclass

import numpy as np
import pytest

from permutect import constants
from permutect.segmentation.contamination import CONTAMINATION_TABLE_COLUMNS
from permutect.segmentation.pileup_summaries import SAMPLE_HEADER
from permutect.segmentation.segmentation_model import SegmentationModel
from permutect.test.segmentation.simulation import SimulatedSample
from permutect.test.segmentation.simulation import simulate_matched_normal
from permutect.test.segmentation.simulation import simulate_sample
from permutect.test.segmentation.test_segmenter import runs
from permutect.test.test_file_names import CONTAMINATION_TEST_DIR
from permutect.tools import segment_pileups

PILEUPS_HEADER = "contig\tposition\tref_count\talt_count\tother_alt_count\tallele_frequency\tref_base\talt_base\n"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def write_pileup_summaries(table, sample_name: str, rows):
    """rows of contig, position, ref count, alt count, other alt count and population allele frequency"""
    with open(table, "w") as file:
        file.write(SAMPLE_HEADER + sample_name + "\n" + PILEUPS_HEADER)
        for row in rows:
            file.write("\t".join(str(value) for value in row) + "\tA\tC\n")


def write_simulated_pileup_summaries(table, sample: SimulatedSample, sample_name: str = "tumor", kept_n=None):
    """the pileup summaries of a simulated sample, optionally of only the sites in kept_n"""
    alt_counts_n, depths_n = sample.pileups.alt_counts_n.int().numpy(), sample.pileups.depths_n.int().numpy()
    contigs_n = np.repeat(sample.contigs_c, sample.lengths_c)
    allele_frequencies_n = sample.pileups.allele_frequencies_n.numpy()
    columns = (contigs_n, sample.positions_n, depths_n - alt_counts_n, alt_counts_n, sample.other_alt_counts_n)
    kept_n = np.ones(len(contigs_n), dtype=bool) if kept_n is None else kept_n
    write_pileup_summaries(table, sample_name, zip(*(column[kept_n] for column in (*columns, allele_frequencies_n))))


@dataclass
class Outputs:
    # the lines of the segments tables
    lines: list[str]
    normal_lines: list[str] | None
    contamination: float
    error: float
    segmentation_model: SegmentationModel


def run_segment_pileups(directory, tumor_table, normal_table=None) -> Outputs:
    """run segment_pileups with every output, and check that it writes the plots and a consistent segmentation model"""
    arguments = {
        constants.INPUT_NAME: tumor_table,
        constants.OUTPUT_NAME: directory / "tumor.segments.table",
        constants.CONTAMINATION_TABLE_NAME: directory / "contamination.table",
        constants.PLOT_NAME: directory / "tumor.png",
        constants.SEGMENTATION_MODEL_NAME: directory / "segmentation_model.pt",
    }
    if normal_table is not None:
        arguments[constants.NORMAL_INPUT_NAME] = normal_table
        arguments[constants.NORMAL_OUTPUT_NAME] = directory / "normal.segments.table"
        arguments[constants.NORMAL_PLOT_NAME] = directory / "normal.png"
    argv = [word for name, value in arguments.items() for word in ("--" + name, str(value))]
    segment_pileups.main_without_parsing(segment_pileups.parse_arguments(argv))

    for plot in (constants.PLOT_NAME, constants.NORMAL_PLOT_NAME):
        if plot in arguments:
            assert arguments[plot].read_bytes()[: len(PNG_SIGNATURE)] == PNG_SIGNATURE
    contamination_lines = arguments[constants.CONTAMINATION_TABLE_NAME].read_text().splitlines()
    assert len(contamination_lines) == 2 and contamination_lines[0].split() == list(CONTAMINATION_TABLE_COLUMNS)
    contamination, error = (float(value) for value in contamination_lines[1].split()[1:])
    lines = arguments[constants.OUTPUT_NAME].read_text().splitlines()
    normal_lines = None if normal_table is None else arguments[constants.NORMAL_OUTPUT_NAME].read_text().splitlines()

    # the segmentation model has the segments of the tables, and the contamination of the table
    model = SegmentationModel.load(arguments[constants.SEGMENTATION_MODEL_NAME])
    assert (model.contamination, model.contamination_error) == (contamination, error)
    assert model.tumor.emission_model.contamination == contamination
    for segmentation, expected_lines in ((model.tumor, lines), (model.normal, normal_lines)):
        if expected_lines is None:
            assert segmentation.segments == []
        else:
            table = directory / "from_model.segments.table"
            segment_pileups.write_segments(table, segmentation.sample, segmentation.segments)
            assert table.read_text().splitlines() == expected_lines
    # the normal is segmented as if uncontaminated
    assert model.normal.emission_model.contamination == 0
    return Outputs(lines, normal_lines, contamination, error, model)


def segments(lines: list[str]) -> list[list[str]]:
    """the rows of a segments table other than the gap segments between segments, which have no MAF interval"""
    assert lines[1].split() == list(segment_pileups.SEGMENTS_COLUMNS)
    return [row for row in (line.split() for line in lines[2:]) if row[5] != "NaN"]


def check_true_mafs(segmentation_model: SegmentationModel, sample: SimulatedSample):
    """filter_variants gets the true MAFs in the middle of the true segments from the segmentation model"""
    segmentation = segmentation_model.tumor.maf_intervals()
    contigs_n = np.repeat(sample.contigs_c, sample.lengths_c)
    for true_start, true_end in runs(sample.true_mafs_n, sample.lengths_c):
        middle = (true_start + true_end) // 2
        (interval,) = segmentation[contigs_n[middle]][sample.positions_n[middle]]
        assert interval.data == pytest.approx(sample.true_mafs_n[middle], abs=0.02)


def test_segment_pileups(tmp_path):
    sample = simulate_sample(
        np.random.default_rng(2),
        purity=0.7,
        cell_fractions=[1.0],
        contaminations=[],
        num_contigs=2,
        segments_per_contig=(2, 3),
        sites_per_segment=(1500, 2500),
    )
    # the tool leaves out sites without coverage, such as the first one
    sample.pileups.depths_n[0], sample.pileups.alt_counts_n[0] = 0, 0
    pileups_table = tmp_path / "tumor.pileups.table"
    write_simulated_pileup_summaries(pileups_table, sample)
    outputs = run_segment_pileups(tmp_path, pileups_table)
    lines = outputs.lines

    assert lines[0] == "#<METADATA>SAMPLE=tumor"
    assert lines[1].split() == list(segment_pileups.SEGMENTS_COLUMNS)
    rows = [line.split() for line in lines[2:]]
    starts_c = np.cumsum(sample.lengths_c) - sample.lengths_c
    for contig, start, length in zip(sample.contigs_c, starts_c, sample.lengths_c):
        positions = sample.positions_n[start : start + length]
        if start == 0:
            positions = positions[1:]
        contig_rows = [row for row in rows if row[0] == contig]
        starts, ends = [int(row[1]) for row in contig_rows], [int(row[2]) for row in contig_rows]
        mafs, secondary_mafs = [row[3] for row in contig_rows], [row[4] for row in contig_rows]
        lows, highs = [row[5] for row in contig_rows], [row[6] for row in contig_rows]

        # the segments cover the contig from its first site to its last without overlapping
        assert starts[0] == positions[0] and ends[-1] == positions[-1]
        assert starts[1:] == [end + 1 for end in ends[:-1]]
        # gap segments, which start and end between sites, have the MAF of the segment before and the secondary MAF of
        # the segment after, and no interval
        gaps = [index for index, segment_start in enumerate(starts) if segment_start not in positions]
        assert gaps
        for index in range(len(contig_rows)):
            if index in gaps:
                assert ends[index] not in positions
                assert (mafs[index], secondary_mafs[index]) == (mafs[index - 1], mafs[index + 1])
                assert lows[index] == highs[index] == "NaN"
            else:
                assert ends[index] in positions and secondary_mafs[index] == mafs[index]
                assert float(lows[index]) <= float(mafs[index]) <= float(highs[index])

    check_true_mafs(outputs.segmentation_model, sample)
    assert outputs.contamination < 0.005


def simulate_contaminated_tumor(rng: np.random.Generator, contaminations: list[float]) -> SimulatedSample:
    """a tumor whose contaminants' genotypes have Hardy-Weinberg proportions"""
    tumor = simulate_sample(
        rng,
        purity=0.7,
        cell_fractions=[1.0],
        contaminations=contaminations,
        num_contigs=2,
        segments_per_contig=(2, 3),
        sites_per_segment=(1500, 2500),
    )
    # this tumor has LoH, which the contamination estimate must leave out without a matched normal, and a balanced
    # segment without a homozygous deletion, which would have too little coverage for the estimate to use
    assert np.min(tumor.true_mafs_n) < 0.2
    balanced_n = tumor.true_mafs_n == 0.5
    assert np.any(balanced_n) and np.mean(tumor.pileups.depths_n.numpy()[balanced_n]) > 50
    return tumor


def check_normal_segments(normal_lines: list[str], contigs_c: list[str]):
    """a normal's segments are one per contig, balanced"""
    assert normal_lines[0] == "#<METADATA>SAMPLE=normal"
    normal_rows = segments(normal_lines)
    assert [row[0] for row in normal_rows] == contigs_c
    assert all(float(row[3]) >= 0.45 for row in normal_rows)


@pytest.mark.parametrize(
    "normal_contaminations", [None, [], [0.03]], ids=["tumor_only", "matched_normal", "contaminated_matched_normal"]
)
@pytest.mark.parametrize("contaminations", [[0.05], [0.03, 0.02]], ids=["one_contaminant", "two_contaminants"])
def test_contaminated_tumor(tmp_path, contaminations, normal_contaminations):
    # a contaminated normal, which is segmented as if it weren't, has a contaminant of its own
    rng = np.random.default_rng(6)
    tumor = simulate_contaminated_tumor(rng, contaminations)
    tumor_table, normal_table = tmp_path / "tumor.pileups.table", None
    write_simulated_pileup_summaries(tumor_table, tumor)
    if normal_contaminations is not None:
        normal = simulate_matched_normal(rng, tumor, normal_contaminations)
        normal_table = tmp_path / "normal.pileups.table"
        write_simulated_pileup_summaries(normal_table, normal, "normal")
    outputs = run_segment_pileups(tmp_path, tumor_table, normal_table)

    assert outputs.contamination == pytest.approx(sum(contaminations), abs=0.01)
    assert 0 < outputs.error < 0.01
    check_true_mafs(outputs.segmentation_model, tumor)
    if normal_contaminations is not None:
        check_normal_segments(outputs.normal_lines, tumor.contigs_c)


def test_tumor_and_normal_with_different_sites(tmp_path):
    # GetPileupSummaries leaves out sites without coverage, so a tumor's and its normal's tables can have different sites
    rng = np.random.default_rng(6)
    tumor = simulate_contaminated_tumor(rng, [0.05])
    normal = simulate_matched_normal(rng, tumor, [])
    num_sites = len(tumor.positions_n)
    in_tumor_n, in_normal_n = rng.random(num_sites) < 0.9, rng.random(num_sites) < 0.9
    tumor_table, normal_table = tmp_path / "tumor.pileups.table", tmp_path / "normal.pileups.table"
    write_simulated_pileup_summaries(tumor_table, tumor, kept_n=in_tumor_n)
    write_simulated_pileup_summaries(normal_table, normal, "normal", kept_n=in_normal_n)
    outputs = run_segment_pileups(tmp_path, tumor_table, normal_table)

    assert outputs.contamination == pytest.approx(0.05, abs=0.01)
    check_true_mafs(outputs.segmentation_model, tumor)
    check_normal_segments(outputs.normal_lines, tumor.contigs_c)


def test_segment_pileups_without_sites(tmp_path):
    pileups_table = tmp_path / "empty.pileups.table"
    with open(pileups_table, "w") as file:
        file.write(SAMPLE_HEADER + "empty\n" + PILEUPS_HEADER + "chr1\t100\t0\t0\t0\t0.3\tA\tC\n")
    # there's still a plot, albeit an empty one, so that a WDL task's plot output always exists
    outputs = run_segment_pileups(tmp_path, pileups_table)
    assert outputs.lines == ["#<METADATA>SAMPLE=empty", "\t".join(segment_pileups.SEGMENTS_COLUMNS)]
    # like GATK, a contamination of 0 with an error of 1 when there is no estimate
    assert (outputs.contamination, outputs.error) == (0.0, 1.0)


# ports of GATK's CalculateContaminationIntegrationTest, with segment_pileups' segments in place of GATK's


@pytest.mark.parametrize("include_hom_alts", [True, False])
def test_gatk_artificial_data(tmp_path, include_hom_alts):
    rng = np.random.default_rng(111)
    num_sites, spacing, loh_start, loh_end, loh_maf = 2000, 100_000, 1000, 1100, 0.2
    sample_depth, contaminant_depth = 100, 7
    mafs_n = np.full(num_sites, 0.5)
    mafs_n[loh_start:loh_end] = loh_maf

    allele_frequencies_n = rng.uniform(0.05, 0.3, num_sites)
    # each read of the contaminant has the alt allele with the population allele frequency
    contaminant_alt_counts_n = rng.binomial(contaminant_depth, allele_frequencies_n)
    het_probabilities_n = 2 * allele_frequencies_n * (1 - allele_frequencies_n)
    hom_alt_probabilities_n = allele_frequencies_n**2 if include_hom_alts else np.zeros(num_sites)
    x_n = rng.random(num_sites)
    hets_n = x_n < het_probabilities_n
    hom_alts_n = ~hets_n & (x_n < het_probabilities_n + hom_alt_probabilities_n)
    het_alt_fractions_n = np.where(rng.random(num_sites) < 0.5, mafs_n, 1 - mafs_n)
    sample_alt_counts_n = np.where(hets_n, rng.binomial(sample_depth, het_alt_fractions_n), hom_alts_n * sample_depth)
    alt_counts_n = contaminant_alt_counts_n + sample_alt_counts_n

    pileups_table = tmp_path / "pileups.table"
    positions_n = spacing * np.arange(1, num_sites + 1)
    ref_counts_n = sample_depth + contaminant_depth - alt_counts_n
    columns = (["chr1"] * num_sites, positions_n, ref_counts_n, alt_counts_n, [0] * num_sites, allele_frequencies_n)
    write_pileup_summaries(pileups_table, "SAMPLE", zip(*columns))
    outputs = run_segment_pileups(tmp_path, pileups_table)

    assert outputs.contamination == pytest.approx(contaminant_depth / (sample_depth + contaminant_depth), abs=0.01)
    assert [float(row[3]) for row in segments(outputs.lines)] == pytest.approx([0.5, loh_maf, 0.5], abs=0.05)


@pytest.mark.parametrize(
    "pileups", ["small-panel-pileups1.tsv", "small-panel-pileups2.tsv", "small-panel-pileups3.tsv"]
)
def test_gatk_small_gene_panel_error_bars(tmp_path, pileups):
    # these samples all have very low contamination.  The estimate is small, with a small but nonzero error.
    outputs = run_segment_pileups(tmp_path, os.path.join(CONTAMINATION_TEST_DIR, pileups))
    assert outputs.contamination < 0.01
    assert 0 < outputs.error < 0.005


NA12891_1_PCT = os.path.join(CONTAMINATION_TEST_DIR, "NA12891_0.01_NA12892_0.99.table")
NA12891_8_PCT = os.path.join(CONTAMINATION_TEST_DIR, "NA12891_0.08_NA12892_0.92.table")
# NA12892's contamination before the spike-in of NA12891
BASELINE_CONTAMINATION_OF_NA12892 = 0.001


def test_gatk_spike_in(tmp_path):
    outputs = run_segment_pileups(tmp_path, NA12891_8_PCT)
    assert outputs.contamination == pytest.approx(0.08 + BASELINE_CONTAMINATION_OF_NA12892, abs=0.015)


def test_gatk_matched_normal(tmp_path):
    # the 1% spike-in is a close approximation to an uncontaminated matched normal
    outputs = run_segment_pileups(tmp_path, NA12891_8_PCT, NA12891_1_PCT)
    assert outputs.contamination == pytest.approx(0.08 + BASELINE_CONTAMINATION_OF_NA12892, abs=0.01)


def test_gatk_small_gene_panel_with_no_hom_alts(tmp_path):
    outputs = run_segment_pileups(tmp_path, os.path.join(CONTAMINATION_TEST_DIR, "small_gene_panel.pileups"))
    assert not math.isnan(outputs.contamination) and outputs.contamination < 0.2
