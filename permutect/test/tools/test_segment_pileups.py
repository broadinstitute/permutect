import numpy as np
import pytest

from permutect import constants
from permutect.test.segmentation.simulation import SimulatedSample
from permutect.test.segmentation.simulation import simulate_sample
from permutect.test.segmentation.test_segmenter import runs
from permutect.tools import segment_pileups
from permutect.tools.filter_variants import get_segmentation

PILEUPS_HEADER = "contig\tposition\tref_count\talt_count\tother_alt_count\tallele_frequency\tref_base\talt_base\n"


def write_pileup_summaries(table, sample: SimulatedSample):
    contigs_n = np.repeat(sample.contigs_c, sample.lengths_c)
    alt_counts_n, depths_n = sample.pileups.alt_counts_n.int().numpy(), sample.pileups.depths_n.int().numpy()
    with open(table, "w") as file:
        file.write("#<METADATA>SAMPLE=tumor\n" + PILEUPS_HEADER)
        for contig, position, alt_count, depth, allele_frequency in zip(
            contigs_n, sample.positions_n, alt_counts_n, depths_n, sample.pileups.allele_frequencies_n.numpy()
        ):
            file.write(f"{contig}\t{position}\t{depth - alt_count}\t{alt_count}\t0\t{allele_frequency}\tA\tC\n")


def run_segment_pileups(pileups_table, segments_table, plot) -> list[str]:
    """the lines of the segments table, after checking that there's a plot"""
    argv = ["--" + constants.INPUT_NAME, str(pileups_table), "--" + constants.OUTPUT_NAME, str(segments_table)]
    argv += ["--" + constants.PLOT_NAME, str(plot)]
    segment_pileups.main_without_parsing(segment_pileups.parse_arguments(argv))
    with open(plot, "rb") as file:
        assert file.read(8) == b"\x89PNG\r\n\x1a\n"
    with open(segments_table) as file:
        return file.read().splitlines()


def test_segment_pileups(tmp_path):
    sample = simulate_sample(
        np.random.default_rng(2),
        purity=0.7,
        cell_fractions=[1.0],
        contamination=0.0,
        num_contigs=2,
        segments_per_contig=(2, 3),
        sites_per_segment=(1500, 2500),
    )
    # the tool leaves out sites without coverage, such as the first one
    sample.pileups.depths_n[0], sample.pileups.alt_counts_n[0] = 0, 0
    pileups_table, segments_table = tmp_path / "tumor.pileups.table", tmp_path / "tumor.segments.table"
    write_pileup_summaries(pileups_table, sample)
    lines = run_segment_pileups(pileups_table, segments_table, tmp_path / "tumor.png")

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

    # filter_variants reads the segments, and gets the true MAFs in the middle of the true segments
    segmentation = get_segmentation(str(segments_table))
    contigs_n = np.repeat(sample.contigs_c, sample.lengths_c)
    for true_start, true_end in runs(sample.true_mafs_n, sample.lengths_c):
        middle = (true_start + true_end) // 2
        (interval,) = segmentation[contigs_n[middle]][sample.positions_n[middle]]
        assert interval.data == pytest.approx(sample.true_mafs_n[middle], abs=0.02)


def test_segment_pileups_without_sites(tmp_path):
    pileups_table, segments_table = tmp_path / "empty.pileups.table", tmp_path / "empty.segments.table"
    with open(pileups_table, "w") as file:
        file.write("#<METADATA>SAMPLE=empty\n" + PILEUPS_HEADER + "chr1\t100\t0\t0\t0\t0.3\tA\tC\n")
    # there's still a plot, albeit an empty one, so that a WDL task's plot output always exists
    assert run_segment_pileups(pileups_table, segments_table, tmp_path / "empty.png") == [
        "#<METADATA>SAMPLE=empty",
        "\t".join(segment_pileups.SEGMENTS_COLUMNS),
    ]
