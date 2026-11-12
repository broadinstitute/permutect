import os
import tarfile

import numpy as np
import pytest

from permutect.read_orientation.picard_histogram_reader import BASES
from permutect.read_orientation.picard_histogram_reader import CONTEXTS
from permutect.read_orientation.picard_histogram_reader import ORIENTATIONS
from permutect.read_orientation.picard_histogram_reader import AltSites
from permutect.read_orientation.picard_histogram_reader import read_f1r2_counts
from permutect.read_orientation.picard_histogram_reader import read_picard_histograms
from permutect.test.test_file_names import DREAM_TUMOR_4_F1R2_SCATTERS

ALT_LABELS = [
    f"{context}_{alt_base}_{orientation}"
    for context in CONTEXTS
    for alt_base in BASES
    if alt_base != context[1]
    for orientation in ORIENTATIONS
]
ALT_TABLE_HEADER = "context\tref_count\talt_count\tref_f1r2\talt_f1r2\tdepth\talt"


def histogram_file_text(sample: str, counts_by_label: dict, max_depth: int) -> str:
    """a Picard metrics file of histograms as htsjdk writes them, with a count for every depth from 1 to max_depth"""
    lines = ["## htsjdk.samtools.metrics.StringHeader", "# " + sample, "", "", "## HISTOGRAM\tjava.lang.Integer"]
    lines.append("\t".join(["depth"] + list(counts_by_label)))
    for depth in range(1, max_depth + 1):
        lines.append("\t".join([str(depth)] + [str(counts[depth - 1]) for counts in counts_by_label.values()]))
    return "\n".join(lines) + "\n\n"


def alt_table_text(sample: str, rows) -> str:
    """an alt table with rows (context, ref count, alt count, ref F1R2 count, alt F1R2 count, alt base)"""
    lines = ["#<METADATA>SAMPLE=" + sample, ALT_TABLE_HEADER]
    for context, ref_count, alt_count, ref_f1r2, alt_f1r2, alt_base in rows:
        lines.append(
            f"{context}\t{ref_count}\t{alt_count}\t{ref_f1r2}\t{alt_f1r2}\t{ref_count + alt_count}\t{alt_base}"
        )
    return "\n".join(lines) + "\n"


def write_tar_gz(path, files: dict):
    """a tar.gz of files with the given names and text, with GATK's ./ prefix"""
    directory = str(path) + ".files"
    os.makedirs(directory)
    with tarfile.open(path, "w:gz") as tar:
        for name, text in files.items():
            with open(os.path.join(directory, name), "w") as file:
                file.write(text)
            tar.add(os.path.join(directory, name), arcname="./" + name)


def f1r2_files(sample: str, max_depth: int, ref_counts: dict, alt_counts: dict, alt_rows_by_context: dict) -> dict:
    """the files of a sample's F1R2 counts, with zero counts for histograms that aren't given"""
    zeros = np.zeros(max_depth, dtype=np.int64)
    ref = {context: ref_counts.get(context, zeros) for context in CONTEXTS}
    alt = {label: alt_counts.get(label, zeros) for label in ALT_LABELS}
    files = {
        sample + ".ref_histogram": histogram_file_text(sample, ref, max_depth),
        sample + ".alt_histogram": histogram_file_text(sample, alt, max_depth),
    }
    for context in CONTEXTS:
        files[f"{sample}.{context}.alt_table"] = alt_table_text(sample, alt_rows_by_context.get(context, []))
    return files


def test_read_picard_histograms():
    lines = [
        "## htsjdk.samtools.metrics.StringHeader",
        "# my sample",
        "",
        "## HISTOGRAM\tjava.lang.Integer",
        "depth\tAAA\tACG_T_F1R2",
        "1\t3\t0",
        "2\t5.0\t7",
        "",
    ]
    histograms = read_picard_histograms(line + "\n" for line in lines)
    assert histograms.string_headers == ["my sample"]
    assert histograms.bin_label == "depth"
    assert histograms.bins_b.tolist() == [1, 2]
    assert {label: counts.tolist() for label, counts in histograms.counts_by_label.items()} == {
        "AAA": [3, 5],
        "ACG_T_F1R2": [0, 7],
    }


def test_inputs_are_summed(tmp_path):
    max_depth = 4
    ref_1 = {"ACG": np.array([1, 2, 3, 4])}
    ref_2 = {"ACG": np.array([10, 0, 0, 1]), "TTT": np.array([0, 0, 5, 0])}
    alt_1 = {"ACG_T_F1R2": np.array([0, 1, 0, 2])}
    alt_2 = {"ACG_T_F1R2": np.array([1, 0, 0, 0]), "ACG_A_F2R1": np.array([0, 0, 3, 0])}
    rows_1 = {"ACG": [("ACG", 10, 5, 5, 4, "T"), ("ACG", 3, 5, 2, 4, "T")], "TTT": [("TTT", 7, 2, 3, 1, "G")]}
    rows_2 = {"ACG": [("ACG", 11, 5, 6, 4, "T"), ("ACG", 10, 5, 3, 4, "T")]}
    write_tar_gz(tmp_path / "1.tar.gz", f1r2_files("S", max_depth, ref_1, alt_1, rows_1))
    write_tar_gz(tmp_path / "2.tar.gz", f1r2_files("S", max_depth, ref_2, alt_2, rows_2))
    # a second sample in one input only
    write_tar_gz(tmp_path / "3.tar.gz", f1r2_files("OTHER", max_depth, ref_1, {}, {}))

    counts_by_sample = read_f1r2_counts([tmp_path / "1.tar.gz", tmp_path / "2.tar.gz", tmp_path / "3.tar.gz"])
    assert counts_by_sample.keys() == {"S", "OTHER"}
    counts = counts_by_sample["S"]
    assert counts.max_depth == max_depth
    assert counts.ref_histograms["ACG"].tolist() == [11, 2, 3, 5]
    assert counts.ref_histograms["TTT"].tolist() == [0, 0, 5, 0]
    assert counts.ref_histograms["CCC"].tolist() == [0, 0, 0, 0]
    assert counts.alt_depth_one_histograms[("ACG", "T", "F1R2")].tolist() == [1, 1, 0, 2]
    assert counts.alt_depth_one_histograms[("ACG", "A", "F2R1")].tolist() == [0, 0, 3, 0]

    # sites with the same alt base, alt count, alt F1R2 count and depth are combined, whatever their ref F1R2 counts
    acg = counts.alt_sites["ACG"]
    rows = sorted(zip(*(column.tolist() for column in acg.columns())))
    t = BASES.index("T")
    assert rows == [(t, 5, 4, 8, 1), (t, 5, 4, 15, 2), (t, 5, 4, 16, 1)]
    assert counts.alt_sites["TTT"].num_sites() == 1
    assert counts.alt_sites["CCC"].num_sites() == 0
    assert counts_by_sample["OTHER"].ref_histograms["ACG"].tolist() == [1, 2, 3, 4]


def test_alt_sites_are_collapsed():
    sites = AltSites.collapse(
        np.array([3, 3, 3, 0]),
        np.array([2, 2, 2, 2]),
        np.array([1, 1, 0, 1]),
        np.array([30, 30, 30, 30]),
        np.array([1, 2, 1, 1]),
    )
    assert sorted(zip(*(column.tolist() for column in sites.columns()))) == [
        (0, 2, 1, 30, 1),
        (3, 2, 0, 30, 1),
        (3, 2, 1, 30, 3),
    ]
    assert (sites + sites).num_sites() == 10
    assert len((sites + sites).num_sites_n) == 3


def test_alt_table_of_an_older_gatk_is_rejected(tmp_path):
    files = f1r2_files("S", 2, {}, {}, {})
    files["S.alt_table"] = alt_table_text("S", [("ACG", 10, 5, 5, 4, "T")])
    write_tar_gz(tmp_path / "counts.tar.gz", files)
    with pytest.raises(ValueError, match="Older versions of GATK"):
        read_f1r2_counts([tmp_path / "counts.tar.gz"])


def test_alt_table_of_a_sample_without_histograms_is_rejected(tmp_path):
    files = f1r2_files("S", 2, {}, {}, {})
    files["OTHER.ACG.alt_table"] = alt_table_text("OTHER", [("ACG", 10, 5, 5, 4, "T")])
    write_tar_gz(tmp_path / "counts.tar.gz", files)
    with pytest.raises(ValueError, match="OTHER"):
        read_f1r2_counts([tmp_path / "counts.tar.gz"])


def test_inputs_with_different_max_depths_are_rejected(tmp_path):
    write_tar_gz(tmp_path / "1.tar.gz", f1r2_files("S", 2, {}, {}, {}))
    write_tar_gz(tmp_path / "2.tar.gz", f1r2_files("S", 3, {}, {}, {}))
    with pytest.raises(ValueError, match="different depth bins"):
        read_f1r2_counts([tmp_path / "1.tar.gz", tmp_path / "2.tar.gz"])


def test_gatk_scatters_are_summed():
    # the counts of GATK's scatters together are the sums of each scatter's counts.  The sample's name has dots, like
    # the alt tables' context.
    sample = "synthetic.challenge.set4.tumour"
    together = read_f1r2_counts(DREAM_TUMOR_4_F1R2_SCATTERS)[sample]
    first, second = (read_f1r2_counts([scatter])[sample] for scatter in DREAM_TUMOR_4_F1R2_SCATTERS)
    assert together.max_depth == first.max_depth == second.max_depth == 200
    for histograms in ("ref_histograms", "alt_depth_one_histograms"):
        for key, counts in getattr(together, histograms).items():
            assert np.array_equal(counts, getattr(first, histograms)[key] + getattr(second, histograms)[key])
    assert all(first.alt_sites[context].num_sites() > 0 for context in ("ACG", "CGT"))
    for context in CONTEXTS:
        summed = first.alt_sites[context] + second.alt_sites[context]
        for together_column, summed_column in zip(together.alt_sites[context].columns(), summed.columns()):
            assert np.array_equal(together_column, summed_column)
