import os
import tarfile

import numpy as np
import pytest
import torch

from permutect.read_orientation.picard_histogram_reader import ALT_HISTOGRAM_LABELS
from permutect.read_orientation.picard_histogram_reader import CONTEXTS
from permutect.read_orientation.picard_histogram_reader import Sites
from permutect.read_orientation.picard_histogram_reader import read_f1r2_counts
from permutect.read_orientation.picard_histogram_reader import read_picard_histograms
from permutect.test.test_file_names import DREAM_TUMOR_4_F1R2_SCATTERS

ALT_TABLE_HEADER = "context\tref_count\talt_count\tref_f1r2\talt_f1r2\tdepth\talt"
A, C, G, T = range(4)


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
    alt = {label: alt_counts.get(label, zeros) for label in ALT_HISTOGRAM_LABELS}
    files = {
        sample + ".ref_histogram": histogram_file_text(sample, ref, max_depth),
        sample + ".alt_histogram": histogram_file_text(sample, alt, max_depth),
    }
    for context in CONTEXTS:
        files[f"{sample}.{context}.alt_table"] = alt_table_text(sample, alt_rows_by_context.get(context, []))
    return files


def rows(sites: Sites) -> list:
    """the sites as sorted (alt base, alt count, alt F1R2 count, depth, number of sites)"""
    return sorted(tuple(key) + (count,) for key, count in zip(sites.keys_n4.tolist(), sites.counts_n.tolist()))


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
    assert histograms.labels == ["AAA", "ACG_T_F1R2"]
    assert histograms.bins_b.tolist() == [1, 2]
    assert histograms.counts_bl.tolist() == [[3, 0], [5, 7]]


def test_sites():
    # identical keys are combined, and keys without sites dropped
    sites = Sites.of([[T, 2, 1, 30], [A, 2, 1, 30], [T, 2, 1, 30], [T, 2, 0, 30], [G, 3, 3, 9]], [1, 1, 2, 1, 0])
    assert rows(sites) == [(A, 2, 1, 30, 1), (T, 2, 0, 30, 1), (T, 2, 1, 30, 3)]
    assert (sites.num_sites(), sites.num_sites(sites.alt_bases_n == T)) == (5, 4)
    assert rows(sites + sites) == [(A, 2, 1, 30, 2), (T, 2, 0, 30, 2), (T, 2, 1, 30, 6)]
    assert rows(Sites.concatenate([])) == []

    # in the reverse complement's orientation, the alt bases are complementary and F1R2 and F2R1 reads swap
    assert rows(sites.reverse_complement()) == [(A, 2, 1, 30, 3), (A, 2, 2, 30, 1), (T, 2, 1, 30, 1)]


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
    assert counts.sites.keys() == set(CONTEXTS)
    # sites without alt reads have the ref base as their alt base, and alt table sites with the same alt base, alt
    # count, alt F1R2 count and depth are combined, whatever their ref F1R2 counts
    ref_sites = [(C, 0, 0, 1, 11), (C, 0, 0, 2, 2), (C, 0, 0, 3, 3), (C, 0, 0, 4, 5)]
    one_alt_read_sites = [(A, 1, 0, 3, 3), (T, 1, 1, 1, 1), (T, 1, 1, 2, 1), (T, 1, 1, 4, 2)]
    alt_table_sites = [(T, 5, 4, 8, 1), (T, 5, 4, 15, 2), (T, 5, 4, 16, 1)]
    assert rows(counts.sites["ACG"]) == sorted(ref_sites + one_alt_read_sites + alt_table_sites)
    assert rows(counts.sites["TTT"]) == [(G, 2, 1, 9, 1), (T, 0, 0, 3, 5)]
    assert rows(counts.sites["CCC"]) == []
    assert rows(counts_by_sample["OTHER"].sites["ACG"]) == [(C, 0, 0, depth, depth) for depth in range(1, 5)]


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
    assert all(first.sites[context].num_sites(first.sites[context].alt_counts_n >= 2) > 0 for context in ("ACG", "CGT"))
    for context in CONTEXTS:
        summed = first.sites[context] + second.sites[context]
        assert torch.equal(together.sites[context].keys_n4, summed.keys_n4)
        assert torch.equal(together.sites[context].counts_n, summed.counts_n)
