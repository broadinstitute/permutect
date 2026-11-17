import os
import tarfile

import pytest
import torch

from permutect.read_orientation.f1r2_counts_reader import ALT_HISTOGRAM_KEY_COLUMNS
from permutect.read_orientation.f1r2_counts_reader import CONTEXTS
from permutect.read_orientation.f1r2_counts_reader import REF_HISTOGRAM_KEY_COLUMNS
from permutect.read_orientation.f1r2_counts_reader import Sites
from permutect.read_orientation.f1r2_counts_reader import read_f1r2_counts
from permutect.test.test_file_names import DREAM_TUMOR_4_F1R2_SCATTERS

ALT_TABLE_HEADER = "ref_count\talt_count\tref_f1r2\talt_f1r2\talt\tcontig\tposition"
ROUNDED_DEPTHS = list(range(1, 25)) + list(range(30, 201, 10))
A, C, G, T = range(4)


def histogram_table_text(sample: str, key_columns, depths, rows) -> str:
    """a histogram table with columns of the given depths, and rows of keys and then a list of the counts of each depth"""
    lines = ["#<METADATA>SAMPLE=" + sample, "\t".join([*key_columns, *map(str, depths)])]
    lines += ["\t".join(map(str, [*keys, *counts])) for *keys, counts in rows]
    return "\n".join(lines) + "\n"


def alt_table_text(sample: str, rows) -> str:
    """an alt table with rows (ref count, alt count, ref F1R2 count, alt F1R2 count, alt base) at increasing positions"""
    lines = ["#<METADATA>SAMPLE=" + sample, ALT_TABLE_HEADER]
    for position, (ref_count, alt_count, ref_f1r2, alt_f1r2, alt_base) in enumerate(rows, start=1):
        lines.append(f"{ref_count}\t{alt_count}\t{ref_f1r2}\t{alt_f1r2}\t{alt_base}\tchr1\t{1000 * position}")
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


def f1r2_files(sample: str, depths, ref_rows, alt_rows, alt_table_rows_by_context: dict) -> dict:
    """the files of a sample's F1R2 counts: its histogram tables, and an alt table for every context"""
    files = {
        sample + ".ref_histogram": histogram_table_text(sample, REF_HISTOGRAM_KEY_COLUMNS, depths, ref_rows),
        sample + ".alt_histogram": histogram_table_text(sample, ALT_HISTOGRAM_KEY_COLUMNS, depths, alt_rows),
    }
    for context in CONTEXTS:
        files[f"{sample}.{context}.alt_table"] = alt_table_text(sample, alt_table_rows_by_context.get(context, []))
    return files


def rows(sites: Sites) -> list:
    """the sites as sorted (alt base, alt count, alt F1R2 count, depth, number of sites)"""
    return sorted(tuple(key) + (count,) for key, count in zip(sites.keys_n4.tolist(), sites.counts_n.tolist()))


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
    depths = [1, 2, 3, 4]
    # the first input has two blocks of the genome with sites of ACG without alt reads
    ref_1 = [("chr1", 1, "ACG", [1, 2, 0, 4]), ("chr1", 1_000_001, "ACG", [0, 0, 3, 0])]
    ref_2 = [("chr1", 2_000_001, "ACG", [10, 0, 0, 1]), ("chr1", 2_000_001, "TTT", [0, 0, 5, 0])]
    alt_1 = [("chr1", 1, "ACG", "T", "F1R2", [0, 1, 0, 2])]
    alt_2 = [
        ("chr1", 2_000_001, "ACG", "T", "F1R2", [1, 0, 0, 0]),
        ("chr1", 2_000_001, "ACG", "A", "F2R1", [0, 0, 3, 0]),
    ]
    alt_table_1 = {"ACG": [(10, 5, 5, 4, "T"), (3, 5, 2, 4, "T")], "TTT": [(7, 2, 3, 1, "G")]}
    alt_table_2 = {"ACG": [(11, 5, 6, 4, "T"), (10, 5, 3, 4, "T")]}
    write_tar_gz(tmp_path / "1.tar.gz", f1r2_files("S", depths, ref_1, alt_1, alt_table_1))
    write_tar_gz(tmp_path / "2.tar.gz", f1r2_files("S", depths, ref_2, alt_2, alt_table_2))
    # a second sample in one input only
    write_tar_gz(tmp_path / "3.tar.gz", f1r2_files("OTHER", depths, ref_1, [], {}))

    counts_by_sample = read_f1r2_counts([tmp_path / "1.tar.gz", tmp_path / "2.tar.gz", tmp_path / "3.tar.gz"])
    assert counts_by_sample.keys() == {"S", "OTHER"}
    counts = counts_by_sample["S"]
    assert counts.max_depth == 4
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


def test_rounded_depths(tmp_path):
    # Mutect2's depths are exact up to 24 and rounded above
    def counts(depth, count):
        return [count if d == depth else 0 for d in ROUNDED_DEPTHS]

    ref = [("chr1", 1, "ACG", counts(30, 7)), ("chr1", 1_000_001, "ACG", counts(30, 2))]
    alt = [("chr1", 1_000_001, "ACG", "T", "F1R2", counts(24, 1))]
    write_tar_gz(tmp_path / "counts.tar.gz", f1r2_files("S", ROUNDED_DEPTHS, ref, alt, {}))

    counts_by_sample = read_f1r2_counts([tmp_path / "counts.tar.gz"])
    assert counts_by_sample["S"].max_depth == 200
    assert rows(counts_by_sample["S"].sites["ACG"]) == [(C, 0, 0, 30, 9), (T, 1, 1, 24, 1)]


def test_inputs_with_different_depths_are_rejected(tmp_path):
    write_tar_gz(tmp_path / "1.tar.gz", f1r2_files("S", [1, 2], [], [], {}))
    write_tar_gz(tmp_path / "2.tar.gz", f1r2_files("S", [1, 2, 3], [], [], {}))
    with pytest.raises(ValueError, match="different depths"):
        read_f1r2_counts([tmp_path / "1.tar.gz", tmp_path / "2.tar.gz"])


def test_histogram_table_without_depth_columns_is_rejected(tmp_path):
    files = f1r2_files("S", [1, 2], [], [], {})
    files["S.ref_histogram"] = histogram_table_text("S", REF_HISTOGRAM_KEY_COLUMNS, ["depth"], [])
    write_tar_gz(tmp_path / "counts.tar.gz", files)
    with pytest.raises(ValueError, match="and then depths"):
        read_f1r2_counts([tmp_path / "counts.tar.gz"])


def test_alt_table_of_an_older_gatk_is_rejected(tmp_path):
    files = f1r2_files("S", [1, 2], [], [], {})
    files["S.alt_table"] = alt_table_text("S", [(10, 5, 5, 4, "T")])
    write_tar_gz(tmp_path / "counts.tar.gz", files)
    with pytest.raises(ValueError, match="Older versions of GATK"):
        read_f1r2_counts([tmp_path / "counts.tar.gz"])


def test_alt_table_of_a_sample_without_histograms_is_rejected(tmp_path):
    files = f1r2_files("S", [1, 2], [], [], {})
    files["OTHER.ACG.alt_table"] = alt_table_text("OTHER", [(10, 5, 5, 4, "T")])
    write_tar_gz(tmp_path / "counts.tar.gz", files)
    with pytest.raises(ValueError, match="OTHER"):
        read_f1r2_counts([tmp_path / "counts.tar.gz"])


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
