import math

import pytest

from permutect.slippage.slippage_counts import NO_CHANGE_BIN
from permutect.slippage.slippage_counts import NOT_WHOLE_UNITS_BIN
from permutect.slippage.slippage_counts import read_slippage_counts

HEADER = (
    "contig\tstart\tend\tunit\tunits\tminus_4_or_more\tminus_3\tminus_2\tminus_1\tno_change\tplus_1\tplus_2\tplus_3\t"
    "plus_4_or_more\tnot_whole_units\ttop_alt_change\ttop_alt_af\tsecond_alt_change\tsecond_alt_af\n"
)

# rows of GATK CollectSTRSlippageCounts on NA12878 with gnomAD: a T x 11 with alleles of 1 and 3 more Ts, and a GATA x 3
# with one more GATA and no second allele
TABLE = (
    "#<METADATA>SAMPLE=NA12878\n"
    + HEADER
    + "20\t10002459\t10002469\tT\t11\t0\t0\t0\t0\t26\t4\t1\t4\t0\t0\t1\t0.013\t3\t0.229\n"
    + "20\t10037111\t10037122\tGATA\t3\t0\t0\t0\t0\t0\t55\t0\t0\t0\t11\t1\t0.796\t0\tNaN\n"
)


def test_read_slippage_counts(tmp_path):
    path = tmp_path / "slippage-counts.table"
    path.write_text(TABLE)
    counts = read_slippage_counts(path)

    assert counts.sample == "NA12878"
    assert counts.num_strs() == 2
    assert counts.contigs == ["20", "20"]
    assert counts.starts_s.tolist() == [10002459, 10037111]
    assert counts.unit_lengths_s.tolist() == [1, 4]
    assert counts.repeat_lengths_s.tolist() == [11, 12]
    assert counts.counts_sk.tolist() == [[0, 0, 0, 0, 26, 4, 1, 4, 0, 0], [0, 0, 0, 0, 0, 55, 0, 0, 0, 11]]
    assert counts.counts_sk[:, NO_CHANGE_BIN].tolist() == [26, 0]
    assert counts.counts_sk[:, NOT_WHOLE_UNITS_BIN].tolist() == [0, 11]
    assert counts.depths_s().tolist() == [35, 66]
    assert counts.alt_changes_sa.tolist() == [[1, 3], [1, 0]]
    assert counts.alt_allele_frequencies_sa[0].tolist() == [0.013, 0.229]
    assert counts.alt_allele_frequencies_sa[1, 0].item() == 0.796
    assert math.isnan(counts.alt_allele_frequencies_sa[1, 1].item())


def test_table_without_a_sample_is_an_error(tmp_path):
    path = tmp_path / "slippage-counts.table"
    path.write_text(TABLE.removeprefix("#<METADATA>SAMPLE=NA12878\n"))
    with pytest.raises(ValueError, match="doesn't name its sample"):
        read_slippage_counts(path)


def test_table_without_a_count_column_is_an_error(tmp_path):
    path = tmp_path / "slippage-counts.table"
    path.write_text(TABLE.replace("\tminus_3", "\tminus_three"))
    with pytest.raises(ValueError, match="minus_3"):
        read_slippage_counts(path)
