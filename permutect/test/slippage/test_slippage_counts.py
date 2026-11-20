import math

import pytest

from permutect.slippage.slippage_counts import NO_CHANGE_BIN
from permutect.slippage.slippage_counts import NOT_WHOLE_UNITS_BIN
from permutect.slippage.slippage_counts import read_slippage_counts

HEADER = (
    "contig\tstart\tend\tunit\tunits\tminus_4_or_more\tminus_3\tminus_2\tminus_1\tno_change\tplus_1\tplus_2\tplus_3\t"
    "plus_4_or_more\tnot_whole_units\ttop_alt_change\ttop_alt_af\tsecond_alt_change\tsecond_alt_af\n"
)

# rows of GATK CollectSTRSlippageCounts on NA12878 with gnomAD: a T x 11 with alleles of 1 and 3 more Ts, and a GATA x 4
# with 2 more GATAs, an allele gnomAD lacks, and no second allele
TABLE = (
    "#<METADATA>SAMPLE=NA12878\n"
    + HEADER
    + "20\t10002459\t10002469\tT\t11\t0\t0\t0\t0\t26\t4\t1\t4\t0\t0\t1\t0.013\t3\t0.229\n"
    + "20\t10037126\t10037141\tGATA\t4\t0\t0\t0\t0\t37\t0\t6\t0\t0\t12\t2\tNaN\t0\tNaN\n"
)

# more NA12878 rows: hom alts, a AAAC x 7 with no fragments of the reference and a T x 11 whose reference ties with its
# slippage to 2 fewer Ts, where only one bin has more fragments than the reference
KEPT_ROWS = (
    "20\t10006820\t10006847\tAAAC\t7\t0\t0\t0\t55\t0\t0\t0\t0\t0\t0\t-1\t0.769\t0\tNaN\n"
    + "20\t10214834\t10214844\tT\t11\t0\t0\t1\t70\t1\t0\t0\t0\t0\t0\t-1\t0.874\t-2\t0.001699\n"
)
# and rows where two bins have more: an A x 12 with alleles of 1 and 2 fewer As, a hom alt GATA x 3 with 11 fragments
# changed by part of a unit, and a hom alt T x 9 whose 3 fragments with 2 fewer Ts outnumber its 2 of the reference
DROPPED_ROWS = (
    "20\t10157870\t10157881\tA\t12\t0\t0\t31\t31\t1\t0\t0\t0\t0\t0\t-1\t0.367\t-2\t0.61\n"
    + "20\t10037111\t10037122\tGATA\t3\t0\t0\t0\t0\t0\t55\t0\t0\t0\t11\t1\t0.796\t0\tNaN\n"
    + "20\t10040813\t10040821\tT\t9\t0\t0\t3\t61\t2\t0\t0\t0\t0\t0\t-1\t0.451\t-2\tNaN\n"
)


def test_read_slippage_counts(tmp_path):
    path = tmp_path / "slippage-counts.table"
    path.write_text(TABLE)
    counts = read_slippage_counts(path)

    assert counts.sample == "NA12878"
    assert counts.num_strs() == 2
    assert counts.contigs == ["20", "20"]
    assert counts.starts_s.tolist() == [10002459, 10037126]
    assert counts.unit_lengths_s.tolist() == [1, 4]
    assert counts.repeat_lengths_s.tolist() == [11, 16]
    assert counts.counts_sk.tolist() == [[0, 0, 0, 0, 26, 4, 1, 4, 0, 0], [0, 0, 0, 0, 37, 0, 6, 0, 0, 12]]
    assert counts.counts_sk[:, NO_CHANGE_BIN].tolist() == [26, 37]
    assert counts.counts_sk[:, NOT_WHOLE_UNITS_BIN].tolist() == [0, 12]
    assert counts.depths_s().tolist() == [35, 55]
    assert counts.alt_changes_sa.tolist() == [[1, 3], [2, 0]]
    assert counts.alt_allele_frequencies_sa[0].tolist() == [0.013, 0.229]
    assert all(math.isnan(af) for af in counts.alt_allele_frequencies_sa[1].tolist())


def test_strs_where_the_reference_has_neither_the_most_nor_second_most_fragments_are_dropped(tmp_path):
    path = tmp_path / "slippage-counts.table"
    path.write_text(TABLE + DROPPED_ROWS + KEPT_ROWS)
    counts = read_slippage_counts(path)
    assert counts.starts_s.tolist() == [10002459, 10037126, 10006820, 10214834]
    assert counts.counts_sk[2:, NO_CHANGE_BIN].tolist() == [0, 1]


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
