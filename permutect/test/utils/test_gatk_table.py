import io

import pytest

from permutect.utils.gatk_table import read_gatk_table

TABLE = (
    "#<METADATA>SAMPLE=tumor 1\n"
    "#<METADATA>OTHER=a=b\n"
    "contig\tposition\tallele_frequency\tbase\n"
    "20\t100\t0.1234567890123456789\tNA\n"
    "chrX\t5\t2.5e-05\tT\n"
)


def test_read_gatk_table():
    table = read_gatk_table(io.StringIO(TABLE), dtype={"contig": str})
    assert table.metadata == {"SAMPLE": "tumor 1", "OTHER": "a=b"}
    assert table.sample == "tumor 1"
    assert table.columns == ["contig", "position", "allele_frequency", "base"]
    # contigs stay strings, NA isn't missing, and decimals are parsed as Python parses them
    assert table.rows["contig"].tolist() == ["20", "chrX"]
    assert table.rows["base"].tolist() == ["NA", "T"]
    assert table.rows["position"].tolist() == [100, 5]
    assert table.rows["allele_frequency"].tolist() == [float("0.1234567890123456789"), float("2.5e-05")]


def test_read_gatk_table_in_chunks():
    chunks = read_gatk_table(io.StringIO(TABLE), chunksize=1, usecols=["position"]).rows
    assert [chunk["position"].tolist() for chunk in chunks] == [[100], [5]]


def test_table_without_rows():
    table = read_gatk_table(io.StringIO("#<METADATA>SAMPLE=s\na\tb\n"))
    assert table.columns == ["a", "b"]
    assert len(table.rows) == 0


def test_tables_without_their_columns_or_a_header_are_rejected():
    with pytest.raises(ValueError, match="does not have the columns"):
        read_gatk_table(io.StringIO(TABLE), usecols=["depth"])
    with pytest.raises(ValueError, match="no header"):
        read_gatk_table(io.StringIO("#<METADATA>SAMPLE=s\n"))
