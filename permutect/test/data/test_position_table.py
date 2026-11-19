import numpy as np
import pytest

from permutect.data.position_table import PositionTable


def sorted_table(keys: list[tuple[int, int]]) -> PositionTable:
    """
    A table of rows with the given position and allele keys, appended in two batches, whose value is each row's index in
    the keys, sorted
    """
    table = PositionTable([("value", np.int64)])
    rows = table.new_rows(len(keys))
    rows["position_key"] = [position for position, _ in keys]
    rows["allele_key"] = [alleles for _, alleles in keys]
    rows["value"] = np.arange(len(keys))
    table.append(rows[:2])
    table.append(rows[2:])
    table.sort()
    return table


def test_rows_are_found_in_order_of_position():
    # rows appended out of order are found in the order of the records of a sorted VCF
    table = sorted_table([(30, 1), (10, 2), (20, 5), (10, 1), (20, 3)])
    assert table.find_in_order(5, 1) is None  # before every row
    assert table.find_in_order(10, 1)["value"] == 3
    assert table.find_in_order(10, 2)["value"] == 1
    assert table.find_in_order(20, 4) is None  # a position with rows, but not with these alleles
    assert table.find_in_order(20, None) is None  # alleles that no datum has
    assert table.find_in_order(20, 3)["value"] == 4
    assert table.find_in_order(20, 5)["value"] == 2
    assert table.find_in_order(30, 1)["value"] == 0
    assert table.find_in_order(40, 1) is None  # after every row
    table.close()


def test_a_variant_found_in_order_gets_the_last_appended_of_rows_with_its_keys():
    # eg two long insertions that differ only after the 13 bases to which alleles are truncated
    table = sorted_table([(10, 1), (20, 1), (10, 1), (10, 1), (20, 2)])
    assert table.find_in_order(10, 1)["value"] == 3
    assert table.find_in_order(20, 1)["value"] == 1
    table.close()


def test_rows_must_be_found_in_order_of_position():
    # the rows are read in one pass alongside the VCF, which therefore must be sorted
    table = sorted_table([(10, 1), (20, 1)])
    table.find_in_order(20, 1)
    with pytest.raises(ValueError, match="sorted by position"):
        table.find_in_order(10, 1)
    table.close()


def test_a_table_without_rows_finds_nothing():
    table = PositionTable([("value", np.int64)])
    table.sort()
    assert table.find_in_order(10, 1) is None
    table.close()
