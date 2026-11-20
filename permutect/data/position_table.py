import tempfile

import numpy as np

# the error of a variant looked up after a variant at a later position
OUT_OF_ORDER_MESSAGE = (
    "filter_variants needs an input VCF sorted by position, with its contigs in the order of its header"
)


class PositionTable:
    """
    Rows of values for variants, keyed by their positions and alleles (see misc_utils.position_key and allele_key), in a
    temporary file rather than in memory, so that memory doesn't grow with the number of rows.  Rows are appended in
    batches, in any order.  Then they are sorted in place by position, alleles, and the order in which they were
    appended, and a variant's rows are found by its position and alleles, either in one pass for variants in order of
    position or by binary search for variants in any order.

    Variants whose alleles are the same once truncated, such as two long insertions that differ only after 13 bases,
    have the same keys, and their rows stay in the order in which they were appended.
    """

    def __init__(self, value_fields: list):
        """
        :param value_fields: the fields of the values in each row, as in a numpy structured dtype
        """
        key_fields = [("position_key", np.uint64), ("allele_key", np.uint64), ("sequence", np.uint64)]
        # aligned, so that every column is: numpy copies an unaligned column, such as the keys of rows of 89 bytes, to
        # search it, which made each binary search take as long as reading every key
        self.dtype = np.dtype(key_fields + value_fields, align=True)
        self._file = tempfile.NamedTemporaryFile(suffix=".position_table")
        self._num_rows = 0
        self._rows = None  # the sorted rows, memory-mapped, once they have all been appended
        self._position_keys = self._allele_keys = None  # columns of the sorted rows
        self._next_row = 0  # the first row not before the position of the last variant found in order
        self._last_position_key = -1

    def new_rows(self, num_rows: int) -> np.ndarray:
        """zeroed rows in memory, to fill and append"""
        return np.zeros(num_rows, dtype=self.dtype)

    def append(self, rows: np.ndarray):
        assert self._rows is None, "rows can't be appended after sorting"
        rows["sequence"] = np.arange(self._num_rows, self._num_rows + len(rows))
        rows.tofile(self._file)
        self._num_rows += len(rows)

    def sort(self):
        """
        Sort the rows by position, alleles and the order in which they were appended.  Quicksort sorts the memory-mapped
        rows in place, without copying them into memory, unlike the stable sort.
        """
        self._file.flush()
        if self._num_rows == 0:
            self._rows = self.new_rows(0)  # an empty file can't be memory-mapped
        else:
            self._rows = np.memmap(self._file.name, dtype=self.dtype, mode="r+", shape=(self._num_rows,))
            self._rows.sort(order=["position_key", "allele_key", "sequence"], kind="quicksort")
        self._position_keys, self._allele_keys = self._rows["position_key"], self._rows["allele_key"]

    def find_in_order(self, position_key: int, allele_key: int | None):
        """
        The last appended row of a variant, or None if there isn't one, in one pass through the rows: variants must be
        looked up in nondecreasing order of position, as the records of a VCF sorted by position are.  A variant with no
        allele key has no row.
        """
        assert self._rows is not None, "rows can only be found after sorting"
        if position_key < self._last_position_key:
            raise ValueError(OUT_OF_ORDER_MESSAGE)
        self._last_position_key = position_key
        while self._next_row < len(self._rows) and self._position_keys[self._next_row] < position_key:
            self._next_row += 1
        # the rows at this position, usually one, or a few at the site of a split multiallelic variant
        row, found = self._next_row, None
        while row < len(self._rows) and self._position_keys[row] == position_key:
            if allele_key is not None and self._allele_keys[row] == allele_key:
                found = row
            row += 1
        return None if found is None else self._rows[found]

    def rows_of(self, position_key: int, allele_key: int) -> np.ndarray:
        """
        A variant's rows, in the order in which they were appended, found by binary search, so that variants may be
        looked up in any order
        """
        assert self._rows is not None, "rows can only be found after sorting"
        start = int(np.searchsorted(self._position_keys, np.uint64(position_key)))
        # the rows at this position, usually one, or a few at the site of a split multiallelic variant
        end = start
        while end < len(self._rows) and self._position_keys[end] == position_key:
            end += 1
        rows = self._rows[start:end]
        return rows[rows["allele_key"] == allele_key]

    def close(self):
        """delete the temporary file"""
        self._rows = self._position_keys = self._allele_keys = None
        self._file.close()
