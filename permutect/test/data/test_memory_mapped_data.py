import numpy as np
import pytest

from permutect.data.datum import COMPRESSED_READS_ARRAY_DTYPE
from permutect.data.datum import NUMBER_OF_BYTES_IN_PACKED_READ
from permutect.data.datum import Data
from permutect.data.datum import Datum
from permutect.data.memory_mapped_data import MemoryMappedData
from permutect.utils.enums import Label


def make_datum(index: int, num_reads: int, label: Label = Label.UNLABELED) -> Datum:
    # the position and the reads' contents identify the datum, so we can check that data are copied correctly
    reads_re = np.full((num_reads, NUMBER_OF_BYTES_IN_PACKED_READ), index % 256, dtype=COMPRESSED_READS_ARRAY_DTYPE)
    datum = Datum(
        int_array=np.zeros(Data.NUM_SCALAR_INT_ELEMENTS + 4),
        float_array=np.zeros(Data.NUM_SCALAR_FLOAT_ELEMENTS + 3),
        reads_re=reads_re,
        compressed=True,
    )
    datum.set(Data.ALT_COUNT, num_reads)
    datum.set(Data.LABEL, label)
    datum.set(Data.POSITION, index)
    return datum


def assert_same_data(memory_mapped_data: MemoryMappedData, expected: list[Datum]):
    generated = list(memory_mapped_data.generate())
    assert len(memory_mapped_data) == len(generated) == len(expected)
    assert memory_mapped_data.num_reads == sum(len(datum.reads_re) for datum in expected)
    for datum, expected_datum in zip(generated, expected):
        assert datum.get(Data.POSITION) == expected_datum.get(Data.POSITION)
        assert np.array_equal(datum.reads_re, expected_datum.reads_re)


def test_empty_generator(tmp_path):
    # e.g. filter_variants when no dataset variant is in the VCF, or edit_dataset removing everything
    empty = MemoryMappedData.from_generator(iter([]), estimated_num_data=10, estimated_num_reads=10)
    assert_same_data(empty, [])

    tarfile = str(tmp_path / "empty.tar")
    empty.save_to_tarfile(tarfile)
    assert_same_data(MemoryMappedData.load_from_tarfile(tarfile), [])


@pytest.mark.parametrize("estimated_num_data, estimated_num_reads", [(0, 0), (1, 1), (0, 100), (100, 0), (100, 100)])
def test_capacity_grows_to_fit(estimated_num_data, estimated_num_reads):
    # the datum with 40 reads needs more than double the capacity of its predecessors, and one datum has no reads
    read_counts = [1, 2, 40, 0, 3, 1, 5, 2, 1, 7]
    data = [make_datum(index, num_reads) for index, num_reads in enumerate(read_counts)]
    memory_mapped_data = MemoryMappedData.from_generator(data, estimated_num_data, estimated_num_reads)
    assert_same_data(memory_mapped_data, data)


def test_restrict_to_labeled_only_when_first_data_are_unlabeled():
    # the labeled proportion is estimated from the first 1001 data, which are all unlabeled here
    data = [make_datum(index, 1) for index in range(1002)]
    labeled = [make_datum(2000, 2, Label.ARTIFACT), make_datum(2001, 1, Label.VARIANT)]
    memory_mapped_data = MemoryMappedData.from_generator(
        data + labeled, estimated_num_data=2000, estimated_num_reads=2000
    )
    assert_same_data(memory_mapped_data.restrict_to_labeled_only(), labeled)

    empty = MemoryMappedData.from_generator(iter([]), estimated_num_data=0, estimated_num_reads=0)
    assert_same_data(empty.restrict_to_labeled_only(), [])
