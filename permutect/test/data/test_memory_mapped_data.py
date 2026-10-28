from collections import defaultdict

import numpy as np
import pytest
from intervaltree import IntervalTree

from permutect.data.datum import COMPRESSED_READS_ARRAY_DTYPE
from permutect.data.datum import NUMBER_OF_BYTES_IN_PACKED_READ
from permutect.data.datum import Data
from permutect.data.datum import Datum
from permutect.data.memory_mapped_data import MemoryMappedData
from permutect.test.test_misc_utils import VCF_HEADER
from permutect.test.test_misc_utils import make_datum as make_datum_with_alleles
from permutect.test.test_misc_utils import vcf_line
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


def annotated_mafs(tmp_path, **segmentations) -> list:
    # the MAF and normal MAF that VCF annotation gives an SNV at chr1:100 and a deletion at chr1:200
    vcf = tmp_path / "variants.vcf"
    vcf.write_text(
        VCF_HEADER
        + vcf_line("C", "T", pos=100, info="POPAF=3")
        + "\n"
        + vcf_line("CAG", "C", pos=200, info="POPAF=5")
        + "\n"
    )
    data = [make_datum_with_alleles(0, 100, "C", "T"), make_datum_with_alleles(0, 200, "CAG", "C")]
    memory_mapped_data = MemoryMappedData.from_generator(data, estimated_num_data=2, estimated_num_reads=6)
    annotated = memory_mapped_data.make_vcf_annotate_memory_mapped_data(
        str(vcf), {0: "chr1"}, filters_to_exclude=set(), artifact_filters=set(), **segmentations
    )
    return [(float(datum.get(Data.MAF)), float(datum.get(Data.NORMAL_MAF))) for datum in annotated.generate()]


def test_vcf_annotation_without_a_segmentation(tmp_path):
    for _ in range(2):
        assert annotated_mafs(tmp_path) == [(0.5, 0.5), (0.5, 0.5)]

    # mutable default arguments used to gain an empty tree for each contig looked up, shared by every later call
    for method in (MemoryMappedData.generate_vcf_annotated_data, MemoryMappedData.make_vcf_annotate_memory_mapped_data):
        assert not any(default for default in method.__defaults__ or ()), method.__name__

    segmentation = defaultdict(IntervalTree)
    segmentation["chr1"][90:150] = 0.25
    assert annotated_mafs(tmp_path, segmentation=segmentation) == [(0.25, 0.5), (0.5, 0.5)]


@pytest.mark.parametrize("num_folds, used_folds", [(3, [0, 2]), (3, [1]), (1000, [7])])
def test_restrict_to_folds(num_folds, used_folds):
    # with 1000 folds the estimated size rounds down to 0, and the capacity must grow to fit
    data = [make_datum(index, index % 4) for index in range(20)]
    memory_mapped_data = MemoryMappedData.from_generator(data, estimated_num_data=20, estimated_num_reads=40)
    restricted = memory_mapped_data.restrict_to_folds(num_folds, used_folds)
    assert_same_data(restricted, [datum for index, datum in enumerate(data) if index % num_folds in used_folds])

    assert memory_mapped_data.restrict_to_folds(num_folds, used_folds=None) is memory_mapped_data
