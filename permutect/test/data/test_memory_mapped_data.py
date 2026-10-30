import pickle
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import psutil
import pytest
from intervaltree import IntervalTree
from torch.utils.data import DataLoader

from permutect import misc_utils
from permutect.data.batch import Batch
from permutect.data.datum import COMPRESSED_READS_ARRAY_DTYPE
from permutect.data.datum import NUMBER_OF_BYTES_IN_PACKED_READ
from permutect.data.datum import Data
from permutect.data.datum import Datum
from permutect.data.memory_mapped_data import MemoryMappedData
from permutect.data.reads_dataset import ReadsDataset
from permutect.test.test_file_names import PREPROCESSED_DATA
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


def test_reads_must_match_counts():
    # the reads of each datum are located by its ref and alt counts, so a datum whose reads didn't match its counts
    # used to misalign the reads of every later datum without an error
    too_many_reads = make_datum(1, num_reads=3)
    too_many_reads.set(Data.ALT_COUNT, 2)
    with pytest.raises(ValueError):
        MemoryMappedData.from_generator([make_datum(0, 2), too_many_reads, make_datum(2, 4)], 3, 9)

    readless = make_datum(1, num_reads=0)
    readless.set(Data.ALT_COUNT, 5)
    with pytest.raises(ValueError):
        MemoryMappedData.from_generator([make_datum(0, 2), readless, make_datum(2, 4)], 3, 6)

    # data without any reads, like the posterior data of filter_variants, keep their counts
    memory_mapped_data = MemoryMappedData.from_generator(
        [readless, readless], estimated_num_data=2, estimated_num_reads=0
    )
    generated = list(memory_mapped_data.generate())
    assert [datum.get(Data.ALT_COUNT) for datum in generated] == [5, 5]
    assert all(len(datum.reads_re) == 0 for datum in generated)


def test_reads_dataset_gives_each_datum_its_reads(monkeypatch):
    read_counts = [2, 0, 3, 1, 4, 2, 1]
    data = [make_datum(index, num_reads) for index, num_reads in enumerate(read_counts)]
    dataset = ReadsDataset(MemoryMappedData.from_generator(data, estimated_num_data=7, estimated_num_reads=13))
    # load the data into RAM in three chunks, as when they don't all fit
    memory = SimpleNamespace(available=4 * dataset.memory_mapped_data.num_bytes(), percent=50.0)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: memory)

    iterated = sorted(dataset, key=lambda datum: datum.get(Data.POSITION))
    assert_same_data(MemoryMappedData.from_generator(iterated, 7, 13), data)


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


def test_vcf_annotation_keys_follow_misc_utils(tmp_path, monkeypatch):
    # like the output join in filter_variants, both sides of the annotation join must use the misc_utils encodings, so
    # that changing the key format there cannot make every datum silently drop out of the annotated data
    original_encode = misc_utils.encode
    monkeypatch.setattr(misc_utils, "encode", lambda *args: "new format " + original_encode(*args))
    assert len(annotated_mafs(tmp_path)) == 2


@pytest.mark.parametrize("num_folds, used_folds", [(3, [0, 2]), (3, [1]), (1000, [7])])
def test_restrict_to_folds(num_folds, used_folds):
    # with 1000 folds the estimated size rounds down to 0, and the capacity must grow to fit
    data = [make_datum(index, index % 4) for index in range(20)]
    memory_mapped_data = MemoryMappedData.from_generator(data, estimated_num_data=20, estimated_num_reads=40)
    restricted = memory_mapped_data.restrict_to_folds(num_folds, used_folds)
    assert_same_data(restricted, [datum for index, datum in enumerate(data) if index % num_folds in used_folds])

    assert memory_mapped_data.restrict_to_folds(num_folds, used_folds=None) is memory_mapped_data


@pytest.mark.parametrize("folds_to_use", [None, [0, 2]])
def test_pickled_reads_dataset_maps_its_files_instead_of_copying_the_data(folds_to_use):
    # DataLoader workers started with spawn, the default on macOS, or forkserver, the default on Linux from Python 3.14,
    # receive a pickled dataset, whose memory maps used to be pickled as copies of all their data.  Without folds the
    # data are memory maps of a tarfile's files, and with folds they are new memory maps.
    dataset = ReadsDataset(
        MemoryMappedData.load_from_tarfile(PREPROCESSED_DATA), num_folds=3, folds_to_use=folds_to_use
    )
    pickled = pickle.dumps(dataset)
    assert len(pickled) < dataset.memory_mapped_data.num_bytes() / 10

    unpickled = pickle.loads(pickled).memory_mapped_data
    for name in ("int_mmap", "float_mmap", "reads_mmap"):
        assert getattr(unpickled, name).tobytes() == getattr(dataset.memory_mapped_data, name).tobytes()


def test_data_loader_workers_started_with_spawn_read_all_the_data():
    # each worker maps the dataset's files itself, so they must stay on disk while the dataset is in use
    dataset = ReadsDataset(MemoryMappedData.load_from_tarfile(PREPROCESSED_DATA), num_folds=10, folds_to_use=[0, 1])
    loader = DataLoader(dataset, batch_size=64, collate_fn=Batch, num_workers=2, multiprocessing_context="spawn")
    assert sum(batch.size() for batch in loader) == len(dataset)
