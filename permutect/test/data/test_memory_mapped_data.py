import gzip
import io
import itertools
import os
import pickle
import shutil
import tarfile
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import psutil
import pytest
import torch
from intervaltree import IntervalTree
from torch.utils.data import DataLoader

from permutect import misc_utils
from permutect.data import memory_mapped_data as memory_mapped_data_module
from permutect.data import reads_dataset
from permutect.data.batch import Batch
from permutect.data.batch import BatchIndexedTensor
from permutect.data.count_binning import MAX_ALT_COUNT
from permutect.data.count_binning import MAX_REF_COUNT
from permutect.data.count_binning import MIN_ALT_COUNT
from permutect.data.count_binning import alt_count_bin_index
from permutect.data.count_binning import ref_count_bin_index
from permutect.data.datum import COMPRESSED_READS_ARRAY_DTYPE
from permutect.data.datum import NUMBER_OF_BYTES_IN_PACKED_READ
from permutect.data.datum import Data
from permutect.data.datum import Datum
from permutect.data.memory_mapped_data import SUFFIX_FOR_METADATA
from permutect.data.memory_mapped_data import MemoryMappedData
from permutect.data.memory_mapped_data import StackedData
from permutect.data.reads_dataset import ReadsDataset
from permutect.data.reads_dataset import bin_count_totals
from permutect.data.reads_dataset import count_totals_shape
from permutect.data.reads_dataset import record_count_totals
from permutect.test.test_file_names import PREPROCESSED_DATA
from permutect.test.test_misc_utils import VCF_HEADER
from permutect.test.test_misc_utils import make_datum as make_datum_with_alleles
from permutect.test.test_misc_utils import vcf_line
from permutect.utils.enums import Label
from permutect.utils.enums import Variation


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


@pytest.mark.parametrize("estimated_num_data, estimated_num_reads", [(0, 0), (2, 3), (100, 100)])
def test_from_chunks_matches_from_generator(estimated_num_data, estimated_num_reads):
    # writing chunks of data all at once gives the same memory maps as writing them datum by datum, as they grow or not
    data = [make_datum(index, index % 4) for index in range(10)]
    by_datum = MemoryMappedData.from_generator(data, estimated_num_data=10, estimated_num_reads=15)
    chunks = [StackedData.of(chunk) for chunk in (data[:3], data[3:4], data[4:])]
    by_chunk = MemoryMappedData.from_chunks(chunks, estimated_num_data, estimated_num_reads)
    for name in ("int_mmap", "float_mmap", "reads_mmap"):
        by_datum_array, by_chunk_array = getattr(by_datum, name), getattr(by_chunk, name)
        assert by_chunk_array.dtype == by_datum_array.dtype and by_chunk_array.tobytes() == by_datum_array.tobytes()
    assert_same_data(by_chunk, data)


def test_unused_capacity_is_left_out(tmp_path):
    # from_generator grows its memory maps beyond the data, as a list grows
    data = [make_datum(index, num_reads=3) for index in range(10)]
    memory_mapped_data = MemoryMappedData.from_generator(data, estimated_num_data=1000, estimated_num_reads=5000)
    tarfile = str(tmp_path / "data.tar")
    memory_mapped_data.save_to_tarfile(tarfile)
    loaded = MemoryMappedData.load_from_tarfile(tarfile)

    datum_bytes = data[0].get_int_array().nbytes + data[0].get_float_array().nbytes
    for mmd in (memory_mapped_data, loaded):
        assert len(mmd.int_mmap) == len(mmd.float_mmap) == 10 and len(mmd.reads_mmap) == 30
        assert mmd.num_bytes() == 10 * datum_bytes + data[0].reads_re.nbytes * 10
    assert_same_data(loaded, data)

    # the saved tarfile is as small as one of the same data made with exactly enough capacity
    exact_tarfile = str(tmp_path / "exact.tar")
    MemoryMappedData.from_generator(data, estimated_num_data=10, estimated_num_reads=30).save_to_tarfile(exact_tarfile)
    assert os.path.getsize(tarfile) == os.path.getsize(exact_tarfile)


def save_to_tarfile(data: list[Datum], tarfile_name: str):
    memory_mapped_data = MemoryMappedData.from_generator(data, estimated_num_data=len(data), estimated_num_reads=0)
    memory_mapped_data.save_to_tarfile(tarfile_name)


@pytest.mark.parametrize("block_bytes", [None, 64])
def test_saved_arrays_are_written_into_the_tarfile_as_np_save_writes_them(tmp_path, monkeypatch, block_bytes):
    # rather than saved to temporary files first and then copied into the tarfile, here in blocks of 64 bytes or not
    data = [make_datum(index, index % 4) for index in range(10)]
    saved = MemoryMappedData.from_generator(data, estimated_num_data=10, estimated_num_reads=20)
    if block_bytes is not None:
        monkeypatch.setattr(memory_mapped_data_module, "NPY_WRITE_BLOCK_BYTES", block_bytes)

    def no_temporary_files(*args, **kwargs):
        raise AssertionError("save_to_tarfile made a temporary file")

    monkeypatch.setattr(memory_mapped_data_module, "NamedTemporaryFile", no_temporary_files)
    tar = str(tmp_path / "data.tar")
    saved.save_to_tarfile(tar)

    with tarfile.open(tar) as archive:
        members = {member.name: archive.extractfile(member).read() for member in archive.getmembers()}
    arrays = {
        "int_array" + memory_mapped_data_module.SUFFIX_FOR_INT_MMAP: saved.int_mmap,
        "float_array" + memory_mapped_data_module.SUFFIX_FOR_FLOAT_MMAP: saved.float_mmap,
        "reads_array" + memory_mapped_data_module.SUFFIX_FOR_READS_MMAP: saved.reads_mmap,
    }
    for name, array in arrays.items():
        np_save_bytes = io.BytesIO()
        np.save(np_save_bytes, array)
        assert members[name] == np_save_bytes.getvalue(), name
    assert_same_data(MemoryMappedData.load_from_tarfile(tar), data)


def test_uncompressed_tarfile_is_mapped_in_place(tmp_path):
    # rather than extracted, which copied all the data, in every process of distributed training
    data = [make_datum(index, index % 4) for index in range(10)]
    tar = str(tmp_path / "data.tar")
    save_to_tarfile(data, tar)
    loaded = MemoryMappedData.load_from_tarfile(tar)
    assert all(os.path.samefile(mmap.filename, tar) for mmap in (loaded.int_mmap, loaded.float_mmap, loaded.reads_mmap))
    assert_same_data(loaded, data)


def test_compressed_tarfile_is_extracted(tmp_path):
    data = [make_datum(index, index % 4) for index in range(10)]
    tar = str(tmp_path / "data.tar")
    save_to_tarfile(data, tar)
    with open(tar, "rb") as uncompressed, gzip.open(tar + ".gz", "wb") as compressed:
        shutil.copyfileobj(uncompressed, compressed)
    assert_same_data(MemoryMappedData.load_from_tarfile(tar + ".gz"), data)


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


def test_reads_dataset_counts_its_data(monkeypatch):
    # in several blocks, and with sources beyond the first, which the counts grow to fit
    monkeypatch.setattr(reads_dataset, "COUNTING_BLOCK_SIZE", 3)
    data = [make_datum(index, index % 4, label=list(Label)[index % len(Label)]) for index in range(10)]
    for index, datum in enumerate(data):
        datum.set(Data.SOURCE, 2 * (index % 2))
        datum.set(Data.VARIANT_TYPE, index % len(Variation))
    dataset = ReadsDataset(MemoryMappedData.from_generator(data, estimated_num_data=10, estimated_num_reads=20))

    expected = torch.zeros(BatchIndexedTensor.shape_for(num_sources=3))
    expected_count_totals = torch.zeros(count_totals_shape(num_sources=3))
    for datum in data:
        source, label, variant_type = (int(datum.get(field)) for field in (Data.SOURCE, Data.LABEL, Data.VARIANT_TYPE))
        ref_bin = int(ref_count_bin_index(datum.get(Data.REF_COUNT)))
        alt_bin = int(alt_count_bin_index(datum.get(Data.ALT_COUNT)))
        expected[source, label, variant_type, ref_bin, alt_bin] += 1
        ref_count, alt_count = int(datum.get(Data.REF_COUNT)), int(datum.get(Data.ALT_COUNT))
        # an alt count of 0 is counted as MIN_ALT_COUNT, as in the bins
        alt_index = max(alt_count, MIN_ALT_COUNT) - MIN_ALT_COUNT
        expected_count_totals[source, label, variant_type, ref_count, alt_index] += 1
    assert torch.equal(dataset.totals_slvra, expected)
    assert torch.equal(dataset.count_totals_slvcd, expected_count_totals)

    assert dataset.num_read_features() == data[0].num_read_features()
    assert dataset.num_info_features() == len(data[0].get_info_1d())
    assert dataset.haplotypes_length() == data[0].get_haplotypes_array_size()


def test_counts_above_the_maximums_are_counted_at_the_top_counts_and_bins():
    datum = Datum(
        int_array=np.zeros(Data.NUM_SCALAR_INT_ELEMENTS), float_array=np.zeros(Data.NUM_SCALAR_FLOAT_ELEMENTS)
    )
    datum.set(Data.LABEL, Label.ARTIFACT)
    datum.set(Data.VARIANT_TYPE, Variation.SNV)
    datum.set(Data.REF_COUNT, MAX_REF_COUNT + 2)
    datum.set(Data.ALT_COUNT, MAX_ALT_COUNT + 5)

    count_totals_slvcd = torch.zeros(count_totals_shape(num_sources=1))
    count_totals_slvcd = record_count_totals(count_totals_slvcd, datum.get_int_array()[None, :])
    assert count_totals_slvcd[0, Label.ARTIFACT, Variation.SNV, -1, -1].item() == count_totals_slvcd.sum().item() == 1
    totals_slvra = bin_count_totals(count_totals_slvcd)
    assert totals_slvra[0, Label.ARTIFACT, Variation.SNV, -1, -1].item() == totals_slvra.sum().item() == 1


def test_binned_count_totals_add_up_the_counts_in_each_bin():
    torch.manual_seed(0)
    count_totals_slvcd = torch.randint(0, 10, count_totals_shape(num_sources=2)).float()
    expected = torch.zeros(BatchIndexedTensor.shape_for(num_sources=2))
    for ref_count, alt_count in itertools.product(range(MAX_REF_COUNT + 1), range(MIN_ALT_COUNT, MAX_ALT_COUNT + 1)):
        ref_bin, alt_bin = int(ref_count_bin_index(ref_count)), int(alt_count_bin_index(alt_count))
        expected[..., ref_bin, alt_bin] += count_totals_slvcd[..., ref_count, alt_count - MIN_ALT_COUNT]
    assert torch.equal(bin_count_totals(count_totals_slvcd), expected)


@pytest.mark.parametrize("folds_to_use", [None, list(range(9))])
def test_loading_a_dataset_does_not_loop_over_its_data(tmp_path, monkeypatch, folds_to_use):
    # making a Datum of each datum took about a minute per million data, in every process of distributed training
    data = [make_datum(index, index % 4) for index in range(50)]
    tar = str(tmp_path / "data.tar")
    MemoryMappedData.from_generator(data, estimated_num_data=50, estimated_num_reads=100).save_to_tarfile(tar)

    num_datum_inits = 0
    datum_init = Datum.__init__

    def counted_init(self, *args, **kwargs):
        nonlocal num_datum_inits
        num_datum_inits += 1
        datum_init(self, *args, **kwargs)

    monkeypatch.setattr(Datum, "__init__", counted_init)
    ReadsDataset(MemoryMappedData.load_from_tarfile(tar), num_folds=10, folds_to_use=folds_to_use)
    assert num_datum_inits <= 1


def iterate_as_worker(dataset: ReadsDataset, worker: int, num_workers: int, monkeypatch) -> list[int]:
    # the positions of the data that a DataLoader worker iterates over
    worker_info = SimpleNamespace(id=worker, num_workers=num_workers)
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: worker_info)
    return [int(datum.get(Data.POSITION)) for datum in dataset]


@pytest.mark.parametrize("num_folds, folds_to_use", [(1, None), (3, [0, 2])])
def test_reads_dataset_shards_are_disjoint_and_the_same_size(monkeypatch, num_folds, folds_to_use):
    # In distributed training every worker of every process iterates over its own shard.  The shards have the same size,
    # so that every process has the same number of batches of the same sizes, which drops fewer data than there are
    # shards.
    data = [make_datum(index, index % 3) for index in range(23)]
    memory_mapped_data = MemoryMappedData.from_generator(data, estimated_num_data=23, estimated_num_reads=23)
    in_folds = [index for index in range(23) if folds_to_use is None or index % num_folds in folds_to_use]
    world_size, num_workers = 2, 3
    shards = []
    for rank in range(world_size):
        dataset = ReadsDataset(memory_mapped_data, num_folds, folds_to_use, rank=rank, world_size=world_size)
        assert len(dataset) == len(in_folds) // world_size
        shards += [iterate_as_worker(dataset, worker, num_workers, monkeypatch) for worker in range(num_workers)]

    positions = [position for shard in shards for position in shard]
    assert len(positions) == len(set(positions)) and set(positions) <= set(in_folds)
    num_shards = world_size * num_workers
    assert [len(shard) for shard in shards] == [len(in_folds) // num_shards] * num_shards


def test_reads_dataset_shards_share_the_memory(monkeypatch, capsys):
    # all the processes and their workers load chunks into the same machine's memory at once, so each shard is loaded in
    # as many chunks as all the data in a single process
    data = [make_datum(index, num_reads) for index, num_reads in enumerate([2, 0, 3, 1, 4, 2, 1, 3])]
    memory_mapped_data = MemoryMappedData.from_generator(data, estimated_num_data=8, estimated_num_reads=16)
    memory = SimpleNamespace(available=4 * memory_mapped_data.num_bytes(), percent=50.0)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: memory)

    iterate_as_worker(ReadsDataset(memory_mapped_data), 0, 1, monkeypatch)
    iterate_as_worker(ReadsDataset(memory_mapped_data, rank=1, world_size=2), 0, 1, monkeypatch)
    chunk_messages = [line for line in capsys.readouterr().out.splitlines() if "chunks" in line]
    assert chunk_messages == ["Worker 0 will divide its portion of the data into 3 chunks."] * 2


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
        str(vcf), {0: "chr1"}, artifact_filters=set(), **segmentations
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


def test_vcf_annotation_records_repeat_lengths(tmp_path):
    # the repeat unit's and the STR's lengths come from Mutect2's RU and RPA, whose first count is the reference's, and
    # are 0 outside STRs
    repeat_header_lines = (
        '##INFO=<ID=RU,Number=1,Type=String,Description="Tandem repeat unit">\n'
        '##INFO=<ID=RPA,Number=R,Type=Integer,Description="Repeats per allele">\n'
    )
    vcf = tmp_path / "variants.vcf"
    vcf.write_text(
        VCF_HEADER.replace("#CHROM", repeat_header_lines + "#CHROM")
        + vcf_line("C", "T", pos=100, info="POPAF=3")
        + "\n"
        + vcf_line("CAG", "C", pos=200, info="POPAF=5;RU=AG;RPA=6,5")
        + "\n"
    )
    data = [make_datum_with_alleles(0, 100, "C", "T"), make_datum_with_alleles(0, 200, "CAG", "C")]
    memory_mapped_data = MemoryMappedData.from_generator(data, estimated_num_data=2, estimated_num_reads=6)
    annotated = memory_mapped_data.make_vcf_annotate_memory_mapped_data(str(vcf), {0: "chr1"}, artifact_filters=set())
    repeat_lengths = [
        (float(d.get(Data.REPEAT_UNIT_LENGTH)), float(d.get(Data.REPEAT_LENGTH))) for d in annotated.generate()
    ]
    assert repeat_lengths == [(0.0, 0.0), (2.0, 12.0)]


def test_data_saved_in_an_older_layout_fail_to_load(tmp_path):
    data = [make_datum(index, num_reads) for index, num_reads in enumerate([2, 1, 3])]
    path = tmp_path / "data.tar"
    MemoryMappedData.from_generator(data, estimated_num_data=3, estimated_num_reads=6).save_to_tarfile(path)
    assert_same_data(MemoryMappedData.load_from_tarfile(path), data)

    # the same data with the metadata of data saved before the layout was recorded, which had five entries
    old_path = tmp_path / "old.tar"
    with tarfile.open(path) as tar, tarfile.open(old_path, "w") as old_tar:
        for member in tar.getmembers():
            contents = tar.extractfile(member).read()
            if member.name.endswith(SUFFIX_FOR_METADATA):
                metadata = torch.load(io.BytesIO(contents), weights_only=False)
                old_metadata = io.BytesIO()
                torch.save(metadata[:5], old_metadata)
                contents = old_metadata.getvalue()
            member.size = len(contents)
            old_tar.addfile(member, io.BytesIO(contents))
    with pytest.raises(ValueError, match="layout version 1"):
        MemoryMappedData.load_from_tarfile(old_path)


def test_vcf_annotation_keys_follow_misc_utils(tmp_path, monkeypatch):
    # like the output join in filter_variants, both sides of the annotation join must use the misc_utils encodings, so
    # that changing the key format there cannot make every datum silently drop out of the annotated data
    original_encode = misc_utils.encode
    monkeypatch.setattr(misc_utils, "encode", lambda *args: "new format " + original_encode(*args))
    assert len(annotated_mafs(tmp_path)) == 2


@pytest.mark.parametrize("num_folds, folds_to_use", [(3, [0, 2]), (3, [1]), (1000, [7]), (3, None)])
def test_reads_dataset_selects_its_folds_without_copying(num_folds, folds_to_use, monkeypatch):
    # the folds' data are selected as they are iterated over, here in nine chunks, some of them empty, rather than copied
    # to new memory maps, which took as much disk as the data in every process of distributed training
    data = [make_datum(index, index % 4) for index in range(20)]
    memory_mapped_data = MemoryMappedData.from_generator(data, estimated_num_data=20, estimated_num_reads=40)
    dataset = ReadsDataset(memory_mapped_data, num_folds, folds_to_use)
    assert dataset.memory_mapped_data is memory_mapped_data

    memory = SimpleNamespace(available=memory_mapped_data.num_bytes(), percent=50.0)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: memory)
    iterated = sorted(dataset, key=lambda datum: datum.get(Data.POSITION))
    expected = [datum for index, datum in enumerate(data) if folds_to_use is None or index % num_folds in folds_to_use]
    assert len(dataset) == dataset.totals_slvra.sum().item() == len(expected)
    assert_same_data(MemoryMappedData.from_generator(iterated, estimated_num_data=20, estimated_num_reads=40), expected)


@pytest.mark.parametrize("folds_to_use", [None, [0, 2]])
def test_pickled_reads_dataset_maps_its_files_instead_of_copying_the_data(folds_to_use):
    # DataLoader workers started with spawn, the default on macOS, or forkserver, the default on Linux from Python 3.14,
    # receive a pickled dataset, whose memory maps used to be pickled as copies of all their data.  With or without folds
    # the data are memory maps of a tarfile's files.
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
