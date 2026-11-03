from __future__ import annotations

import math
import os
import tarfile
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from tempfile import NamedTemporaryFile
from typing import Generator
from typing import Iterable
from typing import NamedTuple
from typing import Set

import cyvcf2
import numpy as np
import torch
from intervaltree import IntervalTree
from tqdm import tqdm

from permutect.data.datum import COMPRESSED_READS_ARRAY_DTYPE
from permutect.data.datum import FLOAT_DTYPE
from permutect.data.datum import INTEGER_DTYPE
from permutect.data.datum import Data
from permutect.data.datum import Datum
from permutect.data.datum import get_data_field
from permutect.misc_utils import Timer
from permutect.misc_utils import encode_datum
from permutect.misc_utils import encode_variant
from permutect.misc_utils import get_first_numeric_element
from permutect.misc_utils import overlapping_filters

# numpy.save appends .npy if the extension doesn't already include it.  We preempt this behavior.
SUFFIX_FOR_INT_MMAP = ".int_mmap.npy"
SUFFIX_FOR_FLOAT_MMAP = ".float_mmap.npy"
SUFFIX_FOR_READS_MMAP = ".reads_mmap.npy"
SUFFIX_FOR_METADATA = ".metadata.npy"


def grown_memory_map(old_mmap, required_rows: int, estimated_rows: int, row_shape: tuple, dtype, suffix: str):
    """
    Create a larger memory map and copy the old one's data into it, like the amortized O(N) append operation on lists.
    The first capacity is the estimate and later ones double, but capacity is always at least required_rows, even if
    the estimate is 0 or a single Datum has more reads than doubling makes room for.

    :return: the new memory map and its temporary file, which callers must keep because the file is deleted when it is
        garbage collected
    """
    old_capacity = 0 if old_mmap is None else len(old_mmap)
    capacity = max(required_rows, estimated_rows if old_mmap is None else 2 * old_capacity)
    new_file = NamedTemporaryFile(suffix=suffix)
    new_mmap = np.memmap(new_file.name, dtype=dtype, mode="w+", shape=(capacity,) + tuple(row_shape))
    if old_mmap is not None:
        new_mmap[:old_capacity] = old_mmap
    return new_mmap, new_file


def stacked_read_offsets(read_counts) -> np.ndarray:
    """
    Offsets into a stacked reads array whose rows are the reads of the 0th datum, the reads of the 1st datum etc: the
    reads of data start, start + 1. . . end - 1 are the rows offsets[start]:offsets[end].  In particular, the reads of
    datum n are the rows offsets[n]:offsets[n + 1].
    """
    # note: this can hold indices up to a bit over 4 billion, which is probably bigger than any training dataset we'll need
    offsets = np.zeros(shape=(len(read_counts) + 1,), dtype=np.uint32)
    offsets[1:] = np.cumsum(read_counts)
    return offsets


class MemoryMapFile(NamedTuple):
    """
    The file of a read-only memory map, which reopens it.  Pickling this rather than the memory map, which numpy pickles
    as a copy of its data, lets another process, such as a DataLoader worker started with spawn or forkserver, map the
    same file instead of receiving a copy of all the data.
    """

    filename: str
    dtype: np.dtype
    offset: int
    shape: tuple

    @classmethod
    def of(cls, mmap: np.memmap) -> MemoryMapFile:
        return cls(mmap.filename, mmap.dtype, mmap.offset, mmap.shape)

    def open(self) -> np.memmap:
        return np.memmap(self.filename, dtype=self.dtype, mode="r", offset=self.offset, shape=self.shape)


def files_by_suffix(files: list, name) -> dict:
    """
    the files of a saved MemoryMappedData, such as the members of its tarfile, by their suffixes, eg SUFFIX_FOR_INT_MMAP

    :param name: a function of a file that gives its name
    """
    suffixes = (SUFFIX_FOR_METADATA, SUFFIX_FOR_INT_MMAP, SUFFIX_FOR_FLOAT_MMAP, SUFFIX_FOR_READS_MMAP)
    result = {}
    for file in files:
        for suffix in suffixes:
            if name(file).endswith(suffix):
                assert suffix not in result, f"more than one file ends with {suffix}"
                result[suffix] = file
    assert all(suffix in result for suffix in suffixes[:3]), "the metadata, int array or float array file is missing"
    return result


def array_in_tarfile(data_tarfile, member: tarfile.TarInfo) -> np.ndarray:
    """
    A read-only memory map of a .npy file within an uncompressed tarfile, where it lies in the tarfile
    """
    with open(data_tarfile, "rb") as file:
        file.seek(member.offset_data)
        version = np.lib.format.read_magic(file)
        read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran_order, dtype = read_header(file)
        offset = file.tell()
    assert not fortran_order, "save_to_tarfile saves C-ordered arrays"
    # numpy can't map an array without any elements
    if math.prod(shape) == 0:
        return np.zeros(shape, dtype=dtype)
    return np.memmap(data_tarfile, dtype=dtype, mode="r", offset=offset, shape=shape)


@dataclass(frozen=True)
class StackedData:
    """
    Some data in memory: their int arrays and float arrays stacked as rows, and their reads stacked as rows, of which
    datum n's are rows read_offsets[n]:read_offsets[n + 1]
    """

    int_array_ve: np.ndarray
    float_array_ve: np.ndarray
    reads_re: np.ndarray
    read_offsets: np.ndarray

    @classmethod
    def of(cls, data: list[Datum]) -> StackedData:
        return cls(
            int_array_ve=np.vstack([datum.int_array for datum in data]),
            float_array_ve=np.vstack([datum.float_array for datum in data]),
            reads_re=np.vstack([datum.reads_re for datum in data]),
            read_offsets=stacked_read_offsets([len(datum.reads_re) for datum in data]),
        )

    def __len__(self) -> int:
        return len(self.int_array_ve)


class MemoryMappedData:
    """
    wrapper for
        1) memory-mapped numpy file for stacked 1D data arrays.  The nth row of this array is the 1D array for the nth Datum.
        2) memory-mapped numpy file for 2D stacked reads array.  The rows of this array are the ref reads of the 0th Datum,
            the alt reads of the 0th Datum, the ref reads of the 1st Datum etc.

        NOTE: the memory-mapped files may be larger than necessary, with junk space after the data from growing them
        as data are added.  The arrays here are views of only the first num_data and num_reads rows.
    """

    def __init__(self, int_mmap, float_mmap, num_data, reads_mmap, num_reads):
        # leave out the files' unused capacity, which would otherwise be saved and counted by num_bytes()
        self.int_mmap = int_mmap[:num_data]
        self.float_mmap = float_mmap[:num_data]
        self.reads_mmap = None if reads_mmap is None else reads_mmap[:num_reads]
        self.num_data = num_data
        self.num_reads = num_reads

        # each datum's reads, if there are any, are as many as its ref and alt counts
        if reads_mmap is None:
            read_counts = np.zeros(num_data, dtype=np.int64)
        else:
            ref_counts = get_data_field(self.int_mmap, None, Data.REF_COUNT).astype(np.int64)
            read_counts = ref_counts + get_data_field(self.int_mmap, None, Data.ALT_COUNT)
        self.read_offsets = stacked_read_offsets(read_counts)
        if self.read_offsets[-1] != num_reads:
            raise ValueError(f"The data's ref and alt counts add up to {self.read_offsets[-1]} reads, not {num_reads}.")

    # a pickled copy maps the same files.  Their temporary files or directory stay with the original, which keeps the
    # files on disk while it is in use.
    def __getstate__(self):
        state = {key: value for key, value in self.__dict__.items() if key not in ("_temp_dir", "_temp_files")}
        for key in ("int_mmap", "float_mmap", "reads_mmap"):
            if isinstance(state[key], np.memmap):
                state[key] = MemoryMapFile.of(state[key])
        return state

    def __setstate__(self, state):
        for key in ("int_mmap", "float_mmap", "reads_mmap"):
            if isinstance(state[key], MemoryMapFile):
                state[key] = state[key].open()
        self.__dict__.update(state)

    def __len__(self):
        return self.num_data

    def num_bytes(self):
        required_array_nbytes = self.int_mmap.nbytes + self.float_mmap.nbytes
        reads_nbytes = 0 if self.reads_mmap is None else self.reads_mmap.nbytes
        return required_array_nbytes + reads_nbytes

    def generate(self) -> Generator[Datum, None, None]:
        print("Generating data from memory maps.")
        for idx in range(self.num_data):
            yield self.get_datum(idx)
        print(f"generated {self.num_data} objects.")

    def get_datum(self, idx: int) -> Datum:
        reads_arr = (
            np.zeros((0, 0), dtype=COMPRESSED_READS_ARRAY_DTYPE)
            if self.reads_mmap is None
            else self.reads_mmap[self.read_offsets[idx] : self.read_offsets[idx + 1]]
        )
        is_compressed = reads_arr.dtype == COMPRESSED_READS_ARRAY_DTYPE
        return Datum(self.int_mmap[idx], self.float_mmap[idx], reads_re=reads_arr, compressed=is_compressed)

    def _restrict(self, reads_datum_source, estimated_proportion: float) -> MemoryMappedData:
        """
        New memory-mapped data holding the data from a generator over a subset of this one's data.  The estimated
        proportion of data kept sets the initial capacity, which grows as needed if the estimate is too low.
        """
        fudge_factor = 1.1
        estimated_num_data = int(self.num_data * estimated_proportion * fudge_factor)
        estimated_num_reads = int(self.num_reads * estimated_proportion * fudge_factor)
        return MemoryMappedData.from_generator(reads_datum_source, estimated_num_data, estimated_num_reads)

    def restrict_to_labeled_only(self) -> MemoryMappedData:
        print("Restricting dataset to labeled data only.")
        labeled_count, total = 0, 0
        # estimated the proportion of labeled data
        for n, datum in enumerate(self.generate()):
            if n > 1000:
                break
            total += 1
            labeled_count += 1 if datum.is_labeled() else 0

        labeled_proportion = labeled_count / total if total > 0 else 0
        reads_datum_source = (datum for datum in self.generate() if datum.is_labeled())
        return self._restrict(reads_datum_source, estimated_proportion=labeled_proportion)

    """
    Add log allele frequency (AF), minor allele frequency (MAF), and normal minor allele frequency (normal MAF) to the
    float array of the output MemoryMappedData using information in a VCF and a segmentation.
    
    Additionally, skip data that have a given set of filters in the VCF, and mark data that have a given set of
    artifact filters in the VCF as known artifacts with an infinite cached artifact logit.
    """

    def generate_vcf_annotated_data(
        self,
        input_vcf,
        contig_index_to_name_map,
        filters_to_exclude: Set[str],
        artifact_filters: Set[str],
        segmentation=None,
        normal_segmentation=None,
    ) -> Generator[Datum, None, None]:
        # a new empty segmentation for each call: lookups add an empty tree for each contig to a defaultdict
        segmentation = defaultdict(IntervalTree) if segmentation is None else segmentation
        normal_segmentation = defaultdict(IntervalTree) if normal_segmentation is None else normal_segmentation

        log_allele_frequencies = {}
        encodings_to_exclude = set()
        artifact_encodings = set()

        print("recording filters and allele frequencies from input VCF")
        pbar = tqdm(enumerate(cyvcf2.VCF(input_vcf)), mininterval=60)
        for n, v in pbar:
            encoding = encode_variant(v)
            # POPAF is -log10(AF); we convert to natural log without exponentiating to avoid underflow
            log_allele_frequencies[encoding] = -math.log(10) * get_first_numeric_element(v, "POPAF")
            if overlapping_filters(v, filters_to_exclude):
                encodings_to_exclude.add(encoding)
            if overlapping_filters(v, artifact_filters):
                artifact_encodings.add(encoding)

        for datum in self.generate():
            contig_name = contig_index_to_name_map[datum.get(Data.CONTIG)]
            position = datum.get(Data.POSITION)
            encoding = encode_datum(datum, contig_index_to_name_map)

            if encoding in log_allele_frequencies and encoding not in encodings_to_exclude:
                # NOTE: we copy the float array because it needs to be modified
                new_datum = Datum(
                    int_array=datum.int_array,
                    float_array=datum.float_array.copy(),
                    reads_re=datum.reads_re,
                    compressed=datum.reads_re.dtype == COMPRESSED_READS_ARRAY_DTYPE,
                )

                # these are default dicts, so if there's no segmentation for the contig we will get no overlaps but not an error
                # For a general IntervalTree there is a list of potentially multiple overlaps but here there is either one or zero
                log_allele_frequency = log_allele_frequencies[encoding]
                segmentation_overlaps = segmentation[contig_name][position]
                normal_segmentation_overlaps = normal_segmentation[contig_name][position]
                maf = list(segmentation_overlaps)[0].data if segmentation_overlaps else 0.5
                normal_maf = list(normal_segmentation_overlaps)[0].data if normal_segmentation_overlaps else 0.5

                new_datum.set(Data.LOG_ALLELE_FREQUENCY, log_allele_frequency)
                new_datum.set(Data.MAF, maf)
                new_datum.set(Data.NORMAL_MAF, normal_maf)
                if encoding in artifact_encodings:
                    new_datum.set(Data.CACHED_ARTIFACT_LOGIT, np.inf)
                yield new_datum

    def make_vcf_annotate_memory_mapped_data(
        self,
        input_vcf,
        contig_index_to_name_map,
        filters_to_exclude: Set[str],
        artifact_filters: Set[str],
        segmentation=None,
        normal_segmentation=None,
    ) -> MemoryMappedData:
        generator = self.generate_vcf_annotated_data(
            input_vcf=input_vcf,
            contig_index_to_name_map=contig_index_to_name_map,
            filters_to_exclude=filters_to_exclude,
            artifact_filters=artifact_filters,
            segmentation=segmentation,
            normal_segmentation=normal_segmentation,
        )
        return MemoryMappedData.from_generator(
            reads_datum_source=generator,
            estimated_num_data=self.num_data,
            estimated_num_reads=self.num_reads,
        )

    def save_to_tarfile(self, output_tarfile):
        """
        It seems a little odd to save to disk when memory-mapped files are already on disk, but:
            1) the files don't know their dtype and shape
            2) the files don't know how much of the data are actually used
            3) the files might be temporary files and won't persist after the Python program executes
            4) it's convenient to package things as a single tarfile
        :return:
        """
        # num_data, data dimension; num_reads, reads dimension
        metadata = np.array(
            [
                self.num_data,
                self.int_mmap.shape[-1],
                self.float_mmap.shape[-1],
                self.num_reads,
                0 if self.reads_mmap is None else self.reads_mmap.shape[-1],
            ],
            dtype=np.uint32,
        )
        metadata_file = NamedTemporaryFile(suffix=SUFFIX_FOR_METADATA)
        torch.save(metadata, metadata_file.name)

        # For some reason, self.data_mmap.filename and self.reads_mmap.filename point to an empty file.  I have no clue why this is,
        # so at risk of redundant copying I just use numpy's save function, followed later by numpy.open_memmap
        int_array_file = NamedTemporaryFile(suffix=SUFFIX_FOR_INT_MMAP)
        np.save(int_array_file.name, self.int_mmap)

        float_array_file = NamedTemporaryFile(suffix=SUFFIX_FOR_FLOAT_MMAP)
        np.save(float_array_file.name, self.float_mmap)

        if self.reads_mmap is not None:
            reads_file = NamedTemporaryFile(suffix=SUFFIX_FOR_READS_MMAP)
            np.save(reads_file.name, self.reads_mmap)

        with tarfile.open(output_tarfile, "w") as output_tar:
            output_tar.add(metadata_file.name, arcname=("metadata" + SUFFIX_FOR_METADATA))
            output_tar.add(int_array_file.name, arcname=("int_array" + SUFFIX_FOR_INT_MMAP))
            output_tar.add(float_array_file.name, arcname=("float_array" + SUFFIX_FOR_FLOAT_MMAP))
            if self.reads_mmap is not None:
                output_tar.add(reads_file.name, arcname=("reads_array" + SUFFIX_FOR_READS_MMAP))

    @classmethod
    def load_from_tarfile(cls, data_tarfile) -> MemoryMappedData:
        """
        The arrays of an uncompressed tarfile, which save_to_tarfile writes, are memory-mapped where they lie in it, so
        that loading copies nothing and the processes of distributed training share one copy in memory.  A compressed
        tarfile's files are extracted to a temporary directory.
        """
        loading_timer = Timer()
        temp_dir = None
        try:
            with tarfile.open(data_tarfile, "r:") as tar:  # only an uncompressed tarfile
                files = files_by_suffix([member for member in tar.getmembers() if member.isfile()], lambda m: m.name)
                loaded_metadata = torch.load(tar.extractfile(files[SUFFIX_FOR_METADATA]), weights_only=False)
            open_array = lambda suffix: array_in_tarfile(data_tarfile, files[suffix])
        except tarfile.ReadError:
            temp_dir = tempfile.TemporaryDirectory()
            with tarfile.open(data_tarfile, "r") as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        tar.extract(member, path=temp_dir.name)
            paths = [os.path.abspath(os.path.join(temp_dir.name, p)) for p in os.listdir(temp_dir.name)]
            files = files_by_suffix(paths, lambda path: path)
            loaded_metadata = torch.load(files[SUFFIX_FOR_METADATA], weights_only=False)
            open_array = lambda suffix: np.lib.format.open_memmap(files[suffix], mode="r")

        num_data, int_array_dim, float_array_dim, num_reads, reads_dim = loaded_metadata[:5]
        assert (SUFFIX_FOR_READS_MMAP in files) == (num_reads != 0)
        # files saved before only the data were saved may have unused capacity after the data, which __init__ leaves out
        int_mmap = open_array(SUFFIX_FOR_INT_MMAP)
        float_mmap = open_array(SUFFIX_FOR_FLOAT_MMAP)
        reads_mmap = None if num_reads == 0 else open_array(SUFFIX_FOR_READS_MMAP)
        loading_timer.report("Time to load data from tarfile")

        result = cls(
            int_mmap=int_mmap,
            float_mmap=float_mmap,
            num_data=num_data,
            reads_mmap=reads_mmap,
            num_reads=num_reads,
        )
        if temp_dir is not None:
            result._temp_dir = temp_dir  # prevent GC of temp dir while mmaps are in use
        return result

    @classmethod
    def from_generator(cls, reads_datum_source, estimated_num_data, estimated_num_reads) -> MemoryMappedData:
        """
        Write Datum objects to memory maps.  We set the file sizes to initial guesses but if these are outgrown we copy
        data to larger files, just like the amortized O(N) append operation on lists.

        :param reads_datum_source: an Iterable or Generator of Datum
        :param estimated_num_data: initial estimate of how much capacity is needed
        :param estimated_num_reads:
        :return:
        """
        num_data, num_reads = 0, 0
        int_mmap, float_mmap, reads_mmap = None, None, None
        int_file, float_file, reads_file = None, None, None

        datum: Datum
        for datum in reads_datum_source:
            int_array = datum.get_int_array()
            float_array = datum.get_float_array()
            reads_array = datum.get_reads_array_re()  # this works both for raw data and compressed reads
            if len(reads_array) not in (0, datum.get_read_count()):
                raise ValueError(
                    f"A datum has {len(reads_array)} reads but ref and alt counts of {datum.get_read_count()}."
                )

            num_data += 1
            num_reads += len(reads_array)

            # set capacity to initial estimate or grow it, create new file and mmap, copy old data
            if num_data > (0 if int_mmap is None else len(int_mmap)):
                int_mmap, int_file = grown_memory_map(
                    int_mmap, num_data, estimated_num_data, int_array.shape, int_array.dtype, SUFFIX_FOR_INT_MMAP
                )
                float_mmap, float_file = grown_memory_map(
                    float_mmap,
                    num_data,
                    estimated_num_data,
                    float_array.shape,
                    float_array.dtype,
                    SUFFIX_FOR_FLOAT_MMAP,
                )

            # likewise for reads
            if num_reads > (0 if reads_mmap is None else len(reads_mmap)):
                reads_mmap, reads_file = grown_memory_map(
                    reads_mmap,
                    num_reads,
                    estimated_num_reads,
                    reads_array.shape[1:],
                    reads_array.dtype,
                    SUFFIX_FOR_READS_MMAP,
                )

            # write new data
            int_mmap[num_data - 1] = int_array
            float_mmap[num_data - 1] = float_array
            if len(reads_array) > 0:
                reads_mmap[num_reads - len(reads_array) : num_reads] = reads_array

        return cls._from_written_memory_maps(
            num_data, num_reads, (int_mmap, float_mmap, reads_mmap), (int_file, float_file, reads_file)
        )

    @classmethod
    def from_chunks(cls, chunks: Iterable[StackedData], estimated_num_data, estimated_num_reads) -> MemoryMappedData:
        """
        Like from_generator, but for chunks of data, each of which is written to the memory maps all at once rather than
        datum by datum
        """
        num_data, num_reads = 0, 0
        int_mmap, float_mmap, reads_mmap = None, None, None
        int_file, float_file, reads_file = None, None, None

        chunk: StackedData
        for chunk in chunks:
            int_array_ve, float_array_ve, reads_re = chunk.int_array_ve, chunk.float_array_ve, chunk.reads_re
            end, reads_end = num_data + len(int_array_ve), num_reads + len(reads_re)
            if end > (0 if int_mmap is None else len(int_mmap)):
                int_mmap, int_file = grown_memory_map(
                    int_mmap, end, estimated_num_data, int_array_ve.shape[1:], int_array_ve.dtype, SUFFIX_FOR_INT_MMAP
                )
                float_mmap, float_file = grown_memory_map(
                    float_mmap,
                    end,
                    estimated_num_data,
                    float_array_ve.shape[1:],
                    float_array_ve.dtype,
                    SUFFIX_FOR_FLOAT_MMAP,
                )
            if reads_end > (0 if reads_mmap is None else len(reads_mmap)):
                reads_mmap, reads_file = grown_memory_map(
                    reads_mmap,
                    reads_end,
                    estimated_num_reads,
                    reads_re.shape[1:],
                    reads_re.dtype,
                    SUFFIX_FOR_READS_MMAP,
                )

            int_mmap[num_data:end] = int_array_ve
            float_mmap[num_data:end] = float_array_ve
            if len(reads_re) > 0:
                reads_mmap[num_reads:reads_end] = reads_re
            num_data, num_reads = end, reads_end

        return cls._from_written_memory_maps(
            num_data, num_reads, (int_mmap, float_mmap, reads_mmap), (int_file, float_file, reads_file)
        )

    @classmethod
    def _from_written_memory_maps(cls, num_data, num_reads, memory_maps, files) -> MemoryMappedData:
        """
        MemoryMappedData of the memory maps that from_generator or from_chunks wrote, reopened read-only, which keeps their
        temporary files
        """
        int_mmap, float_mmap, reads_mmap = memory_maps
        for memmap in (int_mmap, float_mmap, reads_mmap):
            if memmap is not None:
                memmap.flush()

        # re-open new mmap objects in 'r' (read-only) mode for faster access
        if num_data == 0:
            # with no Datum the array widths are unknown, and an empty file can't be memory-mapped
            int_mmap, float_mmap = np.zeros((0, 0), dtype=INTEGER_DTYPE), np.zeros((0, 0), dtype=FLOAT_DTYPE)
        else:
            int_mmap = np.memmap(int_mmap.filename, dtype=int_mmap.dtype, mode="r", shape=int_mmap.shape)
            float_mmap = np.memmap(float_mmap.filename, dtype=float_mmap.dtype, mode="r", shape=float_mmap.shape)
        if reads_mmap is not None:
            reads_mmap = np.memmap(reads_mmap.filename, dtype=reads_mmap.dtype, mode="r", shape=reads_mmap.shape)

        result = cls(
            int_mmap=int_mmap,
            float_mmap=float_mmap,
            num_data=num_data,
            reads_mmap=reads_mmap,
            num_reads=num_reads,
        )
        # keep the files on disk, so that a pickled copy can map them
        result._temp_files = [file for file in files if file is not None]
        return result
