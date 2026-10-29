from __future__ import annotations

import math
import os
import random
import tarfile
import tempfile
from collections import defaultdict
from tempfile import NamedTemporaryFile
from typing import Generator
from typing import List
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


class MemoryMappedData:
    """
    wrapper for
        1) memory-mapped numpy file for stacked 1D data arrays.  The nth row of this array is the 1D array for the nth Datum.
        2) memory-mapped numpy file for 2D stacked reads array.  The rows of this array are the ref reads of the 0th Datum,
            the alt reads of the 0th Datum, the ref reads of the 1st Datum etc.

        NOTE: the memory-mapped files may be larger than necessary.  That is, there may be junk space in the files that
        does not correspond to actual data.  The num_data and num_reads tell us how far into the files is actual data.
    """

    def __init__(self, int_mmap, float_mmap, num_data, reads_mmap, num_reads):
        self.int_mmap = int_mmap
        self.float_mmap = float_mmap
        self.reads_mmap = reads_mmap
        self.num_data = num_data
        self.num_reads = num_reads

        # each datum's reads, if there are any, are as many as its ref and alt counts
        read_counts = [
            0 if reads_mmap is None else int(Datum(self.int_mmap[n], self.float_mmap[n]).get_read_count())
            for n in range(num_data)
        ]
        self.read_offsets = stacked_read_offsets(read_counts)
        if self.read_offsets[-1] != num_reads:
            raise ValueError(f"The data's ref and alt counts add up to {self.read_offsets[-1]} reads, not {num_reads}.")

    def __len__(self):
        return self.num_data

    def num_bytes(self):
        required_array_nbytes = self.int_mmap.nbytes + self.float_mmap.nbytes
        reads_nbytes = 0 if self.reads_mmap is None else self.reads_mmap.nbytes
        return required_array_nbytes + reads_nbytes

    def generate(
        self, num_folds: int = 1, used_folds: List[int] = None, label_probs_l=None
    ) -> Generator[Datum, None, None]:
        folds_set = None if used_folds is None else set(used_folds)
        print("Generating data from memory maps.")
        count = 0
        for idx in range(self.num_data):
            if used_folds is None or (idx % num_folds in folds_set):
                int_arr = self.int_mmap[idx]
                float_arr = self.float_mmap[idx]
                reads_arr = (
                    np.zeros((0, 0), dtype=COMPRESSED_READS_ARRAY_DTYPE)
                    if self.reads_mmap is None
                    else self.reads_mmap[self.read_offsets[idx] : self.read_offsets[idx + 1]]
                )
                is_compressed = reads_arr.dtype == COMPRESSED_READS_ARRAY_DTYPE
                datum = Datum(int_array=int_arr, float_array=float_arr, reads_re=reads_arr, compressed=is_compressed)
                if label_probs_l is None or random.random() < label_probs_l[datum.get(Data.LABEL)]:
                    yield datum
                    count += 1
        print(f"generated {count} objects.")

    def _restrict(self, reads_datum_source, estimated_proportion: float) -> MemoryMappedData:
        """
        New memory-mapped data holding the data from a generator over a subset of this one's data.  The estimated
        proportion of data kept sets the initial capacity, which grows as needed if the estimate is too low.
        """
        fudge_factor = 1.1
        estimated_num_data = int(self.num_data * estimated_proportion * fudge_factor)
        estimated_num_reads = int(self.num_reads * estimated_proportion * fudge_factor)
        return MemoryMappedData.from_generator(reads_datum_source, estimated_num_data, estimated_num_reads)

    def restrict_to_folds(self, num_folds: int, used_folds: List[int] = None, label_probs_l=None) -> MemoryMappedData:
        if used_folds is None:
            return self
        else:
            print(f"Restricting to folds {used_folds} out of {num_folds} total folds.")
            reads_datum_source = self.generate(
                num_folds=num_folds,
                used_folds=used_folds,
                label_probs_l=label_probs_l,
            )
            return self._restrict(reads_datum_source, estimated_proportion=len(used_folds) / num_folds)

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
            encoding = encode_variant(v, zero_based=True)
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

    # Load the list of objects back from the .npy file
    # Remember to set allow_pickle=True when loading as well
    @classmethod
    def load_from_tarfile(cls, data_tarfile) -> MemoryMappedData:
        loading_timer = Timer()
        temp_dir = tempfile.TemporaryDirectory()

        with tarfile.open(data_tarfile, "r") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    tar.extract(member, path=temp_dir.name)

        all_files = [os.path.abspath(os.path.join(temp_dir.name, p)) for p in os.listdir(temp_dir.name)]
        metadata_files = [f for f in all_files if f.endswith(SUFFIX_FOR_METADATA)]
        int_array_files = [f for f in all_files if f.endswith(SUFFIX_FOR_INT_MMAP)]
        float_array_files = [f for f in all_files if f.endswith(SUFFIX_FOR_FLOAT_MMAP)]
        reads_files = [f for f in all_files if f.endswith(SUFFIX_FOR_READS_MMAP)]

        assert len(metadata_files) == 1
        assert len(int_array_files) == 1
        assert len(float_array_files) == 1

        loaded_metadata = torch.load(metadata_files[0], weights_only=False)
        num_data, int_array_dim, float_array_dim, num_reads, reads_dim = loaded_metadata[:5]

        assert len(reads_files) == (0 if num_reads == 0 else 1)
        # NOTE: the original file may have had excess space due to the O(N) amortized growing scheme
        # if we load the same file with the actual num_data, as opposed to the capacity, it DOES work correctly
        int_mmap = np.lib.format.open_memmap(int_array_files[0], mode="r", shape=(num_data, int_array_dim))
        float_mmap = np.lib.format.open_memmap(float_array_files[0], mode="r", shape=(num_data, float_array_dim))
        reads_mmap = (
            None
            if num_reads == 0
            else np.lib.format.open_memmap(reads_files[0], mode="r", shape=(num_reads, reads_dim))
        )
        loading_timer.report("Time to load data from tarfile")

        result = cls(
            int_mmap=int_mmap,
            float_mmap=float_mmap,
            num_data=num_data,
            reads_mmap=reads_mmap,
            num_reads=num_reads,
        )
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

        return cls(
            int_mmap=int_mmap,
            float_mmap=float_mmap,
            num_data=num_data,
            reads_mmap=reads_mmap,
            num_reads=num_reads,
        )
