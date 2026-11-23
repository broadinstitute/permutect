import argparse
from enum import Enum
from typing import Generator

import numpy as np

from permutect import constants
from permutect.data.datum import COMPRESSED_READS_ARRAY_DTYPE
from permutect.data.datum import FLOAT_DTYPE
from permutect.data.datum import INTEGER_DTYPE
from permutect.data.datum import Data
from permutect.data.datum import set_data_field_column
from permutect.data.memory_mapped_data import MemoryMappedData
from permutect.data.memory_mapped_data import StackedData
from permutect.data.memory_mapped_data import stacked_read_offsets
from permutect.utils.enums import Label

# the data that are edited at once
DATA_PER_EDIT = 100000


class EditType(Enum):
    UNLABEL_ARTIFACTS = "unlabel_artifacts"
    UNLABEL_VARIANTS = "unlabel_variants"
    UNLABEL_EVERYTHING = "unlabel_everything"
    REMOVE_ARTIFACTS = "remove_artifacts"
    REMOVE_VARIANTS = "remove_variants"
    KEEP_EVERYTHING = "keep_everything"


def edited_chunks(memory_mapped_datas, edit_type: str, source: int) -> Generator[StackedData, None, None]:
    """
    The data of the original datasets with the edit applied, and with the new source if one is given, DATA_PER_EDIT at a
    time.  Their arrays have the dtypes that a Datum gives them, as they did when the data were edited one at a time.
    """
    edit = EditType(edit_type)
    for memory_mapped_data in memory_mapped_datas:
        read_offsets = memory_mapped_data.read_offsets
        for start in range(0, memory_mapped_data.num_data, DATA_PER_EDIT):
            end = min(start + DATA_PER_EDIT, memory_mapped_data.num_data)
            int_array_ve = np.array(memory_mapped_data.int_mmap[start:end], dtype=INTEGER_DTYPE)
            float_array_ve = np.array(memory_mapped_data.float_mmap[start:end], dtype=FLOAT_DTYPE)
            if source is not None:
                set_data_field_column(int_array_ve, float_array_ve, Data.SOURCE, np.full(end - start, source))

            labels = int_array_ve[:, Data.LABEL.idx]  # a view, so that relabeling edits the int array
            keep = np.ones(end - start, dtype=bool)
            if edit == EditType.UNLABEL_ARTIFACTS:
                labels[labels == Label.ARTIFACT] = Label.UNLABELED
            elif edit == EditType.UNLABEL_VARIANTS:
                labels[labels == Label.VARIANT] = Label.UNLABELED
            elif edit == EditType.UNLABEL_EVERYTHING:
                labels[:] = Label.UNLABELED
            elif edit == EditType.REMOVE_ARTIFACTS:
                keep = labels != Label.ARTIFACT
            elif edit == EditType.REMOVE_VARIANTS:
                keep = labels != Label.VARIANT

            # the reads of the data that are kept
            read_counts = np.diff(read_offsets[start : end + 1]).astype(np.int64)
            if memory_mapped_data.reads_mmap is None:
                reads_re = np.zeros((0, 0), dtype=COMPRESSED_READS_ARRAY_DTYPE)
            else:
                reads_re = memory_mapped_data.reads_mmap[read_offsets[start] : read_offsets[end]]
                reads_re = reads_re[np.repeat(keep, read_counts)]
            kept_read_offsets = stacked_read_offsets(read_counts[keep])
            yield StackedData(int_array_ve[keep], float_array_ve[keep], reads_re, kept_read_offsets)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="edit preprocessed datasets, eg unlabel or remove artifacts or variants or give them a new source, "
        "and combine them into one dataset"
    )
    parser.add_argument(
        "--" + constants.DATASET_EDIT_TYPE_NAME,
        required=True,
        choices=[edit_type.value for edit_type in EditType],
        help="how to edit the data's labels, or keep_everything to only change the source or combine datasets",
    )
    parser.add_argument("--" + constants.SOURCE_NAME, type=int, help="new source integer to apply")

    # input / output
    parser.add_argument(
        "--" + constants.TRAIN_TAR_NAME,
        nargs="+",
        required=True,
        help="tarfile(s) of training/validation datasets produced by preprocess_dataset.py",
    )
    parser.add_argument("--" + constants.OUTPUT_NAME, required=True, help="path to output dataset tarfile")

    return parser.parse_args(argv)


def main_without_parsing(args):
    original_tarfiles = getattr(args, constants.TRAIN_TAR_NAME)  # list of files
    output_tarfile = getattr(args, constants.OUTPUT_NAME)
    edit_type = getattr(args, constants.DATASET_EDIT_TYPE_NAME)
    new_source = getattr(args, constants.SOURCE_NAME)
    memory_mapped_datas = [
        MemoryMappedData.load_from_tarfile(data_tarfile=input_tarfile) for input_tarfile in original_tarfiles
    ]

    for mmd in memory_mapped_datas:
        print(f"Input dataset with {mmd.num_data} data and {mmd.num_reads} reads.")

    total_num_data = sum([mmd.num_data for mmd in memory_mapped_datas])
    total_num_reads = sum([mmd.num_reads for mmd in memory_mapped_datas])
    edited = MemoryMappedData.from_chunks(
        edited_chunks(memory_mapped_datas, edit_type, new_source),
        estimated_num_data=total_num_data,
        estimated_num_reads=total_num_reads,
    )
    print(f"Output dataset with {edited.num_data} data and {edited.num_reads} reads.")
    edited.save_to_tarfile(output_tarfile=output_tarfile)


def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == "__main__":
    main()
