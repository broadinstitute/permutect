import argparse
from enum import Enum

import numpy as np

from permutect import constants
from permutect.data.datum import FLOAT_DTYPE
from permutect.data.datum import INTEGER_DTYPE
from permutect.data.datum import Data
from permutect.data.datum import set_data_field_column
from permutect.data.memory_mapped_data import ArrayRows
from permutect.data.memory_mapped_data import MemoryMappedData
from permutect.data.memory_mapped_data import write_data_tarfile
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


# the labels that each edit removes the data of, or makes UNLABELED
REMOVED_LABELS = {EditType.REMOVE_ARTIFACTS: [Label.ARTIFACT], EditType.REMOVE_VARIANTS: [Label.VARIANT]}
UNLABELED_LABELS = {
    EditType.UNLABEL_ARTIFACTS: [Label.ARTIFACT],
    EditType.UNLABEL_VARIANTS: [Label.VARIANT],
    EditType.UNLABEL_EVERYTHING: list(Label),
}


def save_edited_data(memory_mapped_datas, edit_type: str, source: int, output_tarfile) -> tuple[int, int]:
    """
    Write the data of the original datasets, with the edit applied and with the new source if one is given, straight
    into the output tarfile, DATA_PER_EDIT at a time, rather than into memory maps that are then copied into it.  The
    arrays have the dtypes that a Datum gives them, as they did when the data were edited one at a time.  The size of a
    tarfile's member precedes it, so the data that are kept are found before each array is written in turn.

    :return: the numbers of data and of reads written
    """
    edit = EditType(edit_type)
    layouts = {
        (data.int_mmap.shape[1:], data.float_mmap.shape[1:], reads_layout(data))
        for data in memory_mapped_datas
        if data.num_data > 0
    }
    if len(layouts) > 1:
        raise ValueError(f"The datasets' arrays have different shapes or dtypes: {layouts}.")
    int_row_shape, float_row_shape, reads_dtype_and_row_shape = layouts.pop() if layouts else ((0,), (0,), None)

    # the blocks of each dataset's data, and which of their data are kept
    blocks = [
        (data, start, end, ~np.isin(data.int_mmap[start:end, Data.LABEL.idx], REMOVED_LABELS.get(edit, [])))
        for data, start, end in data_blocks(memory_mapped_datas)
    ]

    def kept_int_array(data: MemoryMappedData, start: int, end: int, keep: np.ndarray) -> np.ndarray:
        int_array_ve = np.array(data.int_mmap[start:end], dtype=INTEGER_DTYPE)
        labels = int_array_ve[:, Data.LABEL.idx]  # a view, so that relabeling edits the int array
        labels[np.isin(labels, UNLABELED_LABELS.get(edit, []))] = Label.UNLABELED
        if source is not None:
            set_data_field_column(int_array_ve, None, Data.SOURCE, np.full(end - start, source))
        return int_array_ve[keep]

    def kept_float_array(data: MemoryMappedData, start: int, end: int, keep: np.ndarray) -> np.ndarray:
        return np.array(data.float_mmap[start:end], dtype=FLOAT_DTYPE)[keep]

    def kept_reads(data: MemoryMappedData, start: int, end: int, keep: np.ndarray) -> np.ndarray:
        reads_re = data.reads_mmap[data.read_offsets[start] : data.read_offsets[end]]
        return reads_re[np.repeat(keep, read_counts(data, start, end))]

    num_data = sum(int(np.sum(keep)) for *_, keep in blocks)
    int_rows = ArrayRows(INTEGER_DTYPE, num_data, int_row_shape, (kept_int_array(*block) for block in blocks))
    float_rows = ArrayRows(FLOAT_DTYPE, num_data, float_row_shape, (kept_float_array(*block) for block in blocks))
    num_reads, reads_rows = 0, None
    if reads_dtype_and_row_shape is not None:
        num_reads = sum(int(read_counts(data, start, end)[keep].sum()) for data, start, end, keep in blocks)
        if num_reads > 0:
            reads_dtype, reads_row_shape = reads_dtype_and_row_shape
            reads_rows = ArrayRows(reads_dtype, num_reads, reads_row_shape, (kept_reads(*block) for block in blocks))
    write_data_tarfile(output_tarfile, int_rows, float_rows, reads_rows)
    return num_data, num_reads


def data_blocks(memory_mapped_datas):
    """each dataset's data, DATA_PER_EDIT at a time: the dataset, and the start and end of the block"""
    for data in memory_mapped_datas:
        for start in range(0, data.num_data, DATA_PER_EDIT):
            yield data, start, min(start + DATA_PER_EDIT, data.num_data)


def read_counts(data: MemoryMappedData, start: int, end: int) -> np.ndarray:
    """the numbers of reads of a dataset's data from start to end"""
    return np.diff(data.read_offsets[start : end + 1])


def reads_layout(data: MemoryMappedData):
    """the dtype and row shape of a dataset's reads, or None if it has none"""
    return None if data.reads_mmap is None else (data.reads_mmap.dtype, data.reads_mmap.shape[1:])


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

    num_data, num_reads = save_edited_data(memory_mapped_datas, edit_type, new_source, output_tarfile)
    print(f"Output dataset with {num_data} data and {num_reads} reads.")


def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == "__main__":
    main()
