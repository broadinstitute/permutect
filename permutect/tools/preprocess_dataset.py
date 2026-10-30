import argparse

from permutect import constants
from permutect.data.plain_text_data import make_normalized_mmap_data

"""
This tool takes as input a list of plain text Permutect training datasets made by Mutect2, normalizes their data, and
saves it as a tarfile of memory-mapped binary arrays.
"""


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="preprocess plain text training datasets into a tarfile of normalized binary data"
    )
    parser.add_argument(
        "--" + constants.TRAINING_DATASETS_NAME,
        nargs="+",
        required=True,
        help="list of plain text data files",
    )
    parser.add_argument(
        "--" + constants.SOURCES_NAME,
        nargs="+",
        type=int,
        help="integer sources corresponding to plain text data files for distinguishing different sequencing conditions",
    )
    parser.add_argument(
        "--" + constants.OUTPUT_NAME,
        required=True,
        help="path to output tarfile of training data",
    )
    args = parser.parse_args(argv)
    datasets, sources = getattr(args, constants.TRAINING_DATASETS_NAME), getattr(args, constants.SOURCES_NAME)
    # one source for all datasets, or one for each.  Otherwise reading the data would fail partway through.
    if sources is not None and len(sources) not in (1, len(datasets)):
        parser.error(
            f"--{constants.SOURCES_NAME} must be one source or one for each of the {len(datasets)} "
            f"--{constants.TRAINING_DATASETS_NAME}, not {len(sources)}"
        )
    return args


def main_without_parsing(args):
    datasets, sources = getattr(args, constants.TRAINING_DATASETS_NAME), getattr(args, constants.SOURCES_NAME)
    make_normalized_mmap_data(datasets, sources=sources).save_to_tarfile(getattr(args, constants.OUTPUT_NAME))


def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == "__main__":
    main()
