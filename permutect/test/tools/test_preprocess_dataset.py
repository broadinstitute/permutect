import tempfile
from argparse import Namespace

from permutect import constants
from permutect.data.memory_mapped_data import MemoryMappedData
from permutect.data.reads_dataset import ReadsDataset
from permutect.test.test_file_names import *
from permutect.tools import preprocess_dataset


def test_on_10_megabases_singular():
    training_data_tarfile = tempfile.NamedTemporaryFile()

    preprocess_args = Namespace()
    setattr(preprocess_args, constants.TRAINING_DATASETS_NAME, [SMALL_PLAIN_TEXT_DATA])
    setattr(preprocess_args, constants.OUTPUT_NAME, training_data_tarfile.name)
    setattr(preprocess_args, constants.SOURCES_NAME, [0])
    preprocess_dataset.main_without_parsing(preprocess_args)

    memory_mapped_data = MemoryMappedData.load_from_tarfile(training_data_tarfile.name)
    ReadsDataset(memory_mapped_data=memory_mapped_data, num_folds=10)
