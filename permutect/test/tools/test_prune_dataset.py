import tempfile
from argparse import Namespace

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from permutect import constants
from permutect.data.memory_mapped_data import MemoryMappedData
from permutect.data.reads_dataset import ReadsDataset
from permutect.test.test_file_names import *
from permutect.tools import prune_dataset


def test_prune_dataset(trained_artifact_model):
    pruned_dataset = tempfile.NamedTemporaryFile()
    training_tensorboard_dir = tempfile.TemporaryDirectory()

    prune_dataset_args = Namespace()
    setattr(prune_dataset_args, constants.TRAIN_TAR_NAME, PREPROCESSED_DATA)
    setattr(prune_dataset_args, constants.ARTIFACT_MODEL_NAME, trained_artifact_model)
    setattr(prune_dataset_args, constants.OUTPUT_NAME, pruned_dataset.name)
    setattr(prune_dataset_args, constants.TENSORBOARD_DIR_NAME, training_tensorboard_dir.name)
    setattr(prune_dataset_args, constants.BATCH_SIZE_NAME, 64)
    setattr(prune_dataset_args, constants.INFERENCE_BATCH_SIZE_NAME, 64)
    setattr(prune_dataset_args, constants.NUM_WORKERS_NAME, 0)
    setattr(prune_dataset_args, constants.NUM_EPOCHS_NAME, 2)
    setattr(prune_dataset_args, constants.NUM_CALIBRATION_EPOCHS_NAME, 1)
    setattr(prune_dataset_args, constants.LEARNING_RATE_NAME, 0.001)
    setattr(prune_dataset_args, constants.WEIGHT_DECAY_NAME, 0.01)

    prune_dataset.main_without_parsing(prune_dataset_args)

    events = EventAccumulator(training_tensorboard_dir.name)
    events.Reload()

    memory_mapped_data = MemoryMappedData.load_from_tarfile(pruned_dataset.name)
    ReadsDataset(memory_mapped_data=memory_mapped_data, num_folds=10)
