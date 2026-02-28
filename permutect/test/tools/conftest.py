import tempfile
from argparse import Namespace

import pytest

from permutect import constants
from permutect.test.test_file_names import *
from permutect.tools import preprocess_dataset
from permutect.tools import train_artifact_model


@pytest.fixture(scope="session")
def trained_artifact_model():
    """Train a small artifact model from scratch for use in integration tests.

    The pre-shipped artifact-model-v0.4.0.pt is incompatible with the current
    architecture, so we train a fresh (tiny) model once per test session.
    """
    # Step 1: preprocess
    training_data_tarfile = tempfile.NamedTemporaryFile()

    preprocess_args = Namespace()
    setattr(preprocess_args, constants.TRAINING_DATASETS_NAME, [SMALL_PLAIN_TEXT_DATA])
    setattr(preprocess_args, constants.OUTPUT_NAME, training_data_tarfile.name)
    setattr(preprocess_args, constants.SOURCES_NAME, [0])
    preprocess_dataset.main_without_parsing(preprocess_args)

    # Step 2: train
    saved_artifact_model = tempfile.NamedTemporaryFile()
    training_tensorboard_dir = tempfile.TemporaryDirectory()

    train_model_args = Namespace()
    setattr(train_model_args, constants.READ_LAYERS_NAME, [10, 10, 10])
    setattr(train_model_args, constants.SELF_ATTENTION_HIDDEN_DIMENSION_NAME, 20)
    setattr(train_model_args, constants.NUM_SELF_ATTENTION_LAYERS_NAME, 3)
    setattr(train_model_args, constants.INFO_LAYERS_NAME, [30, 30, 30])
    setattr(train_model_args, constants.AGGREGATION_LAYERS_NAME, [30, 30, 30, 30])
    setattr(train_model_args, constants.NUM_ARTIFACT_CLUSTERS_NAME, 4)
    setattr(train_model_args, constants.CALIBRATION_LAYERS_NAME, [6, 6])
    cnn_layer_strings = [
        "convolution/kernel_size=3/out_channels=64",
        "pool/kernel_size=2",
        "leaky_relu",
        "convolution/kernel_size=3/dilation=2/out_channels=5",
        "leaky_relu",
        "flatten",
        "linear/out_features=10",
    ]
    setattr(train_model_args, constants.REF_SEQ_LAYER_STRINGS_NAME, cnn_layer_strings)
    setattr(train_model_args, constants.DROPOUT_P_NAME, 0.0)
    setattr(train_model_args, constants.LEARNING_RATE_NAME, 0.001)
    setattr(train_model_args, constants.WEIGHT_DECAY_NAME, 0.01)
    setattr(train_model_args, constants.BATCH_NORMALIZE_NAME, False)
    setattr(train_model_args, constants.REWEIGHTING_RANGE_NAME, 0.3)
    setattr(train_model_args, constants.TRAIN_TAR_NAME, training_data_tarfile.name)
    setattr(train_model_args, constants.PRETRAINED_ARTIFACT_MODEL_NAME, None)
    setattr(train_model_args, constants.BATCH_SIZE_NAME, 64)
    setattr(train_model_args, constants.INFERENCE_BATCH_SIZE_NAME, 64)
    setattr(train_model_args, constants.NUM_WORKERS_NAME, 0)
    setattr(train_model_args, constants.NUM_EPOCHS_NAME, 2)
    setattr(train_model_args, constants.NUM_CALIBRATION_EPOCHS_NAME, 1)
    setattr(train_model_args, constants.GENOMIC_SPAN_NAME, 100000)
    setattr(train_model_args, constants.OUTPUT_NAME, saved_artifact_model.name)
    setattr(train_model_args, constants.TENSORBOARD_DIR_NAME, training_tensorboard_dir.name)

    train_artifact_model.main_without_parsing(train_model_args)

    yield saved_artifact_model.name

    # cleanup happens automatically via NamedTemporaryFile
