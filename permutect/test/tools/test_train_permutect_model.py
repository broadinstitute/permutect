import tempfile

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from permutect import constants
from permutect.architecture.artifact_model import load_model
from permutect.test.test_file_names import *
from permutect.tools import train_artifact_model

MAKE_NEW_ARTIFACT_MODEL = False


def test_train_artifact_model(capsys):
    saved_model = tempfile.NamedTemporaryFile() if not MAKE_NEW_ARTIFACT_MODEL else SMALL_ARTIFACT_MODEL
    training_tensorboard_dir = tempfile.TemporaryDirectory()

    argv = ["--" + constants.READ_LAYERS_NAME, "10", "10", "10"]
    argv += ["--" + constants.SELF_ATTENTION_HIDDEN_DIMENSION_NAME, "20"]
    argv += ["--" + constants.NUM_SELF_ATTENTION_LAYERS_NAME, "2"]
    argv += ["--" + constants.INFO_LAYERS_NAME, "10", "10"]
    argv += ["--" + constants.AGGREGATION_LAYERS_NAME, "20", "20", "20"]
    argv += ["--" + constants.TRAINABLE_PARAMETERS_NAME, "WHOLE_MODEL"]
    argv += ["--" + constants.NUM_ARTIFACT_CLUSTERS_NAME, "4"]
    cnn_layer_strings = [
        "convolution/kernel_size=3/out_channels=64",
        "pool/kernel_size=2",
        "leaky_relu",
        "flatten",
        "linear/out_features=10",
    ]
    argv += ["--" + constants.REF_SEQ_LAYER_STRINGS_NAME] + cnn_layer_strings
    argv += ["--" + constants.DROPOUT_P_NAME, "0.0"]

    # Training data inputs
    argv += ["--" + constants.TRAIN_TAR_NAME, PREPROCESSED_DATA]

    # training hyperparameters
    argv += ["--" + constants.BATCH_SIZE_NAME, "64"]
    argv += ["--" + constants.NUM_WORKERS_NAME, "2"]
    argv += ["--" + constants.NUM_EPOCHS_NAME, "2"]
    argv += ["--" + constants.LEARNING_RATE_NAME, "0.001"]
    argv += ["--" + constants.WEIGHT_DECAY_NAME, "0.01"]

    # path to saved model
    argv += ["--" + constants.OUTPUT_NAME, saved_model if MAKE_NEW_ARTIFACT_MODEL else saved_model.name]
    argv += ["--" + constants.TENSORBOARD_DIR_NAME, training_tensorboard_dir.name]

    train_artifact_model.main_without_parsing(train_artifact_model.parse_arguments(argv))
    # the log says which device it ran on, so that a job with a GPU that runs on the CPU is visible
    assert "Running on " in capsys.readouterr().out

    events = EventAccumulator(training_tensorboard_dir.name)
    events.Reload()

    # the saved model has the requested hyperparameters, and training wrote its losses to TensorBoard
    params = load_model(saved_model)._params
    assert (params.read_layers, params.info_layers, params.aggregation_layers) == ([10, 10, 10], [10, 10], [20, 20, 20])
    assert params.num_artifact_clusters == 4 and params.ref_seq_layer_strings == cnn_layer_strings
    assert events.Tags()["scalars"]


def test_refine_artifact_model():
    # train starting from a pretrained model, whose hyperparameters override the default ones parsed here
    saved_model = tempfile.NamedTemporaryFile()
    training_tensorboard_dir = tempfile.TemporaryDirectory()

    argv = ["--" + constants.TRAIN_TAR_NAME, PREPROCESSED_DATA]
    argv += ["--" + constants.PRETRAINED_ARTIFACT_MODEL_NAME, PERMUTECT_V0_9_2_ARTIFACT_MODEL]
    argv += ["--" + constants.TRAINABLE_PARAMETERS_NAME, "WHOLE_MODEL"]
    argv += ["--" + constants.BATCH_SIZE_NAME, "64"]
    argv += ["--" + constants.NUM_WORKERS_NAME, "0"]
    argv += ["--" + constants.NUM_EPOCHS_NAME, "2"]
    argv += ["--" + constants.LEARNING_RATE_NAME, "0.001"]
    argv += ["--" + constants.WEIGHT_DECAY_NAME, "0.01"]
    argv += ["--" + constants.OUTPUT_NAME, saved_model.name]
    argv += ["--" + constants.TENSORBOARD_DIR_NAME, training_tensorboard_dir.name]

    train_artifact_model.main_without_parsing(train_artifact_model.parse_arguments(argv))

    events = EventAccumulator(training_tensorboard_dir.name)
    events.Reload()

    # the refined model keeps the pretrained model's hyperparameters, not the defaults, and training changed its weights
    pretrained, refined = load_model(PERMUTECT_V0_9_2_ARTIFACT_MODEL), load_model(saved_model)
    assert vars(refined._params) == vars(pretrained._params)
    assert any(not torch.equal(p, q) for p, q in zip(refined.parameters(), pretrained.parameters()))
    assert events.Tags()["scalars"]
