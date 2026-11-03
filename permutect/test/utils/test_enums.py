import argparse

import pytest
import torch

from permutect import constants
from permutect.architecture.artifact_model import load_model
from permutect.parameters import add_training_params_to_parser
from permutect.parameters import parse_training_params
from permutect.test.test_file_names import PERMUTECT_V0_9_2_ARTIFACT_MODEL
from permutect.utils.enums import Epoch
from permutect.utils.enums import ParameterSet
from permutect.utils.enums import Variation

PARAMETER_SET_NAMES = [
    "INITIAL_READ_EMBEDDING",
    "INFO",
    "HAPLOTYPES",
    "GATED_MLP",
    "REDUCER",
    "PRECLUSTERING",
    "NONARTIFACT_COVARIANCE",
    "ARTIFACT_DIRECTIONS",
    "ARTIFACT_EMG",
    "ARTIFACT_STDEV",
    "CLUSTERING",
    "WHOLE_MODEL",
]


def test_variant_types_keep_their_values():
    # datasets store variant types as integers, so new types go at the end
    assert [(v.name, v.value) for v in Variation] == [
        ("SNV", 0),
        ("INSERTION", 1),
        ("DELETION", 2),
        ("BIG_INSERTION", 3),
        ("BIG_DELETION", 4),
        ("MNV", 5),
    ]


def test_variation_get_type():
    assert Variation.get_type("A", "T") == Variation.SNV
    assert Variation.get_type("A", "AT") == Variation.INSERTION
    assert Variation.get_type("AT", "A") == Variation.DELETION
    assert Variation.get_type("A", "ATT") == Variation.BIG_INSERTION
    assert Variation.get_type("ATT", "A") == Variation.BIG_DELETION

    # substitutions of more than one base are MNVs, even with matching bases inside them
    assert Variation.get_type("AT", "GC") == Variation.MNV
    assert Variation.get_type("ACG", "TTA") == Variation.MNV
    assert Variation.get_type("ACGT", "TCGA") == Variation.MNV


def test_variation_get_type_trims_matching_bases_of_substitutions():
    # padded SNVs, as at some multiallelic sites, are SNVs, and padded MNVs are MNVs
    assert Variation.get_type("AT", "GT") == Variation.SNV
    assert Variation.get_type("GA", "GT") == Variation.SNV
    assert Variation.get_type("CAG", "CTG") == Variation.SNV
    assert Variation.get_type("CAGT", "CGTT") == Variation.MNV


@pytest.fixture(scope="module")
def model():
    return load_model(PERMUTECT_V0_9_2_ARTIFACT_MODEL, device=torch.device("cpu"))


def test_parameter_set_members():
    # from Python 3.13, functools.partial values silently stop being enum members
    assert [parameter_set.name for parameter_set in ParameterSet] == PARAMETER_SET_NAMES
    assert all(isinstance(getattr(ParameterSet, name), ParameterSet) for name in PARAMETER_SET_NAMES)


def test_trainable_parameter_sets_are_parsed_by_name(capsys):
    parser = argparse.ArgumentParser()
    add_training_params_to_parser(parser)
    args = parser.parse_args(["--" + constants.TRAINABLE_PARAMETERS_NAME] + PARAMETER_SET_NAMES)
    training_params = parse_training_params(args, default_training_params=[ParameterSet.WHOLE_MODEL])
    assert training_params.trainable_parameter_sets == list(ParameterSet)

    # an invalid name is rejected with the other command line errors, which list the valid names
    with pytest.raises(SystemExit):
        parser.parse_args(["--" + constants.TRAINABLE_PARAMETERS_NAME, "CLUSTERING", "NOT_A_PARAMETER_SET"])
    assert "invalid choice: 'NOT_A_PARAMETER_SET'" in capsys.readouterr().err


def test_get_parameters(model):
    all_parameters = {id(p) for p in model.parameters()}
    for parameter_set in ParameterSet:
        parameters = list(parameter_set.get_parameters(model))
        assert len(parameters) > 0
        assert all(id(p) in all_parameters for p in parameters)

    assert {id(p) for p in ParameterSet.WHOLE_MODEL.get_parameters(model)} == all_parameters


def test_set_trainable_parameters_trains_only_the_chosen_parameter_sets(model):
    model.set_trainable_parameters([ParameterSet.INFO, ParameterSet.REDUCER])
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    chosen = list(ParameterSet.INFO.get_parameters(model)) + list(ParameterSet.REDUCER.get_parameters(model))
    assert trainable == {id(p) for p in chosen}


def test_set_epoch_type_restores_training_mode_after_validation(model):
    model.set_epoch_type(Epoch.TRAIN)
    assert model.training
    model.set_epoch_type(Epoch.VALID)
    assert not model.training
    model.set_epoch_type(Epoch.TRAIN)
    assert model.training
    assert all(module.training for module in model.modules())


def test_set_epoch_type_leaves_the_trainable_parameters_alone(model):
    # DistributedDataParallel needs the parameters that require gradients to stay the same after it wraps the model
    model.set_trainable_parameters([ParameterSet.INFO, ParameterSet.REDUCER])
    trainable = [p.requires_grad for p in model.parameters()]
    for epoch_type in (Epoch.TRAIN, Epoch.VALID, Epoch.TRAIN):
        model.set_epoch_type(epoch_type)
        assert [p.requires_grad for p in model.parameters()] == trainable
