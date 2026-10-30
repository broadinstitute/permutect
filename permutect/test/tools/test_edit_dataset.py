import pytest

from permutect import constants
from permutect.tools import edit_dataset
from permutect.tools.edit_dataset import EditType


def test_edit_type_must_be_one_of_the_choices(capsys):
    argv = ["--" + constants.TRAIN_TAR_NAME, "data.tar", "--" + constants.OUTPUT_NAME, "out.tar"]
    for edit_type in EditType:
        args = edit_dataset.parse_arguments(argv + ["--" + constants.DATASET_EDIT_TYPE_NAME, edit_type.value])
        assert getattr(args, constants.DATASET_EDIT_TYPE_NAME) == edit_type.value

    # a typo is an error when parsing arguments, not after loading every dataset
    with pytest.raises(SystemExit):
        edit_dataset.parse_arguments(argv + ["--" + constants.DATASET_EDIT_TYPE_NAME, "unlabel_artifact"])
    assert "invalid choice" in capsys.readouterr().err
