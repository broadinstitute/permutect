import io
import tarfile

import numpy as np
import pytest

from permutect import constants
from permutect.test.test_file_names import DREAM_TUMOR_4_F1R2_SCATTERS
from permutect.test.test_file_names import DREAM_TUMOR_4_GATK_ORIENTATION_PRIORS
from permutect.test.test_file_names import HAPMAP_F1R2_SCATTERS
from permutect.test.test_file_names import HAPMAP_GATK_ORIENTATION_PRIORS
from permutect.tools import learn_read_orientation_model


def read_prior_tables(tar_gz) -> dict:
    """the lines of each prior table in a tar.gz, by file name, with the table's rows sorted"""
    result = {}
    with tarfile.open(tar_gz, "r:gz") as tar:
        for member in tar.getmembers():
            lines = io.TextIOWrapper(tar.extractfile(member)).read().splitlines()
            result[member.name.removeprefix("./")] = lines[:2] + sorted(lines[2:])
    return result


@pytest.mark.parametrize(
    "f1r2_counts, gatk_priors",
    [
        (DREAM_TUMOR_4_F1R2_SCATTERS, DREAM_TUMOR_4_GATK_ORIENTATION_PRIORS),
        (HAPMAP_F1R2_SCATTERS, HAPMAP_GATK_ORIENTATION_PRIORS),
    ],
)
def test_priors_match_gatk(tmp_path, f1r2_counts, gatk_priors):
    # the priors of GATK LearnReadOrientationModel from the same counts, in tables of the same format, including the
    # flat priors of contexts with too few sites of the hapmap counts
    output = str(tmp_path / "priors.tar.gz")
    learn_read_orientation_model.main_without_parsing(
        learn_read_orientation_model.parse_arguments(["--input", *f1r2_counts, "--output", output])
    )
    gatk_tables, tables = read_prior_tables(gatk_priors), read_prior_tables(output)
    assert tables.keys() == gatk_tables.keys()
    for name, gatk_lines in gatk_tables.items():
        lines = tables[name]
        # the sample metadata, the header and a row for each of the 64 contexts
        assert lines[:2] == gatk_lines[:2]
        assert len(lines) == len(gatk_lines) == 66
        for row, gatk_row in zip(lines[2:], gatk_lines[2:]):
            fields, gatk_fields = row.split("\t"), gatk_row.split("\t")
            # context, reverse complement, 12 probabilities, number of sites and of alt sites
            assert fields[:2] == gatk_fields[:2] and fields[14:] == gatk_fields[14:]
            probabilities, gatk_probabilities = (
                np.array(fields[2:14], dtype=float),
                np.array(gatk_fields[2:14], dtype=float),
            )
            assert np.allclose(probabilities, gatk_probabilities, rtol=1e-9, atol=1e-12), fields[0]


def test_order_of_inputs_does_not_matter(tmp_path):
    # eg the scatters of a Mutect2 run in the workflow's order or as a cache lists them
    in_order, reversed_order = str(tmp_path / "in_order.tar.gz"), str(tmp_path / "reversed.tar.gz")
    learn_read_orientation_model.learn_read_orientation_model(HAPMAP_F1R2_SCATTERS, in_order)
    learn_read_orientation_model.learn_read_orientation_model(HAPMAP_F1R2_SCATTERS[::-1], reversed_order)
    assert read_prior_tables(in_order) == read_prior_tables(reversed_order)


def test_arguments():
    args = learn_read_orientation_model.parse_arguments(
        ["--input", "a.tar.gz", "b.tar.gz", "--output", "priors.tar.gz"]
    )
    assert getattr(args, constants.INPUT_NAME) == ["a.tar.gz", "b.tar.gz"]
    assert getattr(args, constants.OUTPUT_NAME) == "priors.tar.gz"
    # GATK's defaults
    assert getattr(args, constants.CONVERGENCE_THRESHOLD_NAME) == 1e-4
    assert getattr(args, constants.MAX_EM_ITERATIONS_NAME) == 20


def test_output_must_be_a_tar_gz(tmp_path):
    with pytest.raises(ValueError, match="tar.gz"):
        learn_read_orientation_model.learn_read_orientation_model(
            DREAM_TUMOR_4_F1R2_SCATTERS, str(tmp_path / "priors.tsv")
        )
