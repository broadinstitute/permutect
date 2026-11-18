import io
import tarfile

import numpy as np
import pytest
import torch

from permutect import constants
from permutect.read_orientation.baseline_orientation import fit_baseline_orientation
from permutect.read_orientation.f1r2_counts_reader import read_f1r2_counts
from permutect.read_orientation.read_orientation_model import F1R2_FRACTION_PSEUDOCOUNTS_K2
from permutect.read_orientation.read_orientation_model import NUM_ARTIFACT_STATES
from permutect.read_orientation.read_orientation_model import ReadOrientationModel
from permutect.read_orientation.read_orientation_model import learn_priors
from permutect.read_orientation.read_orientation_model import non_artifact_f1r2_pseudocounts_k2
from permutect.read_orientation.read_orientation_model import reverse_complement
from permutect.segmentation.germline_model import GermlineModel
from permutect.segmentation.germline_model import SampleSegmentation
from permutect.segmentation.segmenter import Segment
from permutect.test.test_file_names import DREAM_TUMOR_4_F1R2_SCATTERS
from permutect.test.test_file_names import DREAM_TUMOR_4_GATK_ORIENTATION_PRIORS
from permutect.test.test_file_names import HAPMAP_F1R2_SCATTERS
from permutect.test.test_file_names import HAPMAP_GATK_ORIENTATION_PRIORS
from permutect.tools import learn_read_orientation_model


def read_gatk_priors(tar_gz) -> tuple[str, dict[str, list[str]]]:
    """the sample of the one prior table in a tar.gz from GATK LearnReadOrientationModel, and its rows' fields by context"""
    with tarfile.open(tar_gz, "r:gz") as tar:
        (member,) = [member for member in tar.getmembers() if member.isfile()]
        lines = io.TextIOWrapper(tar.extractfile(member)).read().splitlines()
    # the sample metadata, the header and a row for each context
    sample = lines[0].removeprefix("#<METADATA>SAMPLE=")
    return sample, {fields[0]: fields for fields in (line.split("\t") for line in lines[2:])}


@pytest.mark.parametrize(
    "f1r2_counts, gatk_priors",
    [
        (DREAM_TUMOR_4_F1R2_SCATTERS, DREAM_TUMOR_4_GATK_ORIENTATION_PRIORS),
        (HAPMAP_F1R2_SCATTERS, HAPMAP_GATK_ORIENTATION_PRIORS),
    ],
)
def test_priors_match_gatk(f1r2_counts, gatk_priors):
    # the priors of GATK LearnReadOrientationModel from the same counts and with its likelihoods, including the flat
    # priors of contexts with too few sites of the hapmap counts
    ((sample, counts),) = read_f1r2_counts(f1r2_counts).items()
    priors = learn_priors(counts)
    gatk_sample, gatk_rows = read_gatk_priors(gatk_priors)
    assert sample == gatk_sample
    assert priors.keys() == gatk_rows.keys() and len(priors) == 64
    for context, fields in gatk_rows.items():
        # context, reverse complement, 12 probabilities, number of sites and of alt sites
        prior = priors[context]
        assert fields[1] == reverse_complement(context)
        assert [prior.num_sites, prior.num_alt_sites] == [int(fields[14]), int(fields[15])], context
        gatk_probabilities = np.array(fields[2:14], dtype=float)
        assert np.allclose(prior.prior_k.numpy(), gatk_probabilities, rtol=1e-9, atol=1e-12), context


def test_order_of_inputs_does_not_matter(tmp_path):
    # eg the scatters of a Mutect2 run in the workflow's order or as a cache lists them
    in_order, reversed_order = str(tmp_path / "in_order.pt"), str(tmp_path / "reversed.pt")
    learn_read_orientation_model.learn_read_orientation_model(HAPMAP_F1R2_SCATTERS, in_order)
    learn_read_orientation_model.learn_read_orientation_model(HAPMAP_F1R2_SCATTERS[::-1], reversed_order)
    model, reversed_model = ReadOrientationModel.load(in_order), ReadOrientationModel.load(reversed_order)
    for name, buffer in model.named_buffers():
        assert torch.equal(buffer, reversed_model.get_buffer(name)), name


def test_counts_of_more_than_one_sample_are_an_error(tmp_path):
    # the DREAM tumor's counts and the hapmap sample's
    with pytest.raises(ValueError, match="2 samples"):
        learn_read_orientation_model.learn_read_orientation_model(
            [*DREAM_TUMOR_4_F1R2_SCATTERS, *HAPMAP_F1R2_SCATTERS], str(tmp_path / "model.pt")
        )


def test_arguments():
    args = learn_read_orientation_model.parse_arguments(["--input", "a.tar.gz", "b.tar.gz", "--output", "model.pt"])
    assert getattr(args, constants.INPUT_NAME) == ["a.tar.gz", "b.tar.gz"]
    assert getattr(args, constants.OUTPUT_NAME) == "model.pt"
    # GATK's defaults
    assert getattr(args, constants.CONVERGENCE_THRESHOLD_NAME) == 1e-4
    assert getattr(args, constants.MAX_EM_ITERATIONS_NAME) == 20
    assert getattr(args, constants.GERMLINE_MODEL_NAME) is None


def test_germline_model_is_passed_along(tmp_path, monkeypatch):
    # the GermlineModel that segment_pileups saved, recognizable by its contamination
    germline_model = str(tmp_path / "germline-model.pt")
    GermlineModel(contamination=0.123).save(germline_model)
    received = {}
    monkeypatch.setattr(
        learn_read_orientation_model, "learn_read_orientation_model", lambda **kwargs: received.update(kwargs)
    )
    arguments = ["--input", *HAPMAP_F1R2_SCATTERS, "--output", str(tmp_path / "model.pt")]

    learn_read_orientation_model.main_without_parsing(learn_read_orientation_model.parse_arguments(arguments))
    assert received["germline_model"] is None

    arguments += ["--" + constants.GERMLINE_MODEL_NAME, germline_model]
    learn_read_orientation_model.main_without_parsing(learn_read_orientation_model.parse_arguments(arguments))
    assert received["germline_model"].contamination == pytest.approx(0.123)


def test_germline_model(tmp_path):
    # segment_pileups' model of the DREAM tumor, with chromosome 20 split into loss of heterozygosity and a balanced region
    sample, _ = read_gatk_priors(DREAM_TUMOR_4_GATK_ORIENTATION_PRIORS)
    segments = [Segment("20", 1, 30_000_000, 0.0, 0.0), Segment("20", 30_000_001, 63_025_520, 0.5, 0.5)]
    germline_model = str(tmp_path / "germline-model.pt")
    GermlineModel(SampleSegmentation(sample, segments)).save(germline_model)
    arguments = ["--input", *DREAM_TUMOR_4_F1R2_SCATTERS]
    with_germline_model, without = str(tmp_path / "with.pt"), str(tmp_path / "without.pt")
    learn_read_orientation_model.main_without_parsing(
        learn_read_orientation_model.parse_arguments(
            [*arguments, "--output", with_germline_model, "--" + constants.GERMLINE_MODEL_NAME, germline_model]
        )
    )
    learn_read_orientation_model.main_without_parsing(
        learn_read_orientation_model.parse_arguments([*arguments, "--output", without])
    )

    # the germline hets' likelihoods change the priors, but not where there's too little data to learn from
    priors, priors_without = (
        ReadOrientationModel.load(path).context_priors() for path in (with_germline_model, without)
    )
    changed = [
        context for context in priors if not torch.equal(priors[context].prior_k, priors_without[context].prior_k)
    ]
    assert changed and all(priors_without[context].num_sites > 0 for context in changed)


def test_germline_model_of_another_sample_is_an_error(tmp_path):
    germline_model = str(tmp_path / "germline-model.pt")
    GermlineModel(SampleSegmentation("ANOTHER_SAMPLE")).save(germline_model)
    arguments = ["--input", *DREAM_TUMOR_4_F1R2_SCATTERS, "--output", str(tmp_path / "model.pt")]
    arguments += ["--" + constants.GERMLINE_MODEL_NAME, germline_model]
    with pytest.raises(ValueError, match="germline model is of sample ANOTHER_SAMPLE"):
        learn_read_orientation_model.main_without_parsing(learn_read_orientation_model.parse_arguments(arguments))


def test_baseline_orientation_is_fit_and_saved(tmp_path):
    output = str(tmp_path / "model.pt")
    learn_read_orientation_model.main_without_parsing(
        learn_read_orientation_model.parse_arguments(["--input", *DREAM_TUMOR_4_F1R2_SCATTERS, "--output", output])
    )
    # the non-artifact states' F1R2 pseudocounts are the fit to the ref reads of the alt tables, rather than GATK's
    ((_, counts),) = read_f1r2_counts(DREAM_TUMOR_4_F1R2_SCATTERS).items()
    expected_k2 = non_artifact_f1r2_pseudocounts_k2(*fit_baseline_orientation(counts.ref_orientations).pseudocounts())
    saved_k2 = ReadOrientationModel.load(output).f1r2_fraction_pseudocounts_k2
    assert torch.allclose(saved_k2, expected_k2)
    assert not torch.allclose(saved_k2[NUM_ARTIFACT_STATES:], F1R2_FRACTION_PSEUDOCOUNTS_K2[NUM_ARTIFACT_STATES:])

    # and the alt table sites' own ref reads update them in learning the priors
    saved_priors = ReadOrientationModel.load(output).context_priors()
    for condition_on_ref_reads in (True, False):
        priors = learn_priors(
            counts, f1r2_fraction_pseudocounts_k2=expected_k2, condition_on_ref_reads=condition_on_ref_reads
        )
        same = all(torch.allclose(priors[c].prior_k, saved_priors[c].prior_k) for c in priors)
        assert same == condition_on_ref_reads
