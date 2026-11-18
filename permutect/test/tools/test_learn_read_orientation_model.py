import pytest
import torch

from permutect import constants
from permutect.read_orientation.capture_bias import fit_capture_bias
from permutect.read_orientation.f1r2_counts_reader import read_f1r2_counts
from permutect.read_orientation.read_orientation_model import POSSIBLE_PROCESSES_P
from permutect.read_orientation.read_orientation_model import ReadOrientationModel
from permutect.read_orientation.read_orientation_model import process_name
from permutect.segmentation.germline_model import GermlineModel
from permutect.segmentation.germline_model import SampleSegmentation
from permutect.segmentation.segmenter import Segment
from permutect.test.test_file_names import DREAM_TUMOR_4_F1R2_SCATTERS
from permutect.test.test_file_names import HAPMAP_F1R2_SCATTERS
from permutect.tools import learn_read_orientation_model


def learn(inputs, output, *arguments) -> ReadOrientationModel:
    args = ["--" + constants.INPUT_NAME, *inputs, "--" + constants.OUTPUT_NAME, str(output), *arguments]
    learn_read_orientation_model.main_without_parsing(learn_read_orientation_model.parse_arguments(args))
    return ReadOrientationModel.load(output)


def test_dream_tumor_damage_is_oxidation_of_guanine(tmp_path):
    # the DREAM challenge's synthetic tumors have the artifacts of oxidized guanine, G -> T: the processes of the
    # highest mean rates of damage
    model = learn(DREAM_TUMOR_4_F1R2_SCATTERS, tmp_path / "model.pt")
    alpha_p, beta_p = model.damage_pseudocounts_ca2.flatten(0, 1).unbind(-1)
    means_p = torch.where(POSSIBLE_PROCESSES_P, alpha_p / (alpha_p + beta_p), 0)
    top = [process_name(p) for p in means_p.argsort(descending=True)[:5].tolist()]
    assert all(name[1] == "G" and name[-2] == "T" for name in top), top


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
    ((sample, _),) = read_f1r2_counts(DREAM_TUMOR_4_F1R2_SCATTERS).items()
    segments = [Segment("20", 1, 30_000_000, 0.0, 0.0), Segment("20", 30_000_001, 63_025_520, 0.5, 0.5)]
    germline_model = str(tmp_path / "germline-model.pt")
    GermlineModel(SampleSegmentation(sample, segments)).save(germline_model)
    with_germline_model = learn(
        DREAM_TUMOR_4_F1R2_SCATTERS, tmp_path / "with.pt", "--" + constants.GERMLINE_MODEL_NAME, germline_model
    )
    without = learn(DREAM_TUMOR_4_F1R2_SCATTERS, tmp_path / "without.pt")

    # the germline hets' likelihoods change the genotype priors, and every context of this sample has data
    assert not torch.allclose(with_germline_model.genotype_priors_cs, without.genotype_priors_cs)
    assert torch.all(without.num_sites_c > 0)


def test_germline_model_of_another_sample_is_an_error(tmp_path):
    germline_model = str(tmp_path / "germline-model.pt")
    GermlineModel(SampleSegmentation("ANOTHER_SAMPLE")).save(germline_model)
    arguments = ["--input", *DREAM_TUMOR_4_F1R2_SCATTERS, "--output", str(tmp_path / "model.pt")]
    arguments += ["--" + constants.GERMLINE_MODEL_NAME, germline_model]
    with pytest.raises(ValueError, match="germline model is of sample ANOTHER_SAMPLE"):
        learn_read_orientation_model.main_without_parsing(learn_read_orientation_model.parse_arguments(arguments))


def test_capture_bias_is_fit_and_saved(tmp_path):
    # to the reads of the alt tables' sites
    model = learn(DREAM_TUMOR_4_F1R2_SCATTERS, tmp_path / "model.pt")
    ((_, counts),) = read_f1r2_counts(DREAM_TUMOR_4_F1R2_SCATTERS).items()
    expected = fit_capture_bias(counts.capture_counts).pseudocounts()
    assert model.capture_pseudocounts_2.tolist() == pytest.approx(expected)
