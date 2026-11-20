import pytest
import torch

from permutect import constants
from permutect.segmentation.germline_model import GermlineModel
from permutect.segmentation.germline_model import SampleSegmentation
from permutect.segmentation.segmenter import Segment
from permutect.slippage.slip_rates import DEFAULT_ALLELE_FREQUENCY
from permutect.slippage.slip_rates import MAX_ALLELE_LENGTH
from permutect.slippage.slip_rates import MAX_UNIT_LENGTH
from permutect.slippage.slip_rates import SLIPS
from permutect.slippage.slippage_counts import read_slippage_counts
from permutect.slippage.slippage_model import MIN_SLIPPAGE_STR_LENGTH
from permutect.slippage.slippage_model import SLIPPAGE_ALLELE_FRACTION_PSEUDOCOUNTS
from permutect.slippage.slippage_model import SlippageModel
from permutect.test.slippage.test_slippage_counts import KEPT_ROWS
from permutect.test.slippage.test_slippage_counts import TABLE
from permutect.tools import learn_slippage_model


def learn(tmp_path, germline_model: GermlineModel) -> SlippageModel:
    """the slippage model learned from TABLE, whose sample is NA12878, and germline_model"""
    table, germline_model_path, output = tmp_path / "counts.table", tmp_path / "germline.pt", tmp_path / "slippage.pt"
    table.write_text(TABLE)
    germline_model.save(germline_model_path)
    args = ["--" + constants.INPUT_NAME, str(table), "--" + constants.OUTPUT_NAME, str(output)]
    args += ["--" + constants.GERMLINE_MODEL_NAME, str(germline_model_path)]
    learn_slippage_model.main_without_parsing(learn_slippage_model.parse_arguments(args))
    return SlippageModel.load(output)


def test_learns_slip_rates(tmp_path):
    model = learn(tmp_path, GermlineModel(SampleSegmentation("NA12878")))
    assert model.slip_log_odds_jul.shape == (len(SLIPS), MAX_UNIT_LENGTH, MAX_ALLELE_LENGTH)
    assert torch.all(torch.isfinite(model.slip_log_odds_jul))
    # the likelihoods that filter_variants uses don't change yet
    assert model.allele_fraction_pseudocounts_2.tolist() == list(SLIPPAGE_ALLELE_FRACTION_PSEUDOCOUNTS)
    assert model.min_str_length == MIN_SLIPPAGE_STR_LENGTH


def test_counts_may_be_of_the_matched_normal(tmp_path):
    learn(tmp_path, GermlineModel(SampleSegmentation("TUMOR"), SampleSegmentation("NA12878")))


def test_germline_model_of_another_sample_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="counts are of sample NA12878, but the germline model is of samples TUMOR"):
        learn(tmp_path, GermlineModel(SampleSegmentation("TUMOR")))


def test_strs_get_the_maf_of_their_segment(tmp_path):
    # STRs that start at 20:10002459, 20:10037126, 20:10006820 and 20:10214834
    path = tmp_path / "counts.table"
    path.write_text(TABLE + KEPT_ROWS)
    segments = [Segment("20", 10000000, 10010000, 0.3, 0.3), Segment("20", 10200000, 10300000, 0.1, 0.1)]
    mafs_s = learn_slippage_model.str_mafs(SampleSegmentation("NA12878", segments), read_slippage_counts(path))
    assert mafs_s.tolist() == [0.3, 0.5, 0.3, 0.1]


def test_arguments():
    args = ["--input", "counts.table", "--germline_model", "germline.pt", "--output", "slippage.pt"]
    parsed = learn_slippage_model.parse_arguments(args)
    assert getattr(parsed, constants.INPUT_NAME) == "counts.table"
    assert getattr(parsed, constants.GERMLINE_MODEL_NAME) == "germline.pt"
    assert getattr(parsed, constants.OUTPUT_NAME) == "slippage.pt"
    assert getattr(parsed, constants.DEFAULT_ALLELE_FREQUENCY_NAME) == DEFAULT_ALLELE_FREQUENCY
    parsed = learn_slippage_model.parse_arguments(args + ["--default_allele_frequency", "1e-6"])
    assert getattr(parsed, constants.DEFAULT_ALLELE_FREQUENCY_NAME) == 1e-6
    # the first three are required
    for n in range(0, len(args), 2):
        with pytest.raises(SystemExit):
            learn_slippage_model.parse_arguments(args[:n] + args[n + 2 :])
