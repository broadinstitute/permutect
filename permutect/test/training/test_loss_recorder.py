import math

import numpy as np
import pytest
import torch

from permutect.data.batch import Batch
from permutect.data.datum import RAW_READS_ARRAY_DTYPE
from permutect.data.datum import Data
from permutect.data.datum import Datum
from permutect.training.loss_recorder import LossRecorder
from permutect.utils.enums import Label


def make_datum(label: Label) -> Datum:
    datum = Datum(
        int_array=np.zeros(Data.NUM_SCALAR_INT_ELEMENTS),
        float_array=np.zeros(Data.NUM_SCALAR_FLOAT_ELEMENTS),
        reads_re=np.zeros((3, 2), dtype=RAW_READS_ARRAY_DTYPE),
    )
    datum.set(Data.LABEL, label)
    datum.set(Data.REF_COUNT, 2)
    datum.set(Data.ALT_COUNT, 1)
    return datum


def make_recorder(labels, losses) -> LossRecorder:
    recorder = LossRecorder(torch.device("cpu"), num_sources=1)
    if labels:
        batch = Batch([make_datum(label) for label in labels])
        recorder.primary_metrics.record(batch, torch.tensor(losses), torch.ones(len(losses)))
    return recorder


def test_mean_loss_is_the_mean_of_the_label_averages():
    recorder = make_recorder([Label.ARTIFACT, Label.ARTIFACT, Label.VARIANT, Label.UNLABELED], [1.0, 3.0, 4.0, 0.5])
    # the label averages are 2 (artifact), 4 (variant) and 0.5 (unlabeled)
    assert recorder.mean_loss() == pytest.approx((2.0 + 4.0 + 0.5) / 3)


def test_mean_loss_skips_labels_without_data():
    # all data unlabeled, as in test-time adaptation
    assert make_recorder([Label.UNLABELED] * 3, [1.0, 2.0, 3.0]).mean_loss() == pytest.approx(2.0)

    # no unlabeled data
    assert make_recorder([Label.ARTIFACT, Label.VARIANT], [1.0, 3.0]).mean_loss() == pytest.approx(2.0)

    # no artifacts
    assert make_recorder([Label.VARIANT, Label.UNLABELED], [1.0, 3.0]).mean_loss() == pytest.approx(2.0)


def test_mean_loss_without_any_data_is_nan():
    assert math.isnan(make_recorder([], []).mean_loss())
