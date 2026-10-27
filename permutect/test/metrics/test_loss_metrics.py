import math

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt

from permutect.data.batch import BatchProperty
from permutect.metrics.loss_metrics import AccuracyMetrics
from permutect.metrics.loss_metrics import LossMetrics
from permutect.utils.enums import Epoch
from permutect.utils.enums import Label
from permutect.utils.enums import Variation

matplotlib.use("Agg")

CPU = torch.device("cpu")


def fully_labeled_loss_metrics() -> LossMetrics:
    # artifact and variant SNVs in a single ref/alt count cell, and no unlabeled data
    metrics = LossMetrics(num_sources=1, device=CPU)
    for label, loss in ((Label.ARTIFACT, 0.5), (Label.VARIANT, 0.25)):
        metrics.counts_slvra[0, label, Variation.SNV, 1, 1] = 4
        metrics.totals_slvra[0, label, Variation.SNV, 1, 1] = 4 * loss
    return metrics.put_on_cpu()


class RecordingSummaryWriter:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, heading, value, step):
        self.scalars[heading] = value


def test_marginals_skip_empty_categories(capsys):
    metrics = fully_labeled_loss_metrics()
    marginal_l = metrics.get_marginal(BatchProperty.LABEL)
    assert marginal_l[Label.ARTIFACT].item() == 0.5
    assert marginal_l[Label.VARIANT].item() == 0.25
    assert math.isnan(marginal_l[Label.UNLABELED].item())

    metrics.print_marginals("loss")
    printed = capsys.readouterr().out
    assert "nan" not in printed
    assert "ARTIFACT: 0.500" in printed and "VARIANT: 0.250" in printed
    assert "UNLABELED" not in printed

    summary_writer = RecordingSummaryWriter()
    metrics.write_marginals(Epoch.TRAIN, 0, summary_writer, prefix="loss")
    assert not any(math.isnan(value) for value in summary_writer.scalars.values())
    assert summary_writer.scalars["loss/TRAIN/LABEL/VARIANT"] == 0.25
    assert "loss/TRAIN/LABEL/UNLABELED" not in summary_writer.scalars


def test_heatmaps_leave_empty_cells_blank():
    _, axis = plt.subplots()

    loss_metrics = fully_labeled_loss_metrics()
    loss_values = loss_metrics.plot_losses(Label.VARIANT, Variation.SNV, axis, source=0).get_array()
    assert np.ma.count_masked(loss_values) == loss_values.size - 1
    assert loss_values.compressed().tolist() == [0.25]

    accuracy_metrics = AccuracyMetrics.create(num_sources=1, device=CPU)
    accuracy_metrics[0, Label.VARIANT, Variation.SNV, 1, 1, 0] = 3  # lowest logit bin, ie correctly called variant
    accuracy_values = accuracy_metrics.plot_accuracy(Label.VARIANT, Variation.SNV, axis, source=0).get_array()
    assert np.ma.count_masked(accuracy_values) == accuracy_values.size - 1
    assert accuracy_values.compressed().tolist() == [1.0]

    # accuracy is undefined for unlabeled data
    unlabeled_values = accuracy_metrics.plot_accuracy(Label.UNLABELED, Variation.SNV, axis, source=0).get_array()
    assert np.ma.count_masked(unlabeled_values) == unlabeled_values.size
    plt.close("all")
