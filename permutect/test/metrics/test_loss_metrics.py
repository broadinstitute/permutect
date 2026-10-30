import math

import matplotlib
import numpy as np
import pytest
import torch
from matplotlib import pyplot as plt

from permutect.data.batch import BatchIndexedTensor
from permutect.data.batch import BatchProperty
from permutect.data.count_binning import top_of_logit_bin
from permutect.metrics.loss_metrics import AccuracyMetrics
from permutect.metrics.loss_metrics import LossMetrics
from permutect.training.balancer import Balancer
from permutect.training.balancer import PlotType
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


class FigureRecorder:
    # a stand-in for a SummaryWriter that keeps each figure
    def __init__(self):
        self.figures = []

    def add_figure(self, tag, figure, global_step=None):
        self.figures.append(figure)


def test_balancer_and_loss_metrics_plot_counts_alike():
    # the Balancer plotted counts on a log scale with axes labeled N_alt and N_ref, and LossMetrics on a linear scale
    counts_slvra = 1000 * torch.rand(BatchIndexedTensor.shape_for(1), generator=torch.Generator().manual_seed(0))
    balancer = Balancer(num_sources=1, device=CPU)
    balancer.counts_slvra.copy_(counts_slvra)
    metrics = LossMetrics(num_sources=1, device=CPU)
    metrics.counts_slvra.copy_(counts_slvra)
    metrics.put_on_cpu()

    balancer_writer, metrics_writer = FigureRecorder(), FigureRecorder()
    balancer.make_plots(balancer_writer, "counts", epoch=1, plot_type=PlotType.COUNTS)
    metrics.make_plots(metrics_writer, "counts", Epoch.TRAIN, epoch=1, plot_type="counts")
    (balancer_figure,), (metrics_figure,) = balancer_writer.figures, metrics_writer.figures
    assert balancer_figure.get_supxlabel() == metrics_figure.get_supxlabel()
    assert balancer_figure.get_supylabel() == metrics_figure.get_supylabel()
    for balancer_axis, metrics_axis in zip(balancer_figure.axes, metrics_figure.axes, strict=True):
        for balancer_mesh, metrics_mesh in zip(balancer_axis.collections, metrics_axis.collections, strict=True):
            assert np.array_equal(balancer_mesh.get_array(), metrics_mesh.get_array())


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


def test_roc_data_by_hand():
    # 3 variants in logit bin 5, 1 variant in bin 15, and 2 artifacts each in bins 12 and 18.  Below the threshold at the
    # top of bin 5 nothing is called a non-artifact, so precision is undefined there.
    metrics = AccuracyMetrics.create(num_sources=1, device=CPU)
    for label, logit_bin, count in (
        (Label.VARIANT, 5, 3),
        (Label.VARIANT, 15, 1),
        (Label.ARTIFACT, 12, 2),
        (Label.ARTIFACT, 18, 2),
    ):
        metrics[0, label, Variation.SNV, 0, 0, logit_bin] = count

    # sensitivity, precision and artifact accuracy at the threshold at the top of each bin
    expected = {logit_bin: (0.75, 1.0, 1.0) for logit_bin in range(5, 12)}
    expected |= {logit_bin: (0.75, 0.6, 0.5) for logit_bin in range(12, 15)}
    expected |= {logit_bin: (1.0, 4 / 6, 0.5) for logit_bin in range(15, 18)}
    expected |= {logit_bin: (1.0, 0.5, 0.0) for logit_bin in range(18, 20)}
    for sens_prec in (True, False):
        roc_data = metrics.make_roc_data(None, None, None, Variation.SNV, sens_prec=sens_prec)
        assert [threshold for threshold, _, _ in roc_data] == [top_of_logit_bin(logit_bin) for logit_bin in expected]
        for (_, nonartifact_metric, artifact_metric), (sensitivity, precision, artifact_accuracy) in zip(
            roc_data, expected.values()
        ):
            assert nonartifact_metric == pytest.approx(sensitivity)
            assert artifact_metric == pytest.approx(precision if sens_prec else artifact_accuracy)
