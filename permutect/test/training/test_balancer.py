import math

import numpy as np
import pytest
import torch
from matplotlib import pyplot as plt

from permutect.data.batch import Batch
from permutect.data.batch import BatchIndexedTensor
from permutect.data.datum import Data
from permutect.test.data.test_batch import make_datum
from permutect.test.metrics.test_loss_metrics import FigureRecorder
from permutect.training.balancer import Balancer
from permutect.training.balancer import PlotType
from permutect.utils.enums import Label
from permutect.utils.enums import Variation

CPU = torch.device("cpu")
# the bin of ref count 3 and alt count 1
REF_BIN, ALT_BIN = 1, 0


def unlabeled_batch(size: int) -> Batch:
    data = [make_datum(0, ref_count=3, alt_count=1) for _ in range(size)]
    for datum in data:
        datum.set(Data.LABEL, Label.UNLABELED)
        datum.set(Data.VARIANT_TYPE, Variation.SNV)
    return Batch(data)


def test_labeled_weights_balance_the_labels_from_the_start():
    counts_slvra = torch.zeros(BatchIndexedTensor.shape_for(num_sources=1))
    counts_slvra[0, Label.ARTIFACT, Variation.SNV, REF_BIN, ALT_BIN] = 900
    counts_slvra[0, Label.VARIANT, Variation.SNV, REF_BIN, ALT_BIN] = 90
    balancer = Balancer(num_sources=1, device=CPU, counts_slvra=counts_slvra)

    # before the Balancer sees any data, each label's total weight in the bin is half of the bin's count
    weights_l = balancer.weights_slvra[0, :, Variation.SNV, REF_BIN, ALT_BIN]
    assert 900 * weights_l[Label.ARTIFACT].item() == pytest.approx(495, rel=1e-3)
    assert 90 * weights_l[Label.VARIANT].item() == pytest.approx(495, rel=1e-3)
    # and labels in an empty bin have weight 1
    assert balancer.weights_slvra[0, [Label.ARTIFACT, Label.VARIANT], Variation.SNV, 0, 0].tolist() == [1, 1]


def test_unlabeled_weights_follow_the_models_recent_artifact_probabilities():
    # unlabeled data called artifacts first with probability 0.1 and then, for longer, 0.9.  Old counts decay, so the
    # weights come to balance the recent probabilities: a non-artifact pseudo-label gets weight (1 + 0.9/0.1)/2 = 5
    batch = unlabeled_batch(1000)
    balancer = Balancer(num_sources=1, device=CPU)
    for probability, num_batches in ((0.1, 100), (0.9, 1000)):
        for _ in range(num_batches):
            balancer.process_batch_and_compute_weights(batch, artifact_probs_b=torch.full((batch.size(),), probability))
    weights_l = balancer.unlabeled_weights_slvra[0, :, Variation.SNV, REF_BIN, ALT_BIN]
    assert weights_l[Label.VARIANT].item() == pytest.approx(5, rel=0.01)
    assert weights_l[Label.ARTIFACT].item() == pytest.approx(5 / 9, rel=0.01)


def test_weight_plots_show_the_weights_of_labeled_and_of_unlabeled_data():
    # the weights of unlabeled data, by pseudo-label, weren't plotted, while their unused weights as labeled data were
    balancer = Balancer(num_sources=1, device=CPU)
    balancer.weights_slvra[0, Label.ARTIFACT, Variation.SNV, REF_BIN, ALT_BIN] = math.e
    balancer.unlabeled_weights_slvra[0, Label.VARIANT, Variation.SNV, REF_BIN, ALT_BIN] = math.e**2
    recorder = FigureRecorder()
    balancer.make_plots(recorder, "weights", epoch=1, plot_type=PlotType.WEIGHTS)

    assert recorder.tags == ["weights", "weights, unlabeled data by pseudo-label"]
    for figure, label, log_weight in zip(recorder.figures, (Label.ARTIFACT, Label.VARIANT), (1.0, 2.0)):
        # rows of subplots for artifacts and variants, a column for each variant type, and a color bar
        assert len(figure.axes) == 2 * len(Variation) + 1
        plotted = figure.axes[label * len(Variation) + Variation.SNV].collections[0].get_array()
        assert np.isclose(plotted, log_weight).sum() == 1
    plt.close("all")
