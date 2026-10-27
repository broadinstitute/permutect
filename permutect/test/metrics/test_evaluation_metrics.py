import re

import numpy as np
import pytest
import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from permutect.data.batch import Batch
from permutect.data.datum import RAW_READS_ARRAY_DTYPE
from permutect.data.datum import Data
from permutect.data.datum import Datum
from permutect.metrics.evaluation_metrics import EmbeddingMetrics
from permutect.metrics.evaluation_metrics import EvaluationMetrics
from permutect.utils.enums import Epoch
from permutect.utils.enums import Label
from permutect.utils.enums import Variation


def make_embedding_metrics(variant_types) -> EmbeddingMetrics:
    metrics = EmbeddingMetrics()
    num_data = len(variant_types)
    metrics.label_metadata.extend(["artifact"] * num_data)
    metrics.correct_metadata.extend(["unknown"] * num_data)
    metrics.type_metadata.extend([variant_type.name for variant_type in variant_types])
    metrics.truncated_count_metadata.extend(["2"] * num_data)
    metrics.features.append(torch.randn(num_data, 4))
    metrics.ref_features.append(torch.randn(num_data, 4))
    return metrics


def recorded_embedding_tags(log_dir) -> set:
    # TensorBoard lists each recorded embedding as tensor_name: "<tag>:<step>" in the projector config
    config = (log_dir / "projector_config.pbtxt").read_text()
    return set(re.findall(r'tensor_name: "(.*):\d+"', config))


def test_embeddings_for_every_variant_type(tmp_path):
    summary_writer = SummaryWriter(log_dir=str(tmp_path))
    make_embedding_metrics(list(Variation) * 3).output_to_summary_writer(summary_writer)
    summary_writer.close()

    tags = recorded_embedding_tags(tmp_path)
    for variant_type in Variation:
        assert "embedding for variant type " + variant_type.name in tags
        assert "ref embedding for variant type " + variant_type.name in tags


def test_embeddings_skip_variant_types_without_data(tmp_path):
    # e.g. an SNV-only panel, which used to crash after training but before the model was saved
    summary_writer = SummaryWriter(log_dir=str(tmp_path))
    make_embedding_metrics([Variation.SNV] * 6).output_to_summary_writer(summary_writer)
    summary_writer.close()

    tags = recorded_embedding_tags(tmp_path)
    assert {"embedding for variant type SNV", "ref embedding for variant type SNV"} <= tags
    for variant_type in Variation:
        if variant_type != Variation.SNV:
            assert not any(variant_type.name in tag for tag in tags)


def test_mistake_histograms_with_no_mistakes(tmp_path):
    # filter_variants with a truth VCF where every labeled site is called correctly used to crash here
    summary_writer = SummaryWriter(log_dir=str(tmp_path))
    evaluation_metrics = EvaluationMetrics(num_sources=1, device=torch.device("cpu")).put_on_cpu()
    evaluation_metrics.make_mistake_histograms(summary_writer)
    summary_writer.close()


class FigureTagRecorder:
    # a stand-in for a SummaryWriter that records the tag of each figure
    def __init__(self):
        self.tags = []

    def add_figure(self, tag, figure, global_step=None):
        self.tags.append(tag)
        plt.close(figure)


def make_datum(label: Label, source: int) -> Datum:
    datum = Datum(
        int_array=np.zeros(Data.NUM_SCALAR_INT_ELEMENTS),
        float_array=np.zeros(Data.NUM_SCALAR_FLOAT_ELEMENTS),
        reads_re=np.zeros((3, 2), dtype=RAW_READS_ARRAY_DTYPE),
    )
    datum.set(Data.LABEL, label)
    datum.set(Data.SOURCE, source)
    datum.set(Data.REF_COUNT, 2)
    datum.set(Data.ALT_COUNT, 1)
    return datum


@pytest.mark.parametrize("num_sources", [1, 2])
def test_each_evaluation_figure_is_written_once(num_sources):
    metrics = EvaluationMetrics(num_sources=num_sources, device=torch.device("cpu"))
    data = [make_datum(label, source) for label in (Label.ARTIFACT, Label.VARIANT) for source in range(num_sources)]
    epoch_types = [Epoch.TRAIN, Epoch.VALID]
    for epoch_type in epoch_types:
        metrics.record_batch(epoch_type, Batch(data), logits=torch.zeros(len(data)))
    metrics.put_on_cpu()

    summary_writer = FigureTagRecorder()
    metrics.make_plots(summary_writer, epoch=1)
    assert len(summary_writer.tags) == len(set(summary_writer.tags))

    # for each epoch type, accuracy, calibration and ROC figures for each source and for all sources if there are
    # several, then logit histograms
    num_source_plots = num_sources if num_sources == 1 else num_sources + 1
    assert len(summary_writer.tags) == len(epoch_types) * (3 * num_source_plots + 1)
