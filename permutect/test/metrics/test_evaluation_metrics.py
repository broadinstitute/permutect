import re

import numpy as np
import pytest
import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from permutect.architecture.artifact_model import record_embeddings
from permutect.data.batch import Batch
from permutect.data.count_binning import NUM_ALT_COUNT_BINS
from permutect.data.count_binning import alt_count_bin_name
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


def test_record_batch_metadata():
    metrics = EmbeddingMetrics()
    labels = [Label.ARTIFACT, Label.VARIANT, Label.UNLABELED.value]  # enums or their int values
    variant_types = [Variation.SNV, Variation.INSERTION.value, Variation.DELETION]
    metrics.record_batch(labels, variant_types, [0, 4, 100], features_be=torch.randn(3, 4))
    metrics.record_batch(
        [Label.VARIANT], [Variation.SNV], [2], torch.randn(4), correctness=[EmbeddingMetrics.FALSE_POSITIVE]
    )

    assert metrics.label_metadata == ["ARTIFACT", "VARIANT", "UNLABELED", "VARIANT"]
    assert metrics.correct_metadata == [EmbeddingMetrics.UNKNOWN] * 3 + [EmbeddingMetrics.FALSE_POSITIVE]
    assert metrics.type_metadata == ["SNV", "INSERTION", "DELETION", "SNV"]
    bin_names = [alt_count_bin_name(bin_idx) for bin_idx in range(NUM_ALT_COUNT_BINS)]
    assert metrics.truncated_count_metadata == [bin_names[0], bin_names[1], bin_names[-1], bin_names[0]]
    assert torch.vstack(metrics.features).shape == (4, 4)


class EmbeddingRecorder:
    # a stand-in for a SummaryWriter that records the metadata of each embedding projection
    def __init__(self):
        self.metadata_by_tag = {}

    def add_embedding(self, features, metadata, metadata_header, tag, global_step=None):
        self.metadata_by_tag[tag] = metadata


class FeatureSets:
    def __init__(self, means_be):
        self.means_be = means_be

    def means_over_sets(self):
        return self.means_be


class FakeArtifactModel:
    def calculate_features(self, batch):
        size = batch.size()
        return FeatureSets(torch.randn(size, 4)), FeatureSets(torch.randn(size, 4)), torch.randn(size, 4)


def test_training_projections_use_the_same_metadata_as_filter_variants():
    # projections of the training data used to label data "artifact" / "non-artifact" / "unlabeled", unlike
    # filter_variants, which uses the Label names
    data = [make_datum(label, source=0) for label in Label for _ in range(5)]
    summary_writer = EmbeddingRecorder()
    record_embeddings(FakeArtifactModel(), [Batch(data)], summary_writer)

    assert summary_writer.metadata_by_tag
    for tag, metadata in summary_writer.metadata_by_tag.items():
        labels, correctness = {row[0] for row in metadata}, {row[1] for row in metadata}
        assert labels == {label.name for label in Label}, tag
        assert correctness == {EmbeddingMetrics.UNKNOWN}, tag
