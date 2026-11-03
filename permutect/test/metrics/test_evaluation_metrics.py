import re

import numpy as np
import pytest
import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from permutect import misc_utils
from permutect.architecture.artifact_model import record_embeddings
from permutect.architecture.feature_clustering import FeatureClustering
from permutect.data.batch import Batch
from permutect.data.count_binning import NUM_ALT_COUNT_BINS
from permutect.data.count_binning import alt_count_bin_name
from permutect.data.datum import RAW_READS_ARRAY_DTYPE
from permutect.data.datum import Data
from permutect.data.datum import Datum
from permutect.metrics.evaluation_metrics import EmbeddingMetrics
from permutect.metrics.evaluation_metrics import EvaluationMetrics
from permutect.sets.ragged_sets import RaggedSets
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


def test_unlabeled_data_appear_in_the_logit_histograms_but_not_the_roc_curves():
    # accuracy, calibration and ROC curves need labels and read only the ARTIFACT and VARIANT rows, while the logit
    # histograms show the logits of each label, including unlabeled data, which used to be recorded with zero weight
    labeled, unlabeled = [make_datum(Label.ARTIFACT, 0), make_datum(Label.VARIANT, 0)], [make_datum(Label.UNLABELED, 0)]
    roc_data = []
    for data, logits in ((labeled, [2.0, -1.5]), (labeled + unlabeled * 3, [2.0, -1.5, 0.5, 3.0, -2.0])):
        metrics = EvaluationMetrics(num_sources=1, device=torch.device("cpu"))
        metrics.record_batch(Epoch.VALID, Batch(data), logits=torch.tensor(logits))
        accuracy_metrics = metrics.put_on_cpu().accuracy_metrics_by_epoch_type[Epoch.VALID]
        roc_data.append(accuracy_metrics.make_roc_data(None, None, None, Variation.SNV, sens_prec=True))
    assert roc_data[0] == roc_data[1]

    histograms_fig, _ = accuracy_metrics.make_logit_histograms()
    assert "UNLABELED" in {line.get_label() for axis in histograms_fig.axes for line in axis.get_lines()}
    plt.close(histograms_fig)


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
    # a stand-in for a SummaryWriter that records the features and metadata of each embedding projection
    def __init__(self):
        self.features_by_tag = {}
        self.metadata_by_tag = {}

    def add_embedding(self, features, metadata, metadata_header, tag, global_step=None):
        assert all(len(row) == len(metadata_header) for row in metadata)
        self.features_by_tag[tag] = features
        self.metadata_by_tag[tag] = metadata


class FakeArtifactModel:
    def __init__(self, device=torch.device("cpu")):
        self._device = device
        self.batch_devices = set()
        self.feature_clustering = FeatureClustering(feature_dimension=4, num_artifact_clusters=2)

    def calculate_features(self, batch):
        self.batch_devices.add(batch.get_reads_re().device)
        ref_counts_b, alt_counts_b = batch.get(Data.REF_COUNT), batch.get(Data.ALT_COUNT)
        ref_bre = RaggedSets(torch.randn(int(ref_counts_b.sum()), 4), ref_counts_b)
        alt_bre = RaggedSets(torch.randn(int(alt_counts_b.sum()), 4), alt_counts_b)
        return ref_bre, alt_bre, torch.randn(batch.size(), 4)


def test_training_projections_use_the_same_metadata_as_filter_variants():
    # projections of the training data used to label data "artifact" / "non-artifact" / "unlabeled", unlike
    # filter_variants, which uses the Label names
    data = [make_datum(label, source=0) for label in Label for _ in range(5)]
    summary_writer = EmbeddingRecorder()
    record_embeddings(FakeArtifactModel(), [Batch(data)], summary_writer)

    assert summary_writer.metadata_by_tag
    for tag, metadata in summary_writer.metadata_by_tag.items():
        data_rows = [row for row in metadata if row[4] == EmbeddingMetrics.DATA_SOURCE]
        labels, correctness = {row[0] for row in data_rows}, {row[1] for row in data_rows}
        assert labels == {label.name for label in Label}, tag
        assert correctness == {EmbeddingMetrics.UNKNOWN}, tag


TWIN_LABELS = ("model artifact", "model variant", "model unlabeled")


def test_training_projections_show_twins_simulated_from_the_model():
    torch.manual_seed(0)
    data = [make_datum(label, source=0) for label in Label for _ in range(20)]
    summary_writer = EmbeddingRecorder()
    record_embeddings(FakeArtifactModel(), [Batch(data)], summary_writer)

    for tag, metadata in summary_writer.metadata_by_tag.items():
        twin_rows = [row for row in metadata if row[4] != EmbeddingMetrics.DATA_SOURCE]
        if tag.startswith("embedding"):
            # the twins' correctness is "model", whatever their data's
            assert {row[1] for row in twin_rows} == {EmbeddingMetrics.MODEL}, tag

            # labeled data's twins are artifacts if and only if the data are, and unlabeled data's twins are either
            sources_by_label = {label: {row[4] for row in twin_rows if row[0] == label} for label in TWIN_LABELS}
            assert sources_by_label["model artifact"] == {"model artifact cluster 0", "model artifact cluster 1"}, tag
            assert sources_by_label["model variant"] == {"model nonartifact"}, tag
            assert sources_by_label["model unlabeled"] <= {
                "model nonartifact",
                "model artifact cluster 0",
                "model artifact cluster 1",
            }, tag
        else:
            # the model doesn't describe ref embeddings or the ref and alt allele context
            assert not twin_rows, tag


def test_projections_pair_each_datum_with_its_twin():
    metrics = EmbeddingMetrics()
    labels, variant_types, alt_counts = [Label.ARTIFACT, Label.VARIANT, Label.UNLABELED], [Variation.SNV] * 3, [1, 5, 9]
    correctness = [EmbeddingMetrics.TRUE_NEGATIVE_ARTIFACT, EmbeddingMetrics.TRUE_POSITIVE, EmbeddingMetrics.UNKNOWN]
    features_be, ref_features_be, simulated_be = torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 4)
    sources = ["artifact cluster 1", "nonartifact", "artifact cluster 0"]
    metrics.record_batch(
        labels, variant_types, alt_counts, features_be, ref_features_be, correctness, simulated_be, sources
    )
    summary_writer = EmbeddingRecorder()
    metrics.output_to_summary_writer(summary_writer)

    tag = "embedding for variant type SNV"
    metadata, features = summary_writer.metadata_by_tag[tag], summary_writer.features_by_tag[tag]
    assert len(metadata) == 6
    data_rows, twin_rows = metadata[:3], metadata[3:]
    assert [row[4] for row in data_rows] == ["data"] * 3
    for data_row, twin_row, data_features, twin_features in zip(data_rows, twin_rows, features[:3], features[3:]):
        # the projection's subsample shuffles the data, so find each datum by its label
        n = [label.name for label in labels].index(data_row[0])
        assert data_row[1] == correctness[n]

        # the twin shares the datum's type and count, but its label and correctness say that it's from the model
        assert twin_row == (TWIN_LABELS[n], EmbeddingMetrics.MODEL, data_row[2], data_row[3], "model " + sources[n])
        assert torch.equal(data_features, features_be[n])
        assert torch.equal(twin_features, simulated_be[n])

    ref_metadata = summary_writer.metadata_by_tag["ref embedding for variant type SNV"]
    assert [row[4] for row in ref_metadata] == ["data"] * 3


def test_record_embeddings_sends_batches_to_the_model_device(monkeypatch):
    # if the default device isn't the model's device, batches must still go to the model's device
    monkeypatch.setattr(misc_utils, "gpu_if_available", lambda exploit_mps=False: torch.device("meta"))
    model = FakeArtifactModel(device=torch.device("cpu"))
    data = [make_datum(label, source=0) for label in Label for _ in range(5)]
    record_embeddings(model, [Batch(data)], EmbeddingRecorder())
    assert model.batch_devices == {torch.device("cpu")}
