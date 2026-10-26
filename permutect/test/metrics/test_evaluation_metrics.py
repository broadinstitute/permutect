import re

import torch
from torch.utils.tensorboard import SummaryWriter

from permutect.metrics.evaluation_metrics import EmbeddingMetrics
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
