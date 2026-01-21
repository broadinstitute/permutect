from typing import List

import torch
from matplotlib import pyplot as plt
from torch import nn, Tensor, IntTensor, repeat_interleave
from torch.nn import Parameter

from permutect.architecture.monotonic import MonoDense
from permutect.data.count_binning import MAX_REF_COUNT, MAX_ALT_COUNT
from permutect.metrics import plotting
from permutect.sets.ragged_sets import RaggedSets
from permutect.utils.enums import Variation


class FeatureClustering(nn.Module):
    VAR_TYPE_EMBEDDING_DIM = 10
    def __init__(self, feature_dimension: int, num_artifact_clusters: int, calibration_hidden_layer_sizes: List[int]):
        super(FeatureClustering, self).__init__()

        self.feature_dim = feature_dimension
        self.num_artifact_clusters = num_artifact_clusters

        # the 0th cluster is non-artifact
        self.num_clusters = self.num_artifact_clusters + 1

        # num_clusters different centroids, each a vector in feature space.  Initialize even weights.
        self.centroids_ke = Parameter(torch.rand(self.num_clusters, self.feature_dim))
        self.cluster_weights_pre_softmax_k = Parameter(torch.ones(self.num_artifact_clusters))
        self.centroid_distance_normalization = Parameter(1 / torch.sqrt(torch.tensor(self.feature_dim)), requires_grad=False)

        # calibration takes [distance from centroid, ref count, alt count] as input and maps it to [calibrated distance]
        # It is monotonically increasing in the input distance.  We don't currently constrain the count dependences, but
        # it's worth considering.

        # To ensure that zero maps to zero we take f(distance, counts) = g(distance, counts) - g(distance=0, counts),
        # where g has the desired monotonicity.

        # the input features are distance, ref count, alt count, var_type embedding
        # calibration is increasing in distance.
        self.distance_calibration = MonoDense(3 + FeatureClustering.VAR_TYPE_EMBEDDING_DIM, calibration_hidden_layer_sizes + [1], 1, 0)
        self.var_type_embeddings_ve = Parameter(torch.rand(len(Variation), FeatureClustering.VAR_TYPE_EMBEDDING_DIM))

    def centroid_distances(self, features_be: Tensor) -> Tensor:
        batch_size = len(features_be)
        centroids_bke = self.centroids_ke.view(1, self.num_clusters, self.feature_dim)
        features_bke = features_be.view(batch_size, 1, self.feature_dim)
        diff_bke = centroids_bke - features_bke
        dist_bk = torch.norm(diff_bke, dim=-1) * self.centroid_distance_normalization
        return dist_bk

    def calculate_logits(self, ref_bre: RaggedSets, alt_bre: RaggedSets, ref_counts_b: IntTensor, alt_counts_b: IntTensor, var_types_b: IntTensor):
        batch_size = len(features_be)
        dist_bk = self.centroid_distances(features_be)

        # flatten b,k indices to a single pseudo-batch index, then unflatten
        cal_dist_bk = self.calibrated_distances(dist_bk.view(-1), repeat_interleave(ref_counts_b, self.num_clusters),
            repeat_interleave(alt_counts_b, self.num_clusters), repeat_interleave(var_types_b, self.num_clusters)).view(batch_size, self.num_clusters)

        uncal_logits_bk = -dist_bk
        cal_logits_bk = -cal_dist_bk

        # these are the log of weights that sum to 1
        log_artifact_cluster_weights_k = torch.log_softmax(self.cluster_weights_pre_softmax_k, dim=-1)
        log_artifact_cluster_weights_bk = log_artifact_cluster_weights_k.view(1, -1)    # dummy batch dimension for broadcasting

        # total all the artifact clusters in log space and subtract the non-artifact cluster
        uncalibrated_logits_b = torch.logsumexp(log_artifact_cluster_weights_bk + uncal_logits_bk[:, 1:],
                                                dim=-1) - uncal_logits_bk[:, 0]
        weighted_cal_logits_bk = cal_logits_bk
        weighted_cal_logits_bk[:, 1:] += log_artifact_cluster_weights_bk
        calibrated_logits_b = torch.logsumexp(weighted_cal_logits_bk[:, 1:], dim=-1) - cal_logits_bk[:, 0]
        return calibrated_logits_b, uncalibrated_logits_b, weighted_cal_logits_bk

    # avoid implicit forward calls because PyCharm doesn't recognize them
    def forward(self, features: Tensor):
        pass

    def calibrated_distances(self, distances_b: Tensor, ref_counts_b: Tensor, alt_counts_b: Tensor, var_types_b: IntTensor):
        # indices: 'b' for batch, 3 for logit, ref, alt
        ref_b1 = ref_counts_b.view(-1, 1) / MAX_REF_COUNT
        alt_b1 = alt_counts_b.view(-1, 1) / MAX_ALT_COUNT
        var_type_embeddings_ve = self.var_type_embeddings_ve[var_types_b]

        monotonic_inputs_be = torch.hstack((distances_b.view(-1, 1), ref_b1, alt_b1, var_type_embeddings_ve))
        zero_inputs_be = torch.hstack((torch.zeros_like(distances_b).view(-1, 1), ref_b1, alt_b1, var_type_embeddings_ve))
        result_b1 = self.distance_calibration.forward(monotonic_inputs_be) - self.distance_calibration.forward(zero_inputs_be)
        return result_b1.view(-1)