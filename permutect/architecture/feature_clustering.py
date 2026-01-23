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

        # num_clusters different centroids for each variant type, each a vector in feature space.  Initialize even weights.
        self.alt_centroids_vke = Parameter(torch.rand(len(Variation), self.num_clusters, self.feature_dim))
        self.ref_centroids_vke = Parameter(torch.rand(len(Variation), self.num_clusters, self.feature_dim))

        # cluster standard deviations.  Isotropic diagonal
        self.alt_log_stdev_vk = Parameter(torch.zeros(len(Variation), self.num_clusters))
        self.ref_log_stdev_vk = Parameter(torch.zeros(len(Variation), self.num_clusters))

        self.cluster_weights_pre_softmax_vk = Parameter(torch.ones(len(Variation), self.num_artifact_clusters))

    def log_likelihoods(self, reads_bre: RaggedSets, centroids_vke, log_stdev_vk, counts_b: IntTensor, var_types_b: IntTensor):
        reads_rke = reads_bre.flattened_tensor_nf[:, None, :]
        centroids_bke = centroids_vke[var_types_b]
        centroids_rke = torch.repeat_interleave(centroids_bke, repeats=counts_b, dim=0)
        diff_rke = reads_rke - centroids_rke
        log_stdev_bk = log_stdev_vk[var_types_b]
        log_stdev_rk = torch.repeat_interleave(log_stdev_bk, repeats=counts_b, dim=0)
        stdev_rk = torch.exp(log_stdev_rk)

        log_lks_rk = -self.feature_dim * log_stdev_rk  - torch.sum(torch.square(diff_rke), dim=-1) / (2 * torch.square(stdev_rk))
        log_lks_brk = RaggedSets.from_flattened_tensor_and_sizes(log_lks_rk, counts_b)
        log_lks_bk = log_lks_brk.sums_over_sets()
        return log_lks_bk

    def calculate_logits(self, ref_bre: RaggedSets, alt_bre: RaggedSets, ref_counts_b: IntTensor, alt_counts_b: IntTensor, var_types_b: IntTensor):

        alt_log_lks_bk = self.log_likelihoods(reads_bre=alt_bre, centroids_vke=self.alt_centroids_vke,
                                              log_stdev_vk=self.alt_log_stdev_vk, counts_b=alt_counts_b, var_types_b=var_types_b)

        ref_log_lks_bk = self.log_likelihoods(reads_bre=ref_bre, centroids_vke=self.ref_centroids_vke,
                                              log_stdev_vk=self.ref_log_stdev_vk, counts_b=ref_counts_b,
                                              var_types_b=var_types_b)

        log_lks_bk = alt_log_lks_bk + ref_log_lks_bk

        # these are the log of weights that sum to 1
        log_artifact_cluster_weights_vk = torch.log_softmax(self.cluster_weights_pre_softmax_vk, dim=-1)
        log_artifact_cluster_weights_bk = log_artifact_cluster_weights_vk[var_types_b]
        log_lks_bk[:, 1:] += log_artifact_cluster_weights_bk

        logits_b = torch.logsumexp(log_lks_bk[:, 1:], dim=-1) - log_lks_bk[:, 0]
        return logits_b, log_lks_bk

    # avoid implicit forward calls because PyCharm doesn't recognize them
    def forward(self, features: Tensor):
        pass