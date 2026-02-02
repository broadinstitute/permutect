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
        self.alt_centroids_ke = Parameter(torch.rand(self.num_clusters, self.feature_dim))
        self.ref_centroids_ke = Parameter(torch.rand(self.num_clusters, self.feature_dim))

        # cluster standard deviations.  Isotropic diagonal
        self.alt_log_stdev_k = Parameter(torch.zeros(self.num_clusters))
        self.ref_log_stdev_k = Parameter(torch.zeros(self.num_clusters))

        self.cluster_weights_pre_softmax_k = Parameter(torch.ones(self.num_artifact_clusters))

    def log_likelihoods(self, reads_bre: RaggedSets, centroids_ke, log_stdev_k, counts_b: IntTensor, detach_reads: bool = False):
        reads_rke = reads_bre.flattened_tensor_nf[:, None, :]
        reads_rke = reads_rke.detach() if detach_reads else reads_rke
        centroids_rke = centroids_ke[None, :, :]
        diff_rke = reads_rke - centroids_rke

        log_stdev_rk = log_stdev_k[None, :]
        stdev_rk = torch.exp(log_stdev_rk)

        log_lks_rk = -self.feature_dim * log_stdev_rk  - torch.sum(torch.square(diff_rke), dim=-1) / (2 * torch.square(stdev_rk))
        log_lks_brk = RaggedSets.from_flattened_tensor_and_sizes(log_lks_rk, counts_b)
        log_lks_bk = log_lks_brk.sums_over_sets()
        return log_lks_bk

    def weighted_log_likelihoods_bk(self, ref_bre: RaggedSets, alt_bre: RaggedSets, ref_counts_b: IntTensor, alt_counts_b: IntTensor, var_types_b: IntTensor, detach_reads: bool = False):

        alt_log_lks_bk = self.log_likelihoods(reads_bre=alt_bre, centroids_ke=self.alt_centroids_ke,
                                              log_stdev_k=self.alt_log_stdev_k, counts_b=alt_counts_b, detach_reads=detach_reads)

        ref_log_lks_bk = self.log_likelihoods(reads_bre=ref_bre, centroids_ke=self.ref_centroids_ke,
                                              log_stdev_k=self.ref_log_stdev_k, counts_b=ref_counts_b,
                                              detach_reads=detach_reads)

        log_lks_bk = alt_log_lks_bk + ref_log_lks_bk

        # these are the log of weights that sum to 1
        log_artifact_cluster_weights_k = torch.log_softmax(self.cluster_weights_pre_softmax_k, dim=-1)
        log_artifact_cluster_weights_bk = log_artifact_cluster_weights_k[None:, ]
        log_lks_bk[:, 1:] += log_artifact_cluster_weights_bk
        return log_lks_bk

    def calculate_logits(self, ref_bre: RaggedSets, alt_bre: RaggedSets, ref_counts_b: IntTensor, alt_counts_b: IntTensor, var_types_b: IntTensor, detach_reads: bool = False):
        log_lks_bk = self.weighted_log_likelihoods_bk(ref_bre=ref_bre, alt_bre=alt_bre, ref_counts_b=ref_counts_b,
            alt_counts_b=alt_counts_b, var_types_b=var_types_b, detach_reads=detach_reads)

        artifact_log_lk_b = torch.logsumexp(log_lks_bk[:, 1:], dim=-1)
        non_artifact_log_lk_b = log_lks_bk[:, 0]

        logits_b = artifact_log_lk_b - non_artifact_log_lk_b
        return logits_b, log_lks_bk

    # avoid implicit forward calls because PyCharm doesn't recognize them
    def forward(self, features: Tensor):
        pass