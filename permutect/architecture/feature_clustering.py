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
        self.centroids_vke = Parameter(torch.rand(len(Variation), self.num_clusters, self.feature_dim))

        # cluster standard deviations.  Assume isotropic for now although generalizing would be
        # very easy.
        # TODO: perhaps make this depend on variant type as well?
        self.stdev_pre_exp_vk = Parameter(torch.zeros(len(Variation), self.num_clusters))

        # TODO: make this also depend on variant type?
        self.cluster_weights_pre_softmax_vk = Parameter(torch.ones(len(Variation), self.num_artifact_clusters))


    def calculate_logits(self, ref_bre: RaggedSets, alt_bre: RaggedSets, ref_counts_b: IntTensor, alt_counts_b: IntTensor, var_types_b: IntTensor):
        stdev_bk = torch.exp(self.stdev_pre_exp_vk[var_types_b])

        ref_rke, alt_rke = ref_bre.flattened_tensor_nf[:, None, :], alt_bre.flattened_tensor_nf[:, None, :]
        centroids_bke = self.centroids_vke[var_types_b]

        alt_centroids_rke = torch.repeat_interleave(centroids_bke, repeats=alt_counts_b, dim=0)
        alt_dist_rk = torch.norm(alt_rke - alt_centroids_rke, dim=-1)

        stdev_bk = torch.exp(self.stdev_pre_exp_vk)[var_types_b]
        alt_stdev_rk = torch.repeat_interleave(stdev_bk, repeats=alt_counts_b, dim=0)

        # TODO: maybe include ref reads as well in the generative modeling?
        log_lks_rk = -(self.feature_dim/2) * torch.log(alt_stdev_rk) - torch.square(alt_dist_rk) / (2 * torch.square(alt_stdev_rk))
        log_lks_brk = RaggedSets.from_flattened_tensor_and_sizes(log_lks_rk, alt_counts_b)
        log_lks_bk = log_lks_brk.sums_over_sets()


        # these are the log of weights that sum to 1
        log_artifact_cluster_weights_vk = torch.log_softmax(self.cluster_weights_pre_softmax_vk, dim=-1)
        log_artifact_cluster_weights_bk = log_artifact_cluster_weights_vk[var_types_b]
        log_lks_bk[:, 1:] += log_artifact_cluster_weights_bk

        logits_b = torch.logsumexp(log_lks_bk[:, 1:], dim=-1) - log_lks_bk[:, 0]
        return logits_b, log_lks_bk

    # avoid implicit forward calls because PyCharm doesn't recognize them
    def forward(self, features: Tensor):
        pass