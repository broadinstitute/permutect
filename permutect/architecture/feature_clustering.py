import math
from typing import List
from typing import Tuple

import torch
from torch import IntTensor
from torch import Tensor
from torch import nn
from torch.nn import Parameter
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import orthogonal

from permutect.architecture.euclidean_transformation import EuclideanTransformation
from permutect.architecture.exponentially_modified_gaussian import ExponentiallyModifiedGaussian
from permutect.architecture.parameterizations import BoundedNumber
from permutect.architecture.parameterizations import UnitVector
from permutect.data.batch import Batch
from permutect.data.datum import Data
from permutect.sets.ragged_sets import RaggedSets

LOG2PI = torch.log(torch.tensor(2.0 * math.pi))
MAX_LOGIT = 20


def parallel_and_orthogonal_projections(vectors_re: Tensor, direction_vectors_ke) -> Tuple[Tensor, Tensor]:
    unit_vectors_ke = direction_vectors_ke / torch.norm(direction_vectors_ke, dim=-1, keepdim=True)
    dot_products_rk = vectors_re.matmul(unit_vectors_ke.t())

    parallel_projections_rke = dot_products_rk[:, :, None] * unit_vectors_ke[None, :, :]
    orthogonal_projections_rke = vectors_re[:, None, :] - parallel_projections_rke

    return dot_products_rk, orthogonal_projections_rke


"""
The -1 (last) dimension is the vector feature dimension.  Preceding dimensions can be batch or whatever.
Here b stands not for a single batch dimension but for an arbitrary number of leading dimensions and 'f'
stand for "features".

Returns a tensor indexed by whatever dimensions 'b' represented i.e. the -1 dimension is summed over and gone.
"""


def diagonal_gaussian_log_likelihood(vectors_bf: Tensor, stdev_bf: Tensor) -> Tensor:
    feature_dim = vectors_bf.shape[-1]
    normalization_part = -(feature_dim / 2) * LOG2PI - torch.sum(torch.log(stdev_bf), dim=-1)
    exponential_part = -torch.sum(torch.square(vectors_bf / stdev_bf), dim=-1) / 2
    return normalization_part + exponential_part


MIN_STDEV = 0.01
MAX_STDEV = 100
STDEV_CONSTRAINT = BoundedNumber(min_val=MIN_STDEV, max_val=MAX_STDEV)


class FeatureClustering(nn.Module):
    def __init__(self, feature_dimension: int, num_artifact_clusters: int):
        super(FeatureClustering, self).__init__()

        self.feature_dim = feature_dimension
        self.num_artifact_clusters = num_artifact_clusters

        # anisotropic, diagonal stdev of nonartifact Gaussian.  Due to the rotation above the covariance is, WLOG, diagonal
        self.nonartifact_stdev_e = Parameter(torch.ones(self.feature_dim))
        parametrize.register_parametrization(self, "nonartifact_stdev_e", STDEV_CONSTRAINT)

        # artifact clusters each have a characteristic direction vector of deviation away from the
        # nonartifact centroid.
        self.artifact_directions_ke = Parameter(torch.rand(self.num_artifact_clusters, self.feature_dim))
        parametrize.register_parametrization(self, "artifact_directions_ke", UnitVector())

        # the (1D) parallel projections of artifact reads onto the clusters' directions are posited to follow
        # exponentially-modified Gaussian distributions
        self.artifact_emg = ExponentiallyModifiedGaussian(self.num_artifact_clusters)

        # the orthogonal projections of artifact reads onto the clusters' directions is posited to follow an
        # (F-1)-dimensional isotropic Gaussian
        self.artifact_stdev_k = Parameter(torch.ones(self.num_artifact_clusters))
        parametrize.register_parametrization(self, "artifact_stdev_k", STDEV_CONSTRAINT)

        self.cluster_weights_pre_softmax_k = Parameter(torch.ones(self.num_artifact_clusters))

    def weighted_log_likelihoods_bk(self, alt_bre: RaggedSets, alt_counts_b: IntTensor):
        alt_re = alt_bre.flattened_tensor_nf

        # nonartifact Gaussian in F dimensions
        nonartifact_log_lks_r = diagonal_gaussian_log_likelihood(alt_re, stdev_bf=self.nonartifact_stdev_e[None, :])

        # outliers are modeled by a Gaussian with same shape and twice as dispersed as the nonartifact Gaussian.
        # Unsupervised loss tries to minimize probability assigned to this "pseudocluster", which is akin to
        # assigning data to areas of high probability density.
        outlier_log_lks_r = diagonal_gaussian_log_likelihood(alt_re, stdev_bf=2 * self.nonartifact_stdev_e[None, :])

        parallel_projections_rk, orthogonal_projections_rke = parallel_and_orthogonal_projections(
            vectors_re=alt_re, direction_vectors_ke=self.artifact_directions_ke
        )
        orthogonal_dist_rk = torch.norm(orthogonal_projections_rke, dim=-1)

        # the orthogonal (F-1)-dimensional Gaussian component of the artifact log likelihoods
        orthogonal_log_lks_rk = (
            -((self.feature_dim - 1) / 2) * LOG2PI
            - (self.feature_dim - 1) * torch.log(self.artifact_stdev_k)[None, :]
            - torch.square(orthogonal_dist_rk) / (2 * torch.square(self.artifact_stdev_k[None, :]))
        )

        parallel_log_lks_rk = self.artifact_emg.log_likelihood(parallel_projections_rk)

        nonartifact_log_lks_rk = nonartifact_log_lks_r[:, None]
        outlier_log_lks_rk = outlier_log_lks_r[:, None]
        artifact_log_lks_rk = orthogonal_log_lks_rk + parallel_log_lks_rk

        nonartifact_log_lks_bk = RaggedSets(nonartifact_log_lks_rk, lengths_b=alt_counts_b).sums_over_sets()
        outlier_log_lks_bk = RaggedSets(outlier_log_lks_rk, lengths_b=alt_counts_b).sums_over_sets()
        unweighted_artifact_log_lks_bk = RaggedSets(artifact_log_lks_rk, lengths_b=alt_counts_b).sums_over_sets()

        # these are the log of weights that sum to 1
        log_artifact_cluster_weights_k = torch.log_softmax(self.cluster_weights_pre_softmax_k, dim=-1)
        log_artifact_cluster_weights_bk = log_artifact_cluster_weights_k[None, :]
        artifact_log_lks_bk = unweighted_artifact_log_lks_bk + log_artifact_cluster_weights_bk

        # the first column is nonartifact; next is outlier; other columns are different artifact clusters
        return torch.cat((nonartifact_log_lks_bk, outlier_log_lks_bk, artifact_log_lks_bk), dim=-1)

    def calculate_logits(self, alt_bre: RaggedSets, batch: Batch):
        # order is 0) nonartifact; 1) outlier; 2 and up) artifact clusters
        log_lks_bk = self.weighted_log_likelihoods_bk(alt_bre=alt_bre, alt_counts_b=batch.get(Data.ALT_COUNT))

        # outliers are simply ignored for classification.  They are only used for the unsupervised loss.
        # TODO: perhaps outliers should count as artifacts?
        artifact_log_lk_b = torch.logsumexp(log_lks_bk[:, 2:], dim=-1)
        non_artifact_log_lk_b = log_lks_bk[:, 0]

        logits_b = artifact_log_lk_b - non_artifact_log_lk_b

        # the generative model can yield extremely certain results, leading to potentially exploding gradients
        # here we cap the certainty of the output logits
        capped_logits_b = MAX_LOGIT * torch.tanh(logits_b / MAX_LOGIT)
        return capped_logits_b, log_lks_bk

    # avoid implicit forward calls because PyCharm doesn't recognize them
    def forward(self, features: Tensor):
        pass
