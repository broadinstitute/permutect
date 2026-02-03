from typing import List, Tuple

import torch
from torch import nn, Tensor, IntTensor
from torch.nn import Parameter
from torch.nn.utils import parametrize

from permutect.architecture.parameterizations import BoundedNumber, UnitVector
from permutect.sets.ragged_sets import RaggedSets

LOG2 = torch.log(torch.tensor(2.0))
SQRT2 = torch.sqrt(torch.tensor(2.0))

def parallel_and_orthogonal_projections(vectors_re: Tensor, direction_vectors_ke) -> Tuple[Tensor, Tensor]:
    unit_vectors_ke = direction_vectors_ke / torch.norm(direction_vectors_ke, dim=-1, keepdim=True)
    dot_products_rk = vectors_re.matmul(unit_vectors_ke.t())

    parallel_projections_rke = dot_products_rk[:, :, None] * unit_vectors_ke[None, :, :]
    orthogonal_projections_rke = vectors_re[:, None, :] - parallel_projections_rke

    return dot_products_rk, orthogonal_projections_rke


# P(x) = (lambda / 2) * [1 - erf((mu + lambda*sigma^2 - x)/(sqrt(2)*sigma))] * \
#   exp[(lambda/2) * (2*mu + lambda*sigma^2 - 2*x)]
def emg_log_likelihood(x: Tensor, mu: Tensor, sigma: Tensor, lambd: Tensor) -> Tensor:
    variance = torch.square(sigma)

    return torch.log(lambd/2) + torch.log(torch.erfc((mu + lambd * variance - x) / (SQRT2 * sigma))) + \
        (lambd/2) * (2*mu + lambd * variance - 2*x)

MIN_STDEV = 0.0001
MAX_STDEV = 100

MIN_LAMBDA = 0.0001
MAX_LAMBDA = 100

class FeatureClustering(nn.Module):
    def __init__(self, feature_dimension: int, num_artifact_clusters: int, calibration_hidden_layer_sizes: List[int]):
        super(FeatureClustering, self).__init__()

        self.feature_dim = feature_dimension
        self.num_artifact_clusters = num_artifact_clusters


        # nonartifact reads are posited to have an isotropic Gaussian in F-dimensional space
        self.nonartifact_centroid_e = Parameter(torch.rand(self.feature_dim))
        self.nonartifact_stdev = Parameter(torch.tensor(1.0))
        parametrize.register_parametrization(self, "nonartifact_stdev", BoundedNumber(min_val=MIN_STDEV, max_val=MAX_STDEV))

        # artifact clusters each have a characteristic direction vector of deviation away from the
        # nonartifact centroid.
        self.artifact_directions_ke = Parameter(torch.rand(self.num_artifact_clusters, self.feature_dim))
        parametrize.register_parametrization(self, "artifact_directions_ke", UnitVector())


        # the parallel projections of artifact reads onto the clusters' directions is posited to follow
        # an EMG (exponentially-modified Gaussian) distribution (a Gaussian modulated to have an exp(-lambda*x)
        # long tail in the positive-x direction
        # P(x) = (lambda / 2) * [1 - erf((mu + lambda*sigma^2 - x)/(sqrt(2)*sigma))] * \
        #   exp[(lambda/2) * (2*mu + lambda*sigma^2 - 2*x)]
        self.mu_k = Parameter(2 * torch.ones(self.num_artifact_clusters))

        self.sigma_k = Parameter(torch.ones(self.num_artifact_clusters))
        parametrize.register_parametrization(self, "sigma_k", BoundedNumber(min_val=MIN_STDEV, max_val=MAX_STDEV))

        self.lambda_k = Parameter(torch.ones(self.num_artifact_clusters))
        parametrize.register_parametrization(self, "lambda_k", BoundedNumber(min_val=MIN_LAMBDA, max_val=MAX_LAMBDA))

        # the orthogonal projections of artifact reads onto the clusters' directions is posited to follow an
        # (F-1)-dimensional isotropic Gaussian
        self.artifact_stdev_k = Parameter(torch.ones(self.num_artifact_clusters))  # in (F-1)-dim space for orthogonal projection
        parametrize.register_parametrization(self, "artifact_stdev_k", BoundedNumber(min_val=MIN_STDEV, max_val=MAX_STDEV))

        self.cluster_weights_pre_softmax_k = Parameter(torch.ones(self.num_artifact_clusters))


    def weighted_log_likelihoods_bk(self, ref_bre: RaggedSets, alt_bre: RaggedSets, ref_counts_b: IntTensor, alt_counts_b: IntTensor, var_types_b: IntTensor, detach_reads: bool = False):
        alt_re = alt_bre.flattened_tensor_nf
        delta_re = alt_re - self.nonartifact_centroid_e[None, :]

        # distance from centroid in F dimensions for purpose of nonartifact Gaussian
        nonartifact_dist_r = torch.norm(delta_re, dim=-1)
        nonartifact_log_lks_r = -self.feature_dim * torch.log(self.nonartifact_stdev) - torch.square(nonartifact_dist_r) / (
                    2 * torch.square(self.nonartifact_stdev))

        parallel_projections_rk, orthogonal_projections_rke = parallel_and_orthogonal_projections(vectors_re=delta_re,
            direction_vectors_ke=self.artifact_directions_ke)
        orthogonal_dist_rk = torch.norm(orthogonal_projections_rke, dim=-1)

        # the orthogonal (F-1)-dimensional Gaussian component of the artifact log likelihoods
        orthogonal_log_lks_rk = -(self.feature_dim - 1) * torch.log(self.artifact_stdev_k)[None, :] - torch.square(orthogonal_dist_rk) / (
                2 * torch.square(self.artifact_stdev_k[None, :]))

        parallel_log_lks_rk = emg_log_likelihood(x=parallel_projections_rk, mu=self.mu_k[None, :],
                                                 sigma=self.sigma_k[None, :], lambd=self.lambda_k[None, :])

        nonartifact_log_lks_rk = nonartifact_log_lks_r[:, None]
        artifact_log_lks_rk = orthogonal_log_lks_rk + parallel_log_lks_rk

        nonartifact_log_lks_bk = RaggedSets.from_flattened_tensor_and_sizes(nonartifact_log_lks_rk, alt_counts_b).sums_over_sets()
        artifact_log_lks_bk = RaggedSets.from_flattened_tensor_and_sizes(artifact_log_lks_rk, alt_counts_b).sums_over_sets()


        # these are the log of weights that sum to 1
        log_artifact_cluster_weights_k = torch.log_softmax(self.cluster_weights_pre_softmax_k, dim=-1)
        log_artifact_cluster_weights_bk = log_artifact_cluster_weights_k[None:, ]
        artifact_log_lks_bk += log_artifact_cluster_weights_bk

        # the first column is nonartifact; other columns are different artifact clusters
        return torch.cat((nonartifact_log_lks_bk, artifact_log_lks_bk), dim=-1)

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