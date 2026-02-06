import math
from typing import List, Tuple

import torch
from pytensor.printing import use_ascii
from torch import nn, Tensor, IntTensor
from torch.nn import Parameter
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import orthogonal

from permutect.architecture.parameterizations import BoundedNumber, UnitVector
from permutect.sets.ragged_sets import RaggedSets

LOG2 = torch.log(torch.tensor(2.0))
LOG2PI = torch.log(torch.tensor(2.0 * math.pi))
SQRTPI = torch.sqrt(torch.tensor(math.pi))
SQRT2 = torch.sqrt(torch.tensor(2.0))
MAX_LOGIT = 20

def parallel_and_orthogonal_projections(vectors_re: Tensor, direction_vectors_ke) -> Tuple[Tensor, Tensor]:
    unit_vectors_ke = direction_vectors_ke / torch.norm(direction_vectors_ke, dim=-1, keepdim=True)
    dot_products_rk = vectors_re.matmul(unit_vectors_ke.t())

    parallel_projections_rke = dot_products_rk[:, :, None] * unit_vectors_ke[None, :, :]
    orthogonal_projections_rke = vectors_re[:, None, :] - parallel_projections_rke

    return dot_products_rk, orthogonal_projections_rke

"""
Pytorch has no built-in logerfc, and erfc(z) decays so fast (faster than exp(-z^2)!) that it underflows to zero
leading to NaNs when its log is taken.  Here for large arguments we use the asymptotic expansion:

erfc(z) ~ [exp(-z^2)/(z sqrt(pi))] * [1 - 1/(2z^2) + 3/(4z^4) - 15/(8z^6) . . .]
"""
def logerfc(z: Tensor) -> Tensor:
    use_asymptotic = z > 5

    z_squared = z*z
    z_fourth = z_squared * z_squared
    z_sixth = z_squared * z_fourth

    asymptotic = -z_squared - torch.log(z * SQRTPI) + torch.log1p(-1 / (2 * z_squared) + 3 / (4 * z_fourth) - 15 / (8 * z_sixth))
    built_in = torch.log(torch.erfc(z))
    return torch.where(use_asymptotic, asymptotic, built_in)


# P(x) = (lambda / 2) * [1 - erf((mu + lambda*sigma^2 - x)/(sqrt(2)*sigma))] * \
#   exp[(lambda/2) * (2*mu + lambda*sigma^2 - 2*x)]
def emg_log_likelihood(x: Tensor, mu: Tensor, sigma: Tensor, lambd: Tensor) -> Tensor:
    variance = torch.square(sigma)

    return torch.log(lambd/2) + logerfc((mu + lambd * variance - x) / (SQRT2 * sigma)) + \
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


        # nonartifact reads are posited to have a Gaussian in F-dimensional space
        # we shift and rotate so that the Gaussian is zero-centered and has diagonal covariance
        self.read_translation_e = Parameter(torch.rand(self.feature_dim))
        self.read_rotation_ee = orthogonal(torch.nn.Linear(self.feature_dim, self.feature_dim, bias=False))

        # anisotropic, diagonal stdev of nonartifact Gaussian.  Due to the rotation above the covariance is, WLOG, diagonal
        self.nonartifact_stdev_e = Parameter(torch.ones(self.feature_dim))
        parametrize.register_parametrization(self, "nonartifact_stdev_e", BoundedNumber(min_val=MIN_STDEV, max_val=MAX_STDEV))

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

    def transform_reads(self, reads_bre: RaggedSets) -> RaggedSets:
        return reads_bre.apply_elementwise(lambda reads_re: self.read_rotation_ee(reads_re + self.read_translation_e[None, :]))


    def weighted_log_likelihoods_bk(self, ref_bre: RaggedSets, alt_bre: RaggedSets, ref_counts_b: IntTensor, alt_counts_b: IntTensor, var_types_b: IntTensor, detach_reads: bool = False):
        # recenter reads so that Gaussian's centroid is the origin
        shifted_alt_bre, shifted_ref_bre = self.transform_reads(alt_bre), self.transform_reads(ref_bre)
        alt_re = shifted_alt_bre.flattened_tensor_nf

        # nonartifact Gaussian in F dimensions
        nonartifact_log_lks_r = -(self.feature_dim/2)*LOG2PI -torch.sum(torch.log(self.nonartifact_stdev_e)) - \
                                torch.sum(torch.square(alt_re / self.nonartifact_stdev_e[None, :]), dim=-1)/2

        parallel_projections_rk, orthogonal_projections_rke = parallel_and_orthogonal_projections(vectors_re=alt_re,
            direction_vectors_ke=self.artifact_directions_ke)
        orthogonal_dist_rk = torch.norm(orthogonal_projections_rke, dim=-1)

        # the orthogonal (F-1)-dimensional Gaussian component of the artifact log likelihoods
        orthogonal_log_lks_rk = -((self.feature_dim-1)/2)*LOG2PI -(self.feature_dim - 1) * torch.log(self.artifact_stdev_k)[None, :] - torch.square(orthogonal_dist_rk) / (
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

        # the generative model can yield extremely certain results, leading to potentially exploding gradients
        # here we cap the certainty of the output logits
        capped_logits_b = MAX_LOGIT * torch.tanh(logits_b / MAX_LOGIT)
        return capped_logits_b, log_lks_bk

    # avoid implicit forward calls because PyCharm doesn't recognize them
    def forward(self, features: Tensor):
        pass