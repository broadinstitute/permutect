import math

import torch
from torch import IntTensor
from torch import Tensor
from torch import nn
from torch.distributions import Distribution
from torch.distributions import Independent
from torch.distributions import Normal
from torch.distributions import constraints
from torch.nn import Parameter
from torch.nn.utils import parametrize

from permutect.architecture.exponentially_modified_gaussian import STDEV_CONSTRAINT
from permutect.architecture.exponentially_modified_gaussian import ExponentiallyModifiedGaussian
from permutect.architecture.exponentially_modified_gaussian import ExponentiallyModifiedGaussianDistribution
from permutect.architecture.parameterizations import LogWeights
from permutect.architecture.parameterizations import UnitVector
from permutect.data.batch import Batch
from permutect.data.datum import Data
from permutect.sets.ragged_sets import RaggedSets

LOG2PI = math.log(2.0 * math.pi)
LOGIT_CAP = 20


def parallel_and_orthogonal_projections(vectors: Tensor, unit_vectors: Tensor) -> tuple[Tensor, Tensor]:
    """
    The (scalar) projections of vectors onto unit vectors, and the (vector) projections onto their orthogonal
    complements.  The last dimension is the vector dimension; the others broadcast.
    """
    dot_products = torch.sum(vectors * unit_vectors, dim=-1)
    return dot_products, vectors - dot_products[..., None] * unit_vectors


def diagonal_gaussian(stdev_e: Tensor) -> Independent:
    """
    The zero-centered Gaussian with diagonal covariance whose standard deviations are stdev_e.  Events are vectors.
    """
    # the parametrizations already constrain the standard deviations, and validating them makes the CPU wait for the GPU
    return Independent(Normal(torch.zeros_like(stdev_e), stdev_e, validate_args=False), 1, validate_args=False)


class ArtifactClusterDistribution(Distribution):
    """
    The distribution of artifact read embeddings in a cluster with a characteristic unit direction vector of deviation
    away from the nonartifact centroid at the origin.  The (1D) parallel projection x.u of an embedding x onto the
    cluster's direction u follows an exponentially-modified Gaussian distribution, and the orthogonal projection
    x - (x.u)u follows an isotropic Gaussian in the (F-1)-dimensional subspace orthogonal to u.

    Events are F-dimensional embeddings.  Here k stands not for a single cluster dimension but for the batch shape, an
    arbitrary number of leading dimensions: the directions_ke have shape (batch shape) + (F,), while the parallel EMGs and
    the orthogonal standard deviations have shape (batch shape).  They broadcast against each other.
    """

    arg_constraints = {"directions_ke": constraints.real_vector, "orthogonal_stdev_k": constraints.positive}
    support = constraints.real_vector
    has_rsample = True

    def __init__(
        self,
        directions_ke: Tensor,
        parallel_emg: ExponentiallyModifiedGaussianDistribution,
        orthogonal_stdev_k: Tensor,
        validate_args=None,
    ):
        batch_shape = torch.broadcast_shapes(
            directions_ke.shape[:-1], parallel_emg.batch_shape, orthogonal_stdev_k.shape
        )
        event_shape = directions_ke.shape[-1:]
        self.directions_ke = directions_ke.expand(batch_shape + event_shape)
        self.parallel_emg = parallel_emg.expand(batch_shape)
        self.orthogonal_stdev_k = orthogonal_stdev_k.expand(batch_shape)
        super().__init__(batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ArtifactClusterDistribution, _instance)
        batch_shape = torch.Size(batch_shape)
        new.directions_ke = self.directions_ke.expand(batch_shape + self.event_shape)
        new.parallel_emg = self.parallel_emg.expand(batch_shape)
        new.orthogonal_stdev_k = self.orthogonal_stdev_k.expand(batch_shape)
        super(ArtifactClusterDistribution, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self) -> Tensor:
        return self.parallel_emg.mean[..., None] * self.directions_ke

    @property
    def variance(self) -> Tensor:
        # the diagonal of the covariance matrix sigma^2 (I - uu^T) + Var(x.u) uu^T
        directions_squared_ke = torch.square(self.directions_ke)
        orthogonal_variance_k = torch.square(self.orthogonal_stdev_k)
        return (
            orthogonal_variance_k[..., None] * (1 - directions_squared_ke)
            + self.parallel_emg.variance[..., None] * directions_squared_ke
        )

    def rsample(self, sample_shape=torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        gaussian = self.orthogonal_stdev_k[..., None] * torch.randn(
            shape, dtype=self.directions_ke.dtype, device=self.directions_ke.device
        )
        _, orthogonal_projections = parallel_and_orthogonal_projections(gaussian, self.directions_ke)
        parallel_projections = self.parallel_emg.rsample(sample_shape)
        return orthogonal_projections + parallel_projections[..., None] * self.directions_ke

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        feature_dim = self.event_shape[0]
        parallel_projections, orthogonal_projections = parallel_and_orthogonal_projections(value, self.directions_ke)
        orthogonal_dist_squared = torch.sum(torch.square(orthogonal_projections), dim=-1)

        # the orthogonal (F-1)-dimensional Gaussian component of the log likelihoods
        orthogonal_log_lks = (
            -((feature_dim - 1) / 2) * LOG2PI
            - (feature_dim - 1) * torch.log(self.orthogonal_stdev_k)
            - orthogonal_dist_squared / (2 * torch.square(self.orthogonal_stdev_k))
        )

        return orthogonal_log_lks + self.parallel_emg.log_prob(parallel_projections)


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

        self.log_cluster_weights_k = Parameter(torch.ones(self.num_artifact_clusters))
        parametrize.register_parametrization(self, "log_cluster_weights_k", LogWeights())

    def nonartifact_distribution(self) -> Distribution:
        """
        The distribution of nonartifact read embeddings, a Gaussian in F dimensions
        """
        return diagonal_gaussian(self.nonartifact_stdev_e)

    def outlier_distribution(self) -> Distribution:
        """
        Outliers are modeled by a Gaussian with same shape and twice as dispersed as the nonartifact Gaussian.
        Unsupervised loss tries to minimize probability assigned to this "pseudocluster", which is akin to
        assigning data to areas of high probability density.
        """
        return diagonal_gaussian(2 * self.nonartifact_stdev_e)

    def artifact_distribution(self) -> ArtifactClusterDistribution:
        """
        The distributions of artifact read embeddings in each artifact cluster, with batch shape (K,)
        """
        # the UnitVector parametrization makes the artifact directions unit vectors
        return ArtifactClusterDistribution(
            directions_ke=self.artifact_directions_ke,
            parallel_emg=self.artifact_emg.distribution(),
            orthogonal_stdev_k=self.artifact_stdev_k,
            validate_args=False,
        )

    def weighted_log_likelihoods_bk(self, alt_bre: RaggedSets, alt_counts_b: IntTensor):
        alt_re = alt_bre.flattened_tensor_nf

        nonartifact_log_lks_r = self.nonartifact_distribution().log_prob(alt_re)
        outlier_log_lks_r = self.outlier_distribution().log_prob(alt_re)

        # broadcast each read against the batch shape (K,) of artifact clusters
        artifact_log_lks_rk = self.artifact_distribution().log_prob(alt_re[:, None, :])

        nonartifact_log_lks_rk = nonartifact_log_lks_r[:, None]
        outlier_log_lks_rk = outlier_log_lks_r[:, None]

        nonartifact_log_lks_bk = RaggedSets(nonartifact_log_lks_rk, lengths_b=alt_counts_b).sums_over_sets()
        outlier_log_lks_bk = RaggedSets(outlier_log_lks_rk, lengths_b=alt_counts_b).sums_over_sets()
        unweighted_artifact_log_lks_bk = RaggedSets(artifact_log_lks_rk, lengths_b=alt_counts_b).sums_over_sets()

        log_artifact_cluster_weights_bk = self.log_cluster_weights_k[None, :]
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
        capped_logits_b = LOGIT_CAP * torch.tanh(logits_b / LOGIT_CAP)
        return capped_logits_b, log_lks_bk

    # avoid implicit forward calls because PyCharm doesn't recognize them
    def forward(self, features: Tensor):
        pass
