import numpy as np
import pytest
import torch
from scipy import stats

from permutect.architecture.exponentially_modified_gaussian import STDEV_CONSTRAINT
from permutect.architecture.exponentially_modified_gaussian import ExponentiallyModifiedGaussianDistribution
from permutect.architecture.feature_clustering import ArtifactClusterDistribution
from permutect.architecture.feature_clustering import FeatureClustering
from permutect.architecture.feature_clustering import component_name
from permutect.sets.ragged_sets import RaggedSets
from permutect.test.architecture.test_exponentially_modified_gaussian import make_emg
from permutect.test.architecture.test_exponentially_modified_gaussian import scipy_emg

# (mu, sigma, lambda) of the parallel EMG and the orthogonal standard deviation of each of three clusters
CLUSTER_PARAMETERS = [((2.0, 1.0, 1.0), 0.5), ((0.5, 0.3, 3.0), 1.0), ((-1.0, 2.0, 0.5), 2.0)]


def test_standard_deviations_share_one_constraint():
    # the Gaussian and exponentially-modified Gaussian standard deviations used to have separate, identical bounds
    clustering = FeatureClustering(feature_dimension=3, num_artifact_clusters=2)
    stdevs = [
        (clustering, "nonartifact_stdev_e"),
        (clustering, "artifact_stdev_k"),
        (clustering.artifact_emg, "sigma_k"),
    ]
    for module, name in stdevs:
        assert module.parametrizations[name][0] is STDEV_CONSTRAINT, name


def random_unit_vectors(num_vectors: int, dimension: int) -> torch.Tensor:
    vectors = torch.randn(num_vectors, dimension, dtype=torch.float64)
    return vectors / torch.norm(vectors, dim=-1, keepdim=True)


def make_artifact_clusters(directions_ke: torch.Tensor, validate_args=None) -> ArtifactClusterDistribution:
    emg_parameters, orthogonal_stdevs = zip(*CLUSTER_PARAMETERS)
    return ArtifactClusterDistribution(
        directions_ke=directions_ke,
        parallel_emg=make_emg(list(emg_parameters)),
        orthogonal_stdev_k=torch.tensor(orthogonal_stdevs, dtype=torch.float64),
        validate_args=validate_args,
    )


def scipy_artifact_cluster_log_prob(x_ne, direction_e, emg_parameters, orthogonal_stdev: float) -> np.ndarray:
    # coordinates in an orthonormal basis whose first vector is the cluster's direction
    feature_dim = len(direction_e)
    other_vectors = np.random.default_rng(0).standard_normal((feature_dim, feature_dim - 1))
    basis_ee, _ = np.linalg.qr(np.column_stack((direction_e, other_vectors)))
    coordinates_ne = x_ne @ basis_ee

    # QR determines the first basis vector only up to its sign
    parallel_n = coordinates_ne[:, 0] * np.sign(basis_ee[:, 0] @ direction_e)
    orthogonal = stats.multivariate_normal(mean=np.zeros(feature_dim - 1), cov=orthogonal_stdev**2)
    return scipy_emg(*emg_parameters).logpdf(parallel_n) + orthogonal.logpdf(coordinates_ne[:, 1:])


def test_artifact_cluster_log_prob_matches_scipy():
    torch.manual_seed(0)
    feature_dim = 4
    clusters = make_artifact_clusters(random_unit_vectors(len(CLUSTER_PARAMETERS), feature_dim))

    # points scattered broadly, and points typical of each cluster
    x_ne = torch.vstack((3 * torch.randn(100, feature_dim, dtype=torch.float64), clusters.sample((30,)).flatten(0, 1)))
    actual_nk = clusters.log_prob(x_ne[:, None, :]).numpy()

    for k, (emg_parameters, orthogonal_stdev) in enumerate(CLUSTER_PARAMETERS):
        direction_e = clusters.directions_ke[k].numpy()
        expected_n = scipy_artifact_cluster_log_prob(x_ne.numpy(), direction_e, emg_parameters, orthogonal_stdev)
        np.testing.assert_allclose(actual_nk[:, k], expected_n, rtol=1e-8)


def test_artifact_cluster_density_integrates_to_one():
    direction_e = torch.tensor([0.6, 0.8], dtype=torch.float64)
    cluster = ArtifactClusterDistribution(
        direction_e, make_emg((1.0, 0.5, 2.0)), torch.tensor(0.7, dtype=torch.float64)
    )

    grid = torch.linspace(-8, 12, 1001, dtype=torch.float64)
    spacing = grid[1] - grid[0]
    x_ije = torch.stack(torch.meshgrid(grid, grid, indexing="ij"), dim=-1)
    assert torch.sum(torch.exp(cluster.log_prob(x_ije))) * spacing**2 == pytest.approx(1.0, abs=1e-6)


def test_artifact_cluster_log_prob_gradients():
    # gradients with respect to the parameters train the model, and gradients with respect to x train the embeddings
    torch.manual_seed(0)
    feature_dim = 3
    x_ne = 2 * torch.randn(10, 1, feature_dim, dtype=torch.float64)
    directions_ke = random_unit_vectors(2, feature_dim)
    mu_k, sigma_k, lambda_k = torch.tensor([(2.0, 1.0, 1.0), (0.5, 0.3, 3.0)], dtype=torch.float64).unbind(dim=-1)
    orthogonal_stdev_k = torch.tensor([0.5, 1.0], dtype=torch.float64)
    inputs = (x_ne, directions_ke, mu_k, sigma_k, lambda_k, orthogonal_stdev_k)

    def log_prob(x_ne, directions_ke, mu_k, sigma_k, lambda_k, orthogonal_stdev_k):
        emg = ExponentiallyModifiedGaussianDistribution(mu_k, sigma_k, lambda_k)
        return ArtifactClusterDistribution(directions_ke, emg, orthogonal_stdev_k).log_prob(x_ne)

    assert torch.autograd.gradcheck(log_prob, tuple(tensor.requires_grad_() for tensor in inputs))


def test_artifact_cluster_samples_follow_distribution():
    torch.manual_seed(0)
    feature_dim, num_samples = 5, 50000
    clusters = make_artifact_clusters(random_unit_vectors(len(CLUSTER_PARAMETERS), feature_dim))
    samples_nke = clusters.sample((num_samples,)).numpy()
    assert samples_nke.shape == (num_samples, len(CLUSTER_PARAMETERS), feature_dim)

    for k, (emg_parameters, orthogonal_stdev) in enumerate(CLUSTER_PARAMETERS):
        samples_ne = samples_nke[:, k, :]
        direction_e = clusters.directions_ke[k].numpy()
        parallel_n = samples_ne @ direction_e
        orthogonal_ne = samples_ne - parallel_n[:, None] * direction_e[None, :]

        # parallel projections follow the EMG, and squared orthogonal distances in units of the orthogonal standard
        # deviation follow a chi-squared distribution with F - 1 degrees of freedom
        assert stats.kstest(parallel_n, scipy_emg(*emg_parameters).cdf).pvalue > 0.001
        orthogonal_chi_squared_n = np.sum(np.square(orthogonal_ne), axis=-1) / orthogonal_stdev**2
        assert stats.kstest(orthogonal_chi_squared_n, stats.chi2(df=feature_dim - 1).cdf).pvalue > 0.001

        # the covariance sigma^2 (I - uu^T) + Var(x.u) uu^T is isotropic in the orthogonal directions and the parallel
        # and orthogonal parts are uncorrelated
        outer_ee = np.outer(direction_e, direction_e)
        covariance_ee = (
            orthogonal_stdev**2 * (np.eye(feature_dim) - outer_ee) + scipy_emg(*emg_parameters).var() * outer_ee
        )
        tolerance = 5 * np.max(np.diag(covariance_ee)) * np.sqrt(2 / num_samples)
        np.testing.assert_allclose(np.cov(samples_ne.T), covariance_ee, atol=tolerance)
        np.testing.assert_allclose(clusters.variance[k].numpy(), np.diag(covariance_ee))

        mean_tolerance = 5 * np.sqrt(np.max(np.diag(covariance_ee)) / num_samples)
        np.testing.assert_allclose(np.mean(samples_ne, axis=0), clusters.mean[k].numpy(), atol=mean_tolerance)
        np.testing.assert_allclose(clusters.mean[k].numpy(), scipy_emg(*emg_parameters).mean() * direction_e)


def test_artifact_cluster_shapes_and_broadcasting():
    torch.manual_seed(0)
    clusters = make_artifact_clusters(random_unit_vectors(3, 4))
    assert clusters.batch_shape == (3,)
    assert clusters.event_shape == (4,)
    assert clusters.sample((5,)).shape == (5, 3, 4)

    # log likelihoods of each of 7 values under each of the 3 clusters, as FeatureClustering needs
    assert clusters.log_prob(torch.zeros(7, 1, 4, dtype=torch.float64)).shape == (7, 3)

    expanded = clusters.expand((2, 3))
    assert expanded.batch_shape == (2, 3)
    assert expanded.sample().shape == (2, 3, 4)
    x_e = torch.ones(4, dtype=torch.float64)
    assert torch.equal(expanded.log_prob(x_e)[1], clusters.log_prob(x_e))

    # one direction and standard deviation broadcast against a batch of EMGs
    one_direction = ArtifactClusterDistribution(clusters.directions_ke[0], clusters.parallel_emg, torch.tensor(1.0))
    assert one_direction.batch_shape == (3,)
    assert one_direction.sample().shape == (3, 4)


def test_artifact_cluster_validation():
    torch.manual_seed(0)
    clusters = make_artifact_clusters(random_unit_vectors(3, 4), validate_args=True)
    with pytest.raises(ValueError):
        clusters.log_prob(torch.zeros(7, 1, 5, dtype=torch.float64))
    with pytest.raises(ValueError):
        ArtifactClusterDistribution(clusters.directions_ke, clusters.parallel_emg, -torch.ones(3), validate_args=True)


def test_component_distributions():
    clustering = FeatureClustering(feature_dimension=4, num_artifact_clusters=3)
    assert clustering.nonartifact_distribution().sample((10,)).shape == (10, 4)
    assert clustering.outlier_distribution().sample((10,)).shape == (10, 4)
    assert clustering.artifact_distribution().sample((10,)).shape == (10, 3, 4)

    # outliers are twice as dispersed as nonartifacts
    nonartifact_stdev_e = clustering.nonartifact_distribution().stddev
    assert torch.equal(nonartifact_stdev_e, clustering.nonartifact_stdev_e)
    assert torch.equal(clustering.outlier_distribution().stddev, 2 * nonartifact_stdev_e)

    # the distributions of chosen clusters, e.g. of reads from different clusters
    clusters_r = torch.tensor([2, 0, 2, 1, 2])
    chosen = clustering.artifact_distribution(clusters_r)
    assert chosen.batch_shape == (5,)
    assert chosen.sample().shape == (5, 4)
    assert torch.equal(chosen.directions_ke, clustering.artifact_directions_ke[clusters_r])
    assert torch.equal(chosen.parallel_emg.lambda_, clustering.artifact_emg.lambda_k[clusters_r])
    assert torch.equal(chosen.orthogonal_stdev_k, clustering.artifact_stdev_k[clusters_r])


def make_random_clustering(feature_dim: int, num_clusters: int) -> FeatureClustering:
    # parameters moved away from their initial values, like those of a trained model
    clustering = FeatureClustering(feature_dimension=feature_dim, num_artifact_clusters=num_clusters).to(torch.float64)
    with torch.no_grad():
        for parameter in clustering.parameters():
            parameter.add_(0.5 * torch.randn_like(parameter))
    return clustering


def test_weighted_log_likelihoods_match_scipy():
    torch.manual_seed(0)
    feature_dim, num_clusters = 4, 3
    clustering = make_random_clustering(feature_dim, num_clusters)

    lengths_b = torch.tensor([2, 1, 4])
    reads_re = 2 * torch.randn(int(lengths_b.sum()), feature_dim, dtype=torch.float64)
    actual_bk = clustering.weighted_log_likelihoods_bk(RaggedSets(reads_re, lengths_b), lengths_b).detach().numpy()

    x_re = reads_re.numpy()
    nonartifact_stdev_e = clustering.nonartifact_stdev_e.detach().numpy()
    emg = clustering.artifact_emg
    emg_parameters_k = zip(*(tensor.detach().numpy() for tensor in (emg.mu_k, emg.sigma_k, emg.lambda_k)))
    directions_ke = clustering.artifact_directions_ke.detach().numpy()
    artifact_stdev_k = clustering.artifact_stdev_k.detach().numpy()

    # columns are nonartifact, outlier, and the artifact clusters
    columns = [
        stats.norm(scale=nonartifact_stdev_e).logpdf(x_re).sum(axis=-1),
        stats.norm(scale=2 * nonartifact_stdev_e).logpdf(x_re).sum(axis=-1),
    ] + [
        scipy_artifact_cluster_log_prob(x_re, direction_e, emg_parameters, stdev)
        for direction_e, emg_parameters, stdev in zip(directions_ke, emg_parameters_k, artifact_stdev_k)
    ]
    log_lks_rk = np.column_stack(columns)

    # reads' log likelihoods add within each set, and each artifact cluster's log weight is added once per set
    bounds = np.cumsum([0] + lengths_b.tolist())
    expected_bk = np.array([log_lks_rk[start:end].sum(axis=0) for start, end in zip(bounds[:-1], bounds[1:])])
    expected_bk[:, 2:] += clustering.log_cluster_weights_k.detach().numpy()
    np.testing.assert_allclose(actual_bk, expected_bk, rtol=1e-8)


def test_sample_mean_embeddings_follow_the_model():
    torch.manual_seed(0)
    feature_dim, num_clusters, num_per_group = 5, 2, 5000
    clustering = make_random_clustering(feature_dim, num_clusters)
    alt_counts = (1, 4, 16)

    # every combination of component (nonartifact, outlier, and each artifact cluster) and alt count, interleaved
    num_components = 2 + num_clusters
    components_b = torch.arange(num_components).repeat(len(alt_counts) * num_per_group)
    alt_counts_b = torch.tensor(alt_counts).repeat_interleave(num_components).repeat(num_per_group)
    means_be = clustering.sample_mean_embeddings_be(components_b, alt_counts_b)
    assert means_be.shape == (len(components_b), feature_dim)
    assert not means_be.requires_grad

    nonartifact_stdev_e = clustering.nonartifact_stdev_e.detach().numpy()
    emg = clustering.artifact_emg
    emg_parameters_k = list(zip(*(tensor.detach().numpy() for tensor in (emg.mu_k, emg.sigma_k, emg.lambda_k))))
    directions_ke = clustering.artifact_directions_ke.detach().numpy()
    artifact_stdev_k = clustering.artifact_stdev_k.detach().numpy()

    means_of = lambda component, alt_count: means_be[(components_b == component) & (alt_counts_b == alt_count)].numpy()

    for alt_count in alt_counts:
        # the mean of n reads from a Gaussian is Gaussian with standard deviations divided by sqrt(n)
        for component, stdev_e in ((0, nonartifact_stdev_e), (1, 2 * nonartifact_stdev_e)):
            standardized_ne = means_of(component, alt_count) * np.sqrt(alt_count) / stdev_e
            assert stats.kstest(standardized_ne.ravel(), stats.norm.cdf).pvalue > 0.001

        for k in range(num_clusters):
            means_ne = means_of(2 + k, alt_count)
            parallel_n = means_ne @ directions_ke[k]
            orthogonal_ne = means_ne - parallel_n[:, None] * directions_ke[k][None, :]

            # the parallel part of the mean of n reads has the EMG's mean and its variance divided by n, and is the EMG
            # itself for one read
            emg_k = scipy_emg(*emg_parameters_k[k])
            assert np.mean(parallel_n) == pytest.approx(emg_k.mean(), abs=5 * emg_k.std() / np.sqrt(num_per_group))
            assert np.var(parallel_n) == pytest.approx(emg_k.var() / alt_count, rel=0.1)
            if alt_count == 1:
                assert stats.kstest(parallel_n, emg_k.cdf).pvalue > 0.001

            # the orthogonal part is an isotropic Gaussian with variance divided by n
            orthogonal_chi_squared_n = np.sum(np.square(orthogonal_ne), axis=-1) * alt_count / artifact_stdev_k[k] ** 2
            assert stats.kstest(orthogonal_chi_squared_n, stats.chi2(df=feature_dim - 1).cdf).pvalue > 0.001


def test_sample_components():
    torch.manual_seed(0)
    num_clusters, num_variants = 3, 20000
    clustering = make_random_clustering(feature_dim=3, num_clusters=num_clusters)

    # variants that aren't artifacts are nonartifact, and artifacts' clusters are drawn according to the cluster weights
    nonartifact_b = clustering.sample_components_b(torch.zeros(num_variants, dtype=torch.float64))
    assert torch.equal(nonartifact_b, torch.zeros(num_variants, dtype=torch.long))
    artifact_b = clustering.sample_components_b(torch.ones(num_variants, dtype=torch.float64))
    cluster_fractions_k = torch.bincount(artifact_b - 2, minlength=num_clusters).double() / num_variants
    torch.testing.assert_close(cluster_fractions_k, torch.exp(clustering.log_cluster_weights_k), atol=0.02, rtol=0)

    half_b = clustering.sample_components_b(torch.full((num_variants,), 0.5, dtype=torch.float64))
    assert set(half_b.tolist()) == {0} | {2 + k for k in range(num_clusters)}
    assert torch.mean((half_b >= 2).float()).item() == pytest.approx(0.5, abs=0.02)

    assert [component_name(component) for component in range(4)] == [
        "nonartifact",
        "outlier",
        "artifact cluster 0",
        "artifact cluster 1",
    ]


def test_sample_mean_embeddings_of_one_component():
    # with no reads from the other components
    clustering = FeatureClustering(feature_dimension=3, num_artifact_clusters=2)
    for component in range(4):
        means_be = clustering.sample_mean_embeddings_be(torch.full((6,), component), torch.tensor([1, 2, 3, 1, 2, 3]))
        assert means_be.shape == (6, 3)
        assert torch.isfinite(means_be).all()
