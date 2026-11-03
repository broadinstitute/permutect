import numpy as np
import pytest
import torch
from matplotlib import pyplot as plt
from scipy import stats

from permutect.architecture.feature_clustering import FeatureClustering
from permutect.metrics import feature_clustering_metrics
from permutect.metrics.feature_clustering_metrics import FeatureClusteringFitMetrics
from permutect.metrics.feature_clustering_metrics import scaled_chi_density
from permutect.sets.ragged_sets import RaggedSets
from permutect.test.architecture.test_exponentially_modified_gaussian import scipy_emg
from permutect.test.architecture.test_feature_clustering import make_random_clustering


@pytest.mark.parametrize("degrees_of_freedom", [1, 3, 10])
@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_scaled_chi_density_matches_scipy(degrees_of_freedom, scale):
    r = np.linspace(0.01, 8 * scale, 200)
    actual = scaled_chi_density(torch.from_numpy(r), degrees_of_freedom, scale).numpy()
    np.testing.assert_allclose(actual, stats.chi(df=degrees_of_freedom, scale=scale).pdf(r), rtol=1e-8)


def test_reads_are_weighted_by_their_variants_component_probabilities():
    torch.manual_seed(0)
    clustering = FeatureClustering(feature_dimension=3, num_artifact_clusters=2).to(torch.float64)
    lengths_b = torch.tensor([1, 2, 3])
    reads_re = torch.randn(6, 3, dtype=torch.float64)

    # columns are nonartifact, outlier, and the two artifact clusters.  The outlier pseudocluster doesn't count, and
    # each artifact's probability is shared by the clusters in proportion to their likelihoods.
    log_lks_bk = torch.log(torch.tensor([[0.5, 0.9, 0.3, 0.1], [0.1, 0.5, 0.2, 0.6], [0.2, 0.1, 0.7, 0.7]]))
    artifact_probs_b = torch.tensor([0.0, 1.0, 0.5])
    metrics = FeatureClusteringFitMetrics(num_artifact_clusters=2)
    metrics.record_batch(clustering, RaggedSets(reads_re, lengths_b), log_lks_bk.double(), artifact_probs_b.double())

    # the second variant is an artifact, so it has no nonartifact weight and its reads are left out
    nonartifact_weights = torch.tensor([1.0, 0.5, 0.5, 0.5], dtype=torch.float64)
    torch.testing.assert_close(metrics.nonartifact.get_weights(), nonartifact_weights)
    stdev_e = clustering.nonartifact_stdev_e
    torch.testing.assert_close(metrics.nonartifact.get(0), torch.vstack((reads_re[:1], reads_re[3:])) / stdev_e)

    # the first variant isn't an artifact, so its read is left out of the artifact clusters
    cluster_weights_rk = torch.tensor([[0.25, 0.75]] * 2 + [[0.25, 0.25]] * 3, dtype=torch.float64)
    directions_ke = clustering.artifact_directions_ke
    for k, cluster in enumerate(metrics.artifact_clusters):
        torch.testing.assert_close(cluster.get_weights(), cluster_weights_rk[:, k])
        parallel_r = reads_re[1:] @ directions_ke[k]
        orthogonal_r = torch.norm(reads_re[1:] - parallel_r[:, None] * directions_ke[k], dim=-1)
        torch.testing.assert_close(cluster.get(0), parallel_r)
        torch.testing.assert_close(cluster.get(1), orthogonal_r)


def weighted_ks_statistic(data: np.ndarray, weights: np.ndarray, cdf) -> tuple[float, float]:
    """
    The Kolmogorov-Smirnov statistic of the weighted empirical distribution of the data against cdf, and the
    critical value for a significance level of 0.001 given the effective number of data
    """
    order = np.argsort(data)
    sorted_data, sorted_weights = data[order], weights[order]
    cumulative = np.cumsum(sorted_weights) / np.sum(sorted_weights)
    model_cdf = cdf(sorted_data)
    statistic = max(
        np.max(np.abs(cumulative - model_cdf)),
        np.max(np.abs(cumulative - sorted_weights / np.sum(sorted_weights) - model_cdf)),
    )
    effective_num = np.sum(weights) ** 2 / np.sum(np.square(weights))
    return statistic, 1.95 / np.sqrt(effective_num)


def test_weighted_statistics_of_reads_from_the_model_follow_the_model():
    # overlapping clusters, where assigning each variant to its most likely component would bias the statistics
    torch.manual_seed(0)
    feature_dim, num_clusters, num_variants, alt_count = 4, 3, 12000, 2
    clustering = make_random_clustering(feature_dim, num_clusters)

    # labeled variants, half artifacts, from the clusters in proportion to their weights
    artifact_probs_b = torch.tensor([0.0, 1.0], dtype=torch.float64).repeat(num_variants // 2)
    components_b = clustering.sample_components_b(artifact_probs_b)
    components_r = components_b.repeat_interleave(alt_count)
    reads_re = torch.zeros(len(components_r), feature_dim, dtype=torch.float64)
    with torch.no_grad():
        reads_re[components_r == 0] = clustering.nonartifact_distribution().sample((int(torch.sum(components_r == 0)),))
        for k in range(num_clusters):
            is_cluster_r = components_r == k + 2
            reads_re[is_cluster_r] = clustering.artifact_distribution(k).sample((int(torch.sum(is_cluster_r)),))
        lengths_b = torch.full((num_variants,), alt_count)
        alt_bre = RaggedSets(reads_re, lengths_b)
        log_lks_bk = clustering.weighted_log_likelihoods_bk(alt_bre, lengths_b)

    metrics = FeatureClusteringFitMetrics(num_artifact_clusters=num_clusters)
    metrics.record_batch(clustering, alt_bre, log_lks_bk, artifact_probs_b)

    nonartifact_re, weights_r = metrics.nonartifact.get(0).numpy(), metrics.nonartifact.get_weights().numpy()
    statistic, critical = weighted_ks_statistic(nonartifact_re[:, 0], weights_r, stats.norm.cdf)
    assert statistic < critical
    statistic, critical = weighted_ks_statistic(
        np.linalg.norm(nonartifact_re, axis=-1), weights_r, stats.chi(df=feature_dim).cdf
    )
    assert statistic < critical

    emg = clustering.artifact_emg
    for k, cluster in enumerate(metrics.artifact_clusters):
        weights_r = cluster.get_weights().numpy()
        emg_k = scipy_emg(emg.mu_k[k].item(), emg.sigma_k[k].item(), emg.lambda_k[k].item())
        statistic, critical = weighted_ks_statistic(cluster.get(0).numpy(), weights_r, emg_k.cdf)
        assert statistic < critical, k
        chi_k = stats.chi(df=feature_dim - 1, scale=clustering.artifact_stdev_k[k].item())
        statistic, critical = weighted_ks_statistic(cluster.get(1).numpy(), weights_r, chi_k.cdf)
        assert statistic < critical, k


def test_reads_per_component_are_limited(monkeypatch):
    monkeypatch.setattr(feature_clustering_metrics, "MAX_READS_PER_COMPONENT", 5)
    clustering = FeatureClustering(feature_dimension=3, num_artifact_clusters=2)
    metrics = FeatureClusteringFitMetrics(num_artifact_clusters=2)
    lengths_b = torch.tensor([2, 2])
    log_lks_bk = torch.zeros(2, 4)
    for _ in range(3):
        metrics.record_batch(clustering, RaggedSets(torch.randn(4, 3), lengths_b), log_lks_bk, torch.zeros(2))
    assert metrics.nonartifact.num_reads == 5
    assert len(metrics.nonartifact.get(0)) == 5
    assert all(cluster.num_reads == 0 for cluster in metrics.artifact_clusters)


def test_figure_has_the_data_and_the_model_for_each_statistic():
    torch.manual_seed(0)
    clustering = make_random_clustering(feature_dim=4, num_clusters=2)
    metrics = FeatureClusteringFitMetrics(num_artifact_clusters=2)

    # reads only from nonartifact variants and from variants that are surely in artifact cluster 0
    lengths_b = torch.tensor([3, 3])
    with torch.no_grad():
        reads_re = torch.vstack(
            (clustering.nonartifact_distribution().sample((3,)), clustering.artifact_distribution(0).sample((3,)))
        )
    log_lks_bk = torch.tensor([[0.0, 0.0, 0.0, -100.0]] * 2, dtype=torch.float64)
    alt_bre = RaggedSets(reads_re, lengths_b)
    metrics.record_batch(clustering, alt_bre, log_lks_bk, torch.tensor([0.0, 1.0], dtype=torch.float64))

    figure = metrics.make_figure(clustering)
    axes = figure.get_axes()
    assert len(axes) == 6
    for n, ax in enumerate(axes):
        assert [line.get_label() for line in ax.get_lines()] == ["model"]
        assert np.all(np.isfinite(ax.get_lines()[0].get_ydata()))
        # artifact cluster 1, in the last row, has no reads
        has_histogram = len(ax.patches) > 0
        assert has_histogram == (n < 4), ax.get_title()
    plt.close(figure)
