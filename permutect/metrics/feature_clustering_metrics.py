import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import Tensor
from torch.distributions import Chi2
from torch.distributions import Normal

from permutect.architecture.exponentially_modified_gaussian import ExponentiallyModifiedGaussianDistribution
from permutect.architecture.feature_clustering import FeatureClustering
from permutect.architecture.feature_clustering import component_name
from permutect.architecture.feature_clustering import parallel_and_orthogonal_projections
from permutect.sets.ragged_sets import RaggedSets

# the training data are shuffled, so the first reads of each component are a random sample of them
MAX_READS_PER_COMPONENT = 100000

# reads with less than this weight in a component are left out of its statistics
MIN_WEIGHT = 0.001
NUM_BINS = 60

# the first two categorical colors of a colorblind-safe palette
DATA_COLOR = "#2a78d6"
MODEL_COLOR = "#eb6834"


def scaled_chi_density(r: Tensor, degrees_of_freedom: int, scale: float) -> Tensor:
    """
    The density of R = scale * X, where X follows a chi distribution, i.e. (R / scale)^2 follows a chi-squared
    distribution, at positive r
    """
    chi_squared = Chi2(torch.tensor(float(degrees_of_freedom), dtype=r.dtype))
    return torch.exp(chi_squared.log_prob(torch.square(r / scale)) + torch.log(2 * r / scale**2))


class ComponentStatistics:
    """
    The statistics of reads in one component of the feature clustering model, each read with its weight
    """

    def __init__(self):
        self.statistics = []  # list (over batches) of tuples of tensors, each with one row per read
        self.weights = []  # list (over batches) of 1D tensors
        self.num_reads = 0

    def record(self, weights_r: Tensor, *statistics):
        # reads of non-negligible weight, as many as fit under the limit
        keep_r = weights_r > MIN_WEIGHT
        num_to_take = min(int(keep_r.sum()), MAX_READS_PER_COMPONENT - self.num_reads)
        self.num_reads += num_to_take
        self.weights.append(weights_r[keep_r][:num_to_take].cpu())
        self.statistics.append(tuple(statistic[keep_r][:num_to_take].cpu() for statistic in statistics))

    def get(self, statistic_index: int) -> Tensor:
        return torch.cat([statistics[statistic_index] for statistics in self.statistics])

    def get_weights(self) -> Tensor:
        return torch.cat(self.weights)


class FeatureClusteringFitMetrics:
    """
    One-dimensional statistics of the alt read embeddings of each component of the feature clustering model, whose
    distributions the model posits exactly, to compare with those distributions without the distortion of projecting
    embeddings to two or three dimensions:

    - nonartifact: the coordinates in units of their standard deviations, which follow a standard normal distribution,
      and the distance from the origin in those units, which follows a chi distribution with F degrees of freedom
    - artifact cluster k: the projection onto the cluster's direction, which follows the cluster's exponentially-modified
      Gaussian, and the distance from the cluster's axis, which follows a chi distribution with F - 1 degrees of freedom
      scaled by the cluster's orthogonal standard deviation

    Each read is weighted in each component by its variant's probability of coming from the component.  Assigning each
    variant to its most likely component would bias the statistics where components overlap, by leaving out of a
    component its reads that look more like another's.  With these weights, if the model is right, the weighted
    statistics of each component follow the model's distributions.
    """

    def __init__(self, num_artifact_clusters: int):
        self.num_artifact_clusters = num_artifact_clusters
        self.nonartifact = ComponentStatistics()  # reads in units of the standard deviations
        self.artifact_clusters = [ComponentStatistics() for _ in range(num_artifact_clusters)]  # parallel, orthogonal

    @torch.no_grad()
    def record_batch(self, clustering: FeatureClustering, alt_bre: RaggedSets, log_lks_bk: Tensor, artifact_probs_b):
        """
        :param log_lks_bk: the variants' weighted log likelihoods from clustering, whose columns are nonartifact,
            outlier, and the artifact clusters
        :param artifact_probs_b: the probability that each variant is an artifact, e.g. from its label, which the
            artifact clusters share according to their posterior probabilities
        """
        # outliers are a pseudocluster for the unsupervised loss, not a component that generates reads
        cluster_probs_bk = artifact_probs_b[:, None] * torch.softmax(log_lks_bk[:, 2:], dim=-1)
        reads_re = alt_bre.flattened_tensor_nf

        nonartifact_weights_r = alt_bre.expand_from_b_to_n(1 - artifact_probs_b)
        self.nonartifact.record(nonartifact_weights_r, reads_re / clustering.nonartifact_stdev_e)

        cluster_weights_rk = alt_bre.expand_from_b_to_n(cluster_probs_bk)
        for k, cluster in enumerate(self.artifact_clusters):
            parallel_r, orthogonal_re = parallel_and_orthogonal_projections(
                reads_re, clustering.artifact_directions_ke[k]
            )
            cluster.record(cluster_weights_rk[:, k], parallel_r, torch.norm(orthogonal_re, dim=-1))

    @torch.no_grad()
    def make_figure(self, clustering: FeatureClustering) -> Figure:
        """
        A row for the nonartifact component and for each artifact cluster, each with a histogram of the data and the
        model's density for each of the component's two statistics
        """
        feature_dim = clustering.feature_dim
        num_rows = 1 + self.num_artifact_clusters
        fig, axes = plt.subplots(num_rows, 2, squeeze=False, figsize=(10, 2.8 * num_rows))

        # the nonartifact component in the first row, and the artifact clusters in the others
        nonartifact_re, weights_r = self.nonartifact.get(0), self.nonartifact.get_weights()
        coordinate_weights = weights_r.repeat_interleave(feature_dim)
        normal = lambda x: torch.exp(Normal(0.0, 1.0).log_prob(x))
        title = "nonartifact: coordinate / stdev"
        plot_statistic(axes[0, 0], title, nonartifact_re.flatten(), coordinate_weights, normal)
        chi = lambda r: scaled_chi_density(r, degrees_of_freedom=feature_dim, scale=1.0)
        distances_r = torch.norm(nonartifact_re, dim=-1)
        title = "nonartifact: distance from origin / stdev"
        plot_statistic(axes[0, 1], title, distances_r, weights_r, chi, is_distance=True)

        emg_module = clustering.artifact_emg
        mu_k, sigma_k, lambda_k = emg_module.mu_k.tolist(), emg_module.sigma_k.tolist(), emg_module.lambda_k.tolist()
        stdev_k = clustering.artifact_stdev_k.tolist()
        for k, cluster in enumerate(self.artifact_clusters):
            name, weights_r = component_name(k + 2), cluster.get_weights()
            emg = ExponentiallyModifiedGaussianDistribution(*(torch.tensor(p[k]) for p in (mu_k, sigma_k, lambda_k)))
            emg_density = lambda x, emg=emg: torch.exp(emg.log_prob(x))
            plot_statistic(axes[k + 1, 0], f"{name}: projection onto direction", cluster.get(0), weights_r, emg_density)
            chi = lambda r, k=k: scaled_chi_density(r, degrees_of_freedom=feature_dim - 1, scale=stdev_k[k])
            title = f"{name}: distance from axis"
            plot_statistic(axes[k + 1, 1], title, cluster.get(1), weights_r, chi, is_distance=True)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / num_rows))
        return fig


def plot_statistic(ax, title: str, data: Tensor, weights: Tensor, density, is_distance: bool = False):
    """
    Plot a weighted histogram of the data and the model's density, a function of the statistic
    """
    data, weights = data.double().numpy(), weights.double().numpy()
    ax.set_title(f"{title} ({np.sum(weights):.0f} reads)", fontsize="medium")
    ax.set_ylabel("density")

    # the bulk of the data, or with no data, a typical range
    low, high = np.quantile(data, [0.002, 0.998]) if len(data) > 1 else (-4.0, 4.0)
    if is_distance:
        # a distance with one degree of freedom has an infinite density at zero
        low = max(low, 1e-6)
    high = max(high, low + 1.0)

    if len(data) > 0:
        ax.hist(data, bins=NUM_BINS, range=(low, high), weights=weights, density=True, color=DATA_COLOR, label="data")
    x = torch.linspace(low, high, 400)
    ax.plot(x.numpy(), density(x).numpy(), color=MODEL_COLOR, linewidth=2, label="model")
