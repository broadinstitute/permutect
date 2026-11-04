import enum
import math

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Buffer
from torch.utils.tensorboard import SummaryWriter

from permutect.data.batch import Batch
from permutect.data.batch import BatchIndexedTensor
from permutect.metrics import plotting
from permutect.misc_utils import all_reduce_sum_
from permutect.misc_utils import distributed_world_size
from permutect.utils.enums import Label
from permutect.utils.enums import Variation


class PlotType(enum.Enum):
    COUNTS = "counts"
    WEIGHTS = "weights"


def balancing_weights(counts_slvra: Tensor) -> BatchIndexedTensor:
    """
    weights that give artifacts and variants each half of the total weight of the data in each source, variant type, and
    ref and alt count bin, given their counts there: a label with count n in a bin where the other has count m gets
    weight (1 + m/n)/2, so that its total weight is (n + m)/2, up to the clipping.  The pseudocounts give both labels
    weight 1 in an empty bin.  Only artifacts and variants are weighted, so the weights of other labels are 0.
    """
    ratio_svra = (counts_slvra[:, Label.ARTIFACT] + 0.01) / (counts_slvra[:, Label.VARIANT] + 0.01)
    weights_slvra = torch.zeros_like(counts_slvra)
    weights_slvra[:, Label.ARTIFACT] = torch.clip((1 + 1 / ratio_svra) / 2, min=0.01, max=100)
    weights_slvra[:, Label.VARIANT] = torch.clip((1 + ratio_svra) / 2, min=0.01, max=100)
    return BatchIndexedTensor(weights_slvra)


class Balancer(Module):
    # the factor by which the counts of unlabeled data decay with each datum seen, so that they reflect about the last
    # 100,000 data
    ATTENUATION_PER_DATUM = 0.99999

    def __init__(self, num_sources: int, device, counts_slvra: Tensor = None):
        """
        :param counts_slvra: numbers of data by source, label, variant type, and ref and alt count bin, from which the
            weights of labeled data come, eg the expected counts of the training data after downsampling.  If None, all
            labeled data have weight 1.
        """
        super(Balancer, self).__init__()
        self.device = device
        self.num_sources = num_sources

        # not weighted, just the counts of data, which are known before training
        self.counts_slvra = Buffer(
            BatchIndexedTensor.zeros(num_sources=num_sources)
            if counts_slvra is None
            else BatchIndexedTensor(counts_slvra.detach().clone())
        )

        # same, but here unlabeled data are probabilistically assigned to artifact or non-artifact based on
        # the model output.  This lets us balance training when, for example, the preponderance of unlabeled
        # data are non-artifacts.  (Very often unlabeled data are sequencing errors, which are not artifacts.
        # also, in test-time adaptation of somatic calls most data are germline variants, which are not artifacts).
        # These are counted as the model sees the data, and old counts decay, so that the weights follow the model as it
        # learns.
        self.pseudo_counts_slvra = Buffer(BatchIndexedTensor.zeros(num_sources=num_sources))

        # weights for labeled data
        self.weights_slvra = Buffer(balancing_weights(self.counts_slvra))

        # weights for unlabeled data, where index l is a guess for the correct label
        self.unlabeled_weights_slvra = Buffer(balancing_weights(self.pseudo_counts_slvra))
        self.to(device=device)

    def process_batch_and_compute_weights(self, batch: Batch, artifact_probs_b: Tensor):
        # this updates the counts of unlabeled data, recomputes their weights, and returns the weights
        # In distributed training this batch's counts are summed over processes, so that every process keeps the same
        # counts and weights.  Every process has a batch of the same size at each step.
        new_pseudo_counts_slvra = torch.zeros_like(self.pseudo_counts_slvra)

        # increment the pseudo-label counts for unlabeled data according to the amount of probability
        # assigned to artifact/nonartifact
        idx = batch.batch_indices()
        art_probs_b = artifact_probs_b.to(device=self.device)
        unlabeled_mask, artifact_labels, nonartifact_labels = self.pseudolabels(batch)
        idx.increment_tensor(new_pseudo_counts_slvra, labels=artifact_labels, values=unlabeled_mask * art_probs_b)
        idx.increment_tensor(
            new_pseudo_counts_slvra, labels=nonartifact_labels, values=unlabeled_mask * (1 - art_probs_b)
        )

        attenuation = math.pow(Balancer.ATTENUATION_PER_DATUM, batch.size() * distributed_world_size())
        self.pseudo_counts_slvra.mul_(attenuation).add_(all_reduce_sum_(new_pseudo_counts_slvra))
        self.unlabeled_weights_slvra.copy_(balancing_weights(self.pseudo_counts_slvra))

        return self.compute_weights(batch, artifact_probs_b)

    def compute_weights(self, batch: Batch, artifact_probs_b: Tensor):
        # the weights from the current counts, without updating them, eg for validation data
        idx = batch.batch_indices()
        art_probs_b = artifact_probs_b.to(device=self.device)
        unlabeled_mask, artifact_labels, nonartifact_labels = self.pseudolabels(batch)

        labeled_weights_b = idx.index_into_tensor(self.weights_slvra)
        pseudo_art_weights_b = idx.index_into_tensor(self.unlabeled_weights_slvra, labels=artifact_labels)
        pseudo_nonart_weights_b = idx.index_into_tensor(self.unlabeled_weights_slvra, labels=nonartifact_labels)
        unlabeled_weights_b = art_probs_b * pseudo_art_weights_b + (1 - art_probs_b) * pseudo_nonart_weights_b
        weights_b = unlabeled_mask * unlabeled_weights_b + (1 - unlabeled_mask) * labeled_weights_b

        return weights_b

    def pseudolabels(self, batch: Batch) -> tuple[Tensor, Tensor, Tensor]:
        """
        the mask of the batch's unlabeled data, and artifact and non-artifact labels for every datum.  The labels are
        made on the device: copying them from the CPU would make the CPU wait for the GPU, eg for the model's forward
        pass, before queueing more work.
        """
        unlabeled_mask = 1 - batch.get_is_labeled_mask()
        artifact_labels = torch.full((batch.size(),), Label.ARTIFACT, device=self.device)
        nonartifact_labels = torch.full((batch.size(),), Label.VARIANT, device=self.device)
        return unlabeled_mask, artifact_labels, nonartifact_labels

    def plot_weights(self, weights_slvra: Tensor, label: Label, var_type: Variation, axis, source: int):
        vmin, vmax = -4, 4
        plot_data_ra = torch.clip(torch.log(weights_slvra[source, label, var_type].cpu()), min=vmin, max=vmax)
        return plotting.count_bins_color_plot_on_axis(axis, plot_data_ra, vmin=vmin, vmax=vmax)

    def make_plots(self, summary_writer: SummaryWriter, prefix: str, epoch: int, plot_type: PlotType):
        for source in range(self.num_sources):
            source_suffix = "" if self.num_sources == 1 else f", source {source}"
            if plot_type == PlotType.COUNTS:
                fig = plotting.label_and_variant_type_figure(
                    lambda label, var_type, axis: plotting.count_plot_on_axis(
                        axis, self.counts_slvra[source, :, var_type].cpu(), label
                    )
                )
                summary_writer.add_figure(prefix + source_suffix, fig, global_step=epoch)
            elif plot_type == PlotType.WEIGHTS:
                # only artifacts and variants are weighted, so the weights of unlabeled data are by pseudo-label
                for weights_slvra, data_suffix in (
                    (self.weights_slvra, ""),
                    (self.unlabeled_weights_slvra, ", unlabeled data by pseudo-label"),
                ):
                    fig = plotting.label_and_variant_type_figure(
                        lambda label, var_type, axis: self.plot_weights(weights_slvra, label, var_type, axis, source),
                        labels=(Label.ARTIFACT, Label.VARIANT),
                    )
                    summary_writer.add_figure(prefix + data_suffix + source_suffix, fig, global_step=epoch)
            else:
                raise ValueError(f"Unknown type_of_plot: {plot_type.name}.")
