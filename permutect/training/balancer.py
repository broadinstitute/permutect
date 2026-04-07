import enum
import itertools
import math

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn import Module
from torch.nn import Parameter
from torch.utils.tensorboard import SummaryWriter

from permutect.data.batch import Batch
from permutect.data.batch import BatchIndexedTensor
from permutect.data.count_binning import ALT_COUNT_BIN_BOUNDS
from permutect.data.count_binning import REF_COUNT_BIN_BOUNDS
from permutect.metrics import plotting
from permutect.utils.enums import Label
from permutect.utils.enums import Variation


class PlotType(enum.Enum):
    COUNTS = "counts"
    WEIGHTS = "weights"

class Balancer(Module):
    ATTENUATION_PER_DATUM = 0.99999
    DATA_BEFORE_RECOMPUTE = 10000

    def __init__(self, num_sources: int, device):
        super(Balancer, self).__init__()
        self.device = device
        self.num_sources = num_sources
        self.count_since_last_recomputation = 0

        # not weighted, just the actual counts of data seen
        self.counts_slvra = Parameter(BatchIndexedTensor.zeros(num_sources=num_sources), requires_grad=False)

        # initialize weights to be flat
        self.weights_slvra = Parameter(BatchIndexedTensor.ones(num_sources=num_sources), requires_grad=False)

        # the overall weights for adversarial source prediction are the regular weights times the source weights
        self.source_weights_s = Parameter(torch.ones(num_sources), requires_grad=False)

        self.to(device=device)

    def process_batch_and_compute_weights(self, batch: Batch):
        # this updates the counts that are used to compute weights, recomputes the weights, and returns the weights
        # increment counts by 1
        batch.batch_indices().increment_tensor(self.counts_slvra, values=torch.ones(batch.size(), device=self.device))
        self.count_since_last_recomputation += batch.size()

        if self.count_since_last_recomputation > Balancer.DATA_BEFORE_RECOMPUTE:
            art_to_nonart_ratios_svra = (self.counts_slvra[:, Label.ARTIFACT] + 0.01) / (
                self.counts_slvra[:, Label.VARIANT] + 0.01
            )
            # TODO: perhaps don't recompute weights at every batch, as we do here
            new_weights_slvra = torch.zeros_like(self.weights_slvra)
            new_weights_slvra[:, Label.ARTIFACT] = torch.clip(
                (1 + 1 / art_to_nonart_ratios_svra) / 2, min=0.01, max=100
            )
            new_weights_slvra[:, Label.VARIANT] = torch.clip((1 + art_to_nonart_ratios_svra) / 2, min=0.01, max=100)

            counts_slv = torch.sum(self.counts_slvra, dim=(-2, -1))
            unlabeled_weight_sv = torch.clip(
                (counts_slv[:, Label.ARTIFACT] + counts_slv[:, Label.VARIANT]) / counts_slv[:, Label.UNLABELED],
                0,
                1,
            )
            new_weights_slvra[:, Label.UNLABELED] = unlabeled_weight_sv.view(self.num_sources, len(Variation), 1, 1)

            attenuation = math.pow(Balancer.ATTENUATION_PER_DATUM, self.count_since_last_recomputation)
            self.weights_slvra.copy_(attenuation * self.weights_slvra + (1 - attenuation) * new_weights_slvra)

            counts_s = torch.sum(counts_slv, dim=(-2, -1))
            total_s = torch.sum(counts_s, dim=0, keepdim=True)
            new_source_weights_s = (total_s / counts_s) / self.num_sources
            self.source_weights_s.copy_(attenuation * self.source_weights_s + (1 - attenuation) * new_source_weights_s)
            self.count_since_last_recomputation = 0
            # TODO: also attenuate counts -- multiply by an attenuation factor or something?
        batch_weights = batch.batch_indices().index_into_tensor(self.weights_slvra)
        source_weights = self.source_weights_s[batch.batch_indices().sources]
        return batch_weights, source_weights

    # TODO: lots of code duplication with the plotting in loss_metrics.py
    def make_plot(self, label: Label, var_type: Variation, axis, source: int, plot_type: PlotType):
        if plot_type == PlotType.WEIGHTS:
            vmin, vmax = -4, 4
            weights_ra = self.weights_slvra[source, label, var_type].cpu()
            plot_data_ra = torch.clip(torch.log(weights_ra), min=vmin, max=vmax)
        elif plot_type == PlotType.COUNTS:
            vmin, vmax = -10, 0
            counts_lra = self.counts_slvra[source, :, var_type].cpu()
            max_count = torch.max(counts_lra)
            normalized_counts_ra = (counts_lra / max_count)[label] + 0.0001
            plot_data_ra = torch.clip(torch.log(normalized_counts_ra), -10, 0)
        else:
            raise ValueError(f"Unknown type_of_plot: {plot_type.name}.")
        xbounds, ybounds = np.array(ALT_COUNT_BIN_BOUNDS), np.array(REF_COUNT_BIN_BOUNDS)
        kwargs = {"x_label": None, "y_label": None, "vmin": vmin, "vmax": vmax}

        return plotting.color_plot_2d_on_axis(axis, xbounds, ybounds, plot_data_ra, **kwargs)

    def make_plots(self, summary_writer: SummaryWriter, prefix: str, epoch: int, plot_type: PlotType):
        for source in range(self.num_sources):
            figsize = (2.5 * len(Variation), 2.5 * len(Label))
            subplots_kwargs = {"sharex": "all", "sharey": "all", "squeeze": False, "figsize": figsize}
            fig, axes = plt.subplots(len(Label), len(Variation), **subplots_kwargs)

            row_names = [label.name for label in Label]
            col_names = [var_type.name for var_type in Variation]
            common_colormesh = None
            for (label, var_type) in itertools.product(Label, Variation):
                common_colormesh = self.make_plot(label, var_type, axes[label, var_type], source, plot_type)

            fig.colorbar(common_colormesh)
            tidy_kwargs = {"x_label": "N_alt", "y_label": "N_ref", "row_labels": row_names, "col_labels": col_names}
            plotting.tidy_subplots(fig, axes, **tidy_kwargs)
            multi_source_suffix = ", all sources" if source is None else f", source {source}"
            source_suffix = "" if self.num_sources == 1 else multi_source_suffix
            summary_writer.add_figure(prefix + source_suffix, fig, global_step=epoch)
