import math
import random
from collections import defaultdict
from typing import List
from typing import NamedTuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from permutect.data.batch import Batch
from permutect.data.count_binning import NUM_ALT_COUNT_BINS
from permutect.data.count_binning import NUM_REF_COUNT_BINS
from permutect.data.count_binning import alt_count_bin_index
from permutect.data.count_binning import alt_count_bin_name
from permutect.data.count_binning import ref_count_bin_name
from permutect.metrics import plotting
from permutect.metrics.loss_metrics import AccuracyMetrics
from permutect.metrics.posterior_result import PosteriorResult
from permutect.misc_utils import resolve_device
from permutect.utils.enums import Call
from permutect.utils.enums import Epoch
from permutect.utils.enums import Label
from permutect.utils.enums import Variation

NUM_DATA_FOR_TENSORBOARD_PROJECTION = 10000


class Mistake(NamedTuple):
    """the call of a variant called wrongly, and the values of its posterior result in the mistake histograms"""

    call: Call
    variant_type: int
    alt_count: int
    depth: int
    artifact_logit: float
    call_probability: float  # the posterior probability of the mistaken call


class EvaluationMetrics:
    def __init__(self, num_sources, device=None):
        device = resolve_device(device)
        # we will have a map from epoch type to EvaluationMetricsForOneEpochType
        self.accuracy_metrics_by_epoch_type = defaultdict(lambda: AccuracyMetrics.create(num_sources, device))

        # a Mistake for each variant called wrongly
        self.mistakes = []
        self.has_been_sent_to_cpu = False

    def put_on_cpu(self):
        """
        Do this at the end of an epoch so that the whole tensor is on CPU in one operation rather than computing various
        marginals etc on GPU and sending them each to CPU for plotting etc.
        :return:
        """
        for key, val in self.accuracy_metrics_by_epoch_type.items():
            self.accuracy_metrics_by_epoch_type[key] = val.cpu()
        self.has_been_sent_to_cpu = True
        return self

    def record_batch(
        self,
        epoch_type: Epoch,
        batch: Batch,
        logits: Tensor,
        weights: Tensor = None,
        use_original_counts: bool = False,
    ):
        assert not self.has_been_sent_to_cpu, "Can't record after already sending to CPU"
        # unlabeled data only appear in the logit histograms, since accuracy, calibration and ROC curves need labels
        self.accuracy_metrics_by_epoch_type[epoch_type].record(
            batch,
            values=weights if weights is not None else torch.ones_like(logits),
            logits=logits,
            use_original_counts=use_original_counts,
        )

    # track bad calls when filtering is given an optional evaluation truth VCF
    def record_mistake(self, posterior_result: PosteriorResult, call: Call):
        # only the values in the histograms rather than the whole result, which has embeddings and lists of values
        self.mistakes.append(
            Mistake(
                call=call,
                variant_type=posterior_result.variant_type,
                alt_count=posterior_result.alt_count,
                depth=posterior_result.depth,
                artifact_logit=posterior_result.artifact_logit,
                call_probability=posterior_result.posterior_probabilities[call],
            )
        )

    def make_mistake_histograms(self, summary_writer: SummaryWriter):
        assert self.has_been_sent_to_cpu, "Can't make plots before sending to CPU"
        # if every labeled site was called correctly there is nothing to plot, and plt.subplots(0, ...) would fail
        if not self.mistakes:
            return

        # indexed by call then var_type, inner is a list of the mistakes with that call and var type
        mistakes_by_call_and_var_type = defaultdict(lambda: defaultdict(list))
        for mistake in self.mistakes:
            mistakes_by_call_and_var_type[mistake.call][mistake.variant_type].append(mistake)

        mistake_calls = mistakes_by_call_and_var_type.keys()
        num_rows = len(mistake_calls)
        variation_types = [var_type.name for var_type in Variation]
        row_names = [mistake.name for mistake in mistake_calls]

        # each figure's tag, x-axis label, and data for the mistakes with a given call
        histograms = (
            (
                "mistake allele fractions",
                "alt allele fraction",
                lambda mistakes: [mistake.alt_count / mistake.depth for mistake in mistakes],
            ),
            ("mistake alt counts", "alt count", lambda mistakes: [mistake.alt_count for mistake in mistakes]),
            # known artifacts have infinite artifact logits, which can't be histogrammed
            (
                "mistake artifact logits",
                "artifact logit",
                lambda mistakes: [m.artifact_logit for m in mistakes if math.isfinite(m.artifact_logit)],
            ),
            # posterior probability assigned to this incorrect call
            (
                "probability assigned to mistake calls",
                "mistake call probability",
                lambda mistakes: [mistake.call_probability for mistake in mistakes],
            ),
        )
        for tag, x_label, histogram_data in histograms:
            fig, axes = plt.subplots(num_rows, len(Variation), sharex="all", sharey="none", squeeze=False)
            for row_idx, mistake_call in enumerate(mistake_calls):
                for var_type in Variation:
                    data = histogram_data(mistakes_by_call_and_var_type[mistake_call][var_type])
                    plotting.simple_histograms_on_axis(axes[row_idx, var_type], [data], [""], 20)
            plotting.tidy_subplots(
                fig, axes, x_label=x_label, y_label="", row_labels=row_names, col_labels=variation_types
            )
            summary_writer.add_figure(tag, fig)

    def make_plots(
        self,
        summary_writer: SummaryWriter,
        given_thresholds=None,
        sens_prec: bool = False,
        epoch: int = None,
    ):
        assert self.has_been_sent_to_cpu, "Can't make plots before sending to CPU"
        # given_thresholds is a dict from Variation to float (logit-scaled) used in the ROC curves
        num_sources = next(iter(self.accuracy_metrics_by_epoch_type.values())).num_sources()
        ref_count_bins = list(range(NUM_REF_COUNT_BINS)) + [None]
        alt_count_bins = list(range(NUM_ALT_COUNT_BINS)) + [None]
        ref_count_names = [ref_count_bin_name(bin_idx) for bin_idx in range(NUM_REF_COUNT_BINS)] + ["ALL"]
        alt_count_names = [alt_count_bin_name(bin_idx) for bin_idx in range(NUM_ALT_COUNT_BINS)] + ["ALL"]

        # None means all sources, which is the same as source 0 when there is only one
        sources = list(range(num_sources)) + ([None] if num_sources > 1 else [])

        accuracy_metrics: AccuracyMetrics
        for epoch_type, accuracy_metrics in self.accuracy_metrics_by_epoch_type.items():
            for source in sources:
                # make accuracy plots: overall figure is rows = label, columns = variant
                # each subplot is color map of accuracy where x is ref count, y is alt count
                acc_fig = plotting.label_and_variant_type_figure(
                    lambda label, var_type, axis: accuracy_metrics.plot_accuracy(label, var_type, axis, source),
                    labels=(Label.VARIANT, Label.ARTIFACT),
                )
                cal_fig, cal_axes = plt.subplots(
                    len(ref_count_bins),
                    len(alt_count_bins),
                    sharex="all",
                    sharey="all",
                    squeeze=False,
                )
                roc_fig, roc_axes = plt.subplots(
                    len(ref_count_bins),
                    len(alt_count_bins),
                    sharex="all",
                    sharey="all",
                    squeeze=False,
                    dpi=200,
                )

                for row, ref_count_bin in enumerate(ref_count_bins):
                    for col, alt_count_bin in enumerate(alt_count_bins):
                        accuracy_metrics.plot_calibration(cal_axes[row, col], ref_count_bin, alt_count_bin, source)
                        accuracy_metrics.plot_roc(
                            roc_axes[row, col],
                            ref_count_bin,
                            alt_count_bin,
                            source,
                            given_thresholds,
                            sens_prec,
                        )

                nonart_label = "sensitivity" if sens_prec else "non-artifact accuracy"
                art_label = "precision" if sens_prec else "artifact accuracy"

                plotting.tidy_subplots(
                    roc_fig,
                    roc_axes,
                    x_label=nonart_label,
                    y_label=art_label,
                    row_labels=ref_count_names,
                    col_labels=alt_count_names,
                )
                plotting.tidy_subplots(
                    cal_fig,
                    cal_axes,
                    x_label="logit",
                    y_label="accuracy",
                    row_labels=ref_count_names,
                    col_labels=alt_count_names,
                )

                source_suffix = (
                    "" if num_sources == 1 else (", all sources" if source is None else f", source {source}")
                )

                summary_writer.add_figure(
                    f"accuracy by alt and ref count ({epoch_type.name})" + source_suffix,
                    acc_fig,
                    global_step=epoch,
                )
                summary_writer.add_figure(
                    f"accuracy vs predicted logit ({epoch_type.name})" + source_suffix,
                    cal_fig,
                    global_step=epoch,
                )
                sp_name = "sensitivity vs precision" if sens_prec else "variant accuracy vs artifact accuracy"
                summary_writer.add_figure(f"{sp_name} ({epoch_type.name})" + source_suffix, roc_fig, global_step=epoch)

            # One more plot.  In each figure the grid of subplots is by variant type and count.  Within each subplot we have
            # overlapping density plots of artifact logit predictions for all combinations of Label and source
            hist_fig, hist_ax = accuracy_metrics.make_logit_histograms()
            summary_writer.add_figure(f"logit histograms ({epoch_type.name})", hist_fig, global_step=epoch)


class EmbeddingDatum(NamedTuple):
    """A datum recorded for the projections: its metadata, its embeddings, and its simulated twin's source"""

    metadata: tuple  # label, correctness, type, and count, as in the projections
    features: Tensor
    ref_features: Tensor | None
    simulated_features: Tensor | None
    simulated_source: str | None


class EmbeddingMetrics:
    TRUE_POSITIVE = "true-positive"
    FALSE_POSITIVE = "false-positive"
    TRUE_NEGATIVE_ARTIFACT = (
        "true-negative-artifact"  # distinguish these because artifact and eg germline should embed differently
    )
    TRUE_NEGATIVE_NONARTIFACT = "true-negative-nonartifact"
    TRUE_NEGATIVE = "true-negative"
    FALSE_NEGATIVE_ARTIFACT = "false-negative-artifact"
    TRUE_NEGATIVE_SEQ_ERROR = "true-negative-seq-error"
    UNKNOWN = "unknown"  # unlabeled data, or labeled data whose correctness isn't evaluated
    MODEL = "model"  # embeddings simulated from the model, which have no correctness

    # the projections keep every datum with these correctness statuses, up to the sample size, but only some others.
    # Without labeled truth, every datum is boring.
    INTERESTING_STATUSES = (TRUE_POSITIVE, FALSE_POSITIVE, FALSE_NEGATIVE_ARTIFACT)

    # the Source metadata of data's embeddings, as opposed to embeddings simulated from the model
    DATA_SOURCE = "data"

    @staticmethod
    def twin_metadata(data_metadata: tuple, source: str) -> tuple:
        """
        The metadata of a datum's simulated twin, given the datum's label, correctness, type, and count: the twin's label
        is "model" and the datum's label, e.g. "model artifact", its correctness is "model", it shares the datum's type
        and count, and its source is where in the model it came from.
        """
        label, _, variant_type, count = data_metadata
        return "model " + label.lower(), EmbeddingMetrics.MODEL, variant_type, count, source

    @staticmethod
    def group_names() -> list[str]:
        """the groups of data with their own projections: each variant type, then each alt count bin"""
        variant_types = ["variant type " + var_type.name for var_type in Variation]
        return variant_types + ["alt count " + alt_count_bin_name(count_bin) for count_bin in range(NUM_ALT_COUNT_BINS)]

    def __init__(self, sample_size: int = NUM_DATA_FOR_TENSORBOARD_PROJECTION):
        """
        :param sample_size: the most data of each group in a projection.  For each group and for its boring and its
            interesting data, the metrics keep the number recorded and a uniform random sample of at most this many, so
            that memory doesn't grow with the number of data.
        """
        self.sample_size = sample_size
        # indexed by group name, then by whether the data are interesting
        self.counts = defaultdict(lambda: [0, 0])
        self.samples = defaultdict(lambda: ([], []))

    def record_batch(
        self,
        labels,
        variant_types,
        alt_counts,
        features_be: Tensor,
        ref_features_be: Tensor = None,
        correctness: List[str] = None,
        simulated_features_be: Tensor = None,
        simulated_sources: List[str] = None,
    ):
        """
        Record metadata and embeddings for a batch of data, so that every TensorBoard projection uses the same metadata.

        :param labels: the Label of each datum (as an enum or int)
        :param variant_types: the Variation of each datum (as an enum or int)
        :param alt_counts: the alt count of each datum
        :param features_be: embeddings, one row per datum, or a single datum's 1D embedding
        :param ref_features_be: optional ref embeddings, one row per datum
        :param correctness: the correctness status of each datum, one of the class constants, or None if not evaluated
        :param simulated_features_be: optional embeddings, one row per datum, simulated from the model for a datum like
            each one, e.g. with the same alt count.  Projections of the embeddings show them alongside the data's.
        :param simulated_sources: with simulated_features_be, where in the model each simulated embedding came from
        """
        features_be = torch.atleast_2d(features_be)
        ref_features_be = None if ref_features_be is None else torch.atleast_2d(ref_features_be)
        simulated_features_be = None if simulated_features_be is None else torch.atleast_2d(simulated_features_be)
        for n, (label, variant_type, alt_count) in enumerate(zip(labels, variant_types, alt_counts, strict=True)):
            status = EmbeddingMetrics.UNKNOWN if correctness is None else correctness[n]
            metadata = (
                Label(label).name,
                status,
                Variation(variant_type).name,
                alt_count_bin_name(alt_count_bin_index(alt_count)),
            )
            interesting = status in EmbeddingMetrics.INTERESTING_STATUSES
            groups = ("variant type " + metadata[2], "alt count " + metadata[3])
            slots = [(group, self._sample_slot(group, interesting)) for group in groups]
            if all(slot is None for _, slot in slots):
                continue

            # copies, so that the sample doesn't keep whole batches in memory
            datum = EmbeddingDatum(
                metadata,
                features_be[n].clone(),
                None if ref_features_be is None else ref_features_be[n].clone(),
                None if simulated_features_be is None else simulated_features_be[n].clone(),
                None if simulated_features_be is None else "model " + simulated_sources[n],
            )
            for group, slot in slots:
                sample = self.samples[group][interesting]
                if slot == len(sample):
                    sample.append(datum)
                elif slot is not None:
                    sample[slot] = datum

    def _sample_slot(self, group: str, interesting: bool) -> int | None:
        """
        Count a new datum of a group and kind, and choose where it goes in their sample, by reservoir sampling: the next
        slot while the sample isn't full, and then the slot of a random datum that it replaces, or None to leave it out,
        so that the sample is always uniform
        """
        self.counts[group][interesting] += 1
        sample = self.samples[group][interesting]
        if len(sample) < self.sample_size:
            return len(sample)
        slot = random.randrange(self.counts[group][interesting])
        return slot if slot < self.sample_size else None

    def projection_data(self, group: str, is_filter_variants: bool) -> list[EmbeddingDatum]:
        """
        A random sample of a group's data for its projection: of every interesting datum and some boring ones, in
        filter_variants a third as many as the interesting ones but at least 100, and otherwise all of them, a uniform
        sample of at most sample_size.  The numbers of interesting and boring data in such a sample are hypergeometric,
        and given them, the data of each kind are a uniform sample of that kind, as a subsample of its uniform sample is.
        """
        num_boring, num_interesting = self.counts[group]
        boring_sample, interesting_sample = self.samples[group]
        num_boring_in_pool = min(max(num_interesting // 3, 100), num_boring) if is_filter_variants else num_boring
        pool_size = num_boring_in_pool + num_interesting
        num_kept = min(pool_size, self.sample_size)
        if num_kept == pool_size:
            num_interesting_kept = num_interesting
        else:
            num_interesting_kept = int(np.random.hypergeometric(num_interesting, num_boring_in_pool, num_kept))

        def subsample(sample: list, size: int) -> list:
            return [sample[n] for n in np.random.choice(len(sample), size=size, replace=False).tolist()]

        return subsample(interesting_sample, num_interesting_kept) + subsample(
            boring_sample, num_kept - num_interesting_kept
        )

    def output_to_summary_writer(
        self,
        summary_writer: SummaryWriter,
        prefix: str = "",
        is_filter_variants: bool = False,
        epoch: int = None,
    ):
        # read average embeddings stratified by variant type, then by alt count
        for group_name in EmbeddingMetrics.group_names():
            data = self.projection_data(group_name, is_filter_variants)
            # skip groups without data, e.g. indels in an SNV-only panel; tensorboard fails on empty embeddings
            if not data:
                continue
            data_metadata = [datum.metadata + (EmbeddingMetrics.DATA_SOURCE,) for datum in data]
            for kind in ("embedding", "ref embedding"):
                features = [datum.features if kind == "embedding" else datum.ref_features for datum in data]
                if features[0] is None:
                    continue
                projected_features, metadata = torch.vstack(features), data_metadata

                # each datum's simulated twin.  The model describes only the alt embeddings, not the ref embeddings.
                if kind == "embedding" and data[0].simulated_features is not None:
                    simulated_features = torch.vstack([datum.simulated_features for datum in data])
                    projected_features = torch.vstack((projected_features, simulated_features))
                    metadata = data_metadata + [
                        EmbeddingMetrics.twin_metadata(datum.metadata, datum.simulated_source) for datum in data
                    ]

                summary_writer.add_embedding(
                    projected_features,
                    metadata=metadata,
                    metadata_header=["Labels", "Correctness", "Types", "Counts", "Source"],
                    tag=prefix + kind + " for " + group_name,
                    global_step=epoch,
                )
