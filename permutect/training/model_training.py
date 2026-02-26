import math
import time
from collections import defaultdict
from queue import PriorityQueue

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm

from permutect.architecture.feature_clustering import MAX_LOGIT, FeatureClustering
from permutect.training.balancer import Balancer
from permutect.training.checkpoint import Checkpoint
from permutect.training.downsampler import Downsampler
from permutect.architecture.artifact_model import ArtifactModel, record_embeddings
from permutect.data.batch import DownsampledBatch
from permutect.data.reads_dataset import ReadsDataset
from permutect.data.datum import Datum, Data
from permutect.data.prefetch_generator import prefetch_generator
from permutect.metrics.evaluation_metrics import EmbeddingMetrics, EvaluationMetrics
from permutect.data.batch import BatchProperty, Batch
from permutect.data.count_binning import alt_count_bin_index, round_alt_count_to_bin_center, alt_count_bin_name
from permutect.parameters import TrainingParameters
from permutect.misc_utils import report_memory_usage, backpropagate, freeze, unfreeze, check_for_nan
from permutect.training.training_losses import TrainingLosses
from permutect.utils.enums import Variation, Epoch, Label

WORST_OFFENDERS_QUEUE_SIZE = 100


def train_artifact_model(model: ArtifactModel, train_dataset: ReadsDataset, valid_dataset: ReadsDataset,
                         training_params: TrainingParameters, summary_writer: SummaryWriter,
                         epochs_per_evaluation: int = None):
    device, dtype = model._device, model._dtype
    bce = nn.BCEWithLogitsLoss(reduction='none')  # no reduction because we may want to first multiply by weights for unbalanced data
    balancer = Balancer(num_sources=train_dataset.num_sources(), device=device).to(device=device, dtype=dtype)
    downsampler: Downsampler = Downsampler(num_sources=train_dataset.num_sources()).to(device=device, dtype=dtype)
    downsampler.optimize_downsampling_balance(train_dataset.totals_slvra.to(device=device))
    checkpoint = Checkpoint(device=device)

    num_sources = train_dataset.validate_sources()
    train_dataset.report_totals()
    model.reset_source_predictor(num_sources)
    is_cuda = device.type == 'cuda'
    print(f"Is CUDA available? {is_cuda}")

    train_optimizer = torch.optim.AdamW(model.parameters(), lr=training_params.learning_rate, weight_decay=training_params.weight_decay)
    train_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(train_optimizer, factor=0.2, patience=5,
        threshold=0.001, min_lr=(training_params.learning_rate / 100), verbose=True)

    train_loader = train_dataset.make_data_loader(training_params.batch_size, is_cuda, training_params.num_workers)
    valid_loader = valid_dataset.make_data_loader(training_params.inference_batch_size, is_cuda, training_params.num_workers)

    first_epoch, last_epoch = 1, training_params.num_epochs + training_params.num_calibration_epochs
    for epoch in trange(1, last_epoch + 1, desc="Epoch"):
        start_of_epoch = time.time()
        report_memory_usage(f"Epoch {epoch}.")
        is_calibration_epoch = epoch > training_params.num_epochs

        model.source_predictor.set_adversarial_strength((2 / (1 + math.exp(-0.1 * (epoch - 1)))) - 1)

        for epoch_type in [Epoch.TRAIN, Epoch.VALID]:
            model.set_epoch_type(epoch_type, is_calibration_epoch=is_calibration_epoch)
            training_losses = TrainingLosses(num_sources=num_sources, device=device)
            loader = train_loader if epoch_type == Epoch.TRAIN else valid_loader

            parent_batch: Batch
            for parent_batch in tqdm(prefetch_generator(loader), mininterval=60, total=len(loader)):
                labels_b = parent_batch.get_training_labels()
                is_labeled_b = parent_batch.get_is_labeled_mask()

                batch: DownsampledBatch
                for downsampling_iteration in range(2):
                    ref_fracs_b, alt_fracs_b = downsampler.calculate_downsampling_fractions(parent_batch)
                    batch = DownsampledBatch(parent_batch, ref_fracs_b=ref_fracs_b, alt_fracs_b=alt_fracs_b)
                    output = model.compute_batch_output(batch, balancer)
                    source_losses_b = model.compute_source_prediction_losses(output.features_be, batch)
                    alt_count_losses_b = model.compute_alt_count_losses(output.features_be, batch)
                    supervised_losses_b = is_labeled_b * bce(output.calibrated_logits_b, labels_b)

                    # to penalize outliers / encourage data in high prob density, unsupervised loss is binary cross entropy
                    # where targets are all "not-outlier" (i.e. 0).  Since some genuine outlier data does exist, such as
                    # rare or unmodeled artifacts, we clip the outlier logit to avert unduly strong influence.
                    outlier_binary_logits_b = FeatureClustering.outlier_binary_logits(output.calibrated_logits_bk)
                    outlier_losses_b = bce(torch.clip(outlier_binary_logits_b, max=MAX_LOGIT/2), torch.zeros_like(outlier_binary_logits_b))
                    unsupervised_losses_b = (1 - is_labeled_b) * outlier_losses_b

                    losses = output.weights * (supervised_losses_b + unsupervised_losses_b + alt_count_losses_b) + output.source_weights * source_losses_b
                    loss = torch.sum(losses)

                    training_losses.record(batch, supervised_losses_b, unsupervised_losses_b, source_losses_b, alt_count_losses_b, output)

                    if epoch_type == Epoch.TRAIN:
                        backpropagate(train_optimizer, loss, params_to_clip=model.parameters())

                # done with this batch
            check_for_nan(model)
            # done with one epoch type -- training or validation -- for this epoch
            if epoch_type == Epoch.TRAIN:
                # TODO: what is it's unsupefvised-only training?
                mean_over_labels = torch.mean(training_losses.supervised_loss_metrics.get_marginal(BatchProperty.LABEL)).item()
                train_scheduler.step(mean_over_labels)

            perform_evaluation = (epochs_per_evaluation is not None and epoch % epochs_per_evaluation == 0) or (epoch == last_epoch)
            training_losses.write_to_summary_writer(epoch_type, epoch, summary_writer, make_plots=perform_evaluation)
            if perform_evaluation:
                balancer.make_plots(summary_writer, "log(label-balancing weights)", epoch_type, epoch, type_of_plot="weights")
                balancer.make_plots(summary_writer, "unweighted data counts after downsampling", epoch_type, epoch, type_of_plot="counts")

                print(f"performing evaluation on epoch {epoch}")
                if epoch_type == Epoch.VALID:
                    evaluate_model(model, epoch, num_sources, balancer, downsampler, train_loader, valid_loader, summary_writer, collect_embeddings=False, report_worst=False)

            if not is_calibration_epoch and epoch_type == Epoch.TRAIN:
                mean_over_labels = torch.mean(training_losses.supervised_loss_metrics.get_marginal(BatchProperty.LABEL))
                checkpoint.save_checkpoint_if_needed(model, train_optimizer, epoch, loss=mean_over_labels)
                checkpoint.load_checkpoint_if_needed(model, train_optimizer, loss=mean_over_labels)

        report_memory_usage(f"Done with training and validation for epoch {epoch}.")
        print(f"Time elapsed(s): {time.time() - start_of_epoch:.1f}")
        # note that we have not learned the AF spectrum yet
    # done with training
    report_memory_usage(f"Training complete, recording embeddings for tensorboard.")
    record_embeddings(model, train_loader, summary_writer)

@torch.inference_mode()
def collect_evaluation_data(model: ArtifactModel, num_sources: int, balancer: Balancer, downsampler: Downsampler,
                            train_loader, valid_loader, report_worst: bool):
    # the keys are tuples of (Label; rounded alt count)
    worst_offenders_by_label_and_alt_count = defaultdict(lambda: PriorityQueue(WORST_OFFENDERS_QUEUE_SIZE))

    evaluation_metrics = EvaluationMetrics(num_sources=num_sources, device=model._device)
    epoch_types = [Epoch.TRAIN, Epoch.VALID]
    for epoch_type in epoch_types:
        assert epoch_type == Epoch.TRAIN or epoch_type == Epoch.VALID  # not doing TEST here
        loader = train_loader if epoch_type == Epoch.TRAIN else valid_loader

        parent_batch: Batch
        for parent_batch in tqdm(prefetch_generator(loader), mininterval=60, total=len(loader)):
            # TODO: magic constant
            for _ in range(3):
                ref_fracs_b, alt_fracs_b = downsampler.calculate_downsampling_fractions(parent_batch)
                batch = DownsampledBatch(parent_batch, ref_fracs_b=ref_fracs_b, alt_fracs_b=alt_fracs_b)
                output = model.compute_batch_output(batch, balancer)

                evaluation_metrics.record_batch(epoch_type, batch, logits=output.calibrated_logits_b, weights=output.weights)

                if report_worst:
                    for int_array, float_array, predicted_logit in zip(batch.get_int_array_be(), batch.get_float_array_be(), output.calibrated_logits_b.detach().cpu().tolist()):
                        datum = Datum(int_array, float_array)
                        wrong_call = (datum.get(Data.LABEL) == Label.ARTIFACT and predicted_logit < 0) or \
                                     (datum.get(Data.LABEL) == Label.VARIANT and predicted_logit > 0)
                        if wrong_call:
                            alt_count = datum.get(Data.ALT_COUNT)
                            rounded_count = round_alt_count_to_bin_center(alt_count)
                            confidence = abs(predicted_logit)

                            # the 0th aka highest priority element in the queue is the one with the lowest confidence
                            pqueue = worst_offenders_by_label_and_alt_count[(Label(datum.get(Data.LABEL)), rounded_count)]

                            # clear space if this confidence is more egregious
                            if pqueue.full() and pqueue.queue[0][0] < confidence:
                                pqueue.get()  # discards the least confident bad call

                            if not pqueue.full():  # if space was cleared or if it wasn't full already
                                pqueue.put((confidence, str(datum.get(Data.CONTIG)) + ":" + str(
                                    datum.get(Data.POSITION)) + ':' + datum.get_ref_allele() + "->" + datum.get_alt_allele()))
        # done with this epoch type
    # done collecting data
    return evaluation_metrics, worst_offenders_by_label_and_alt_count


@torch.inference_mode()
def evaluate_model(model: ArtifactModel, epoch: int, num_sources: int, balancer: Balancer, downsampler: Downsampler, train_loader, valid_loader,
                   summary_writer: SummaryWriter, collect_embeddings: bool = False, report_worst: bool = False):

    # self.freeze_all()
    evaluation_metrics, worst_offenders_by_label_and_alt_count = collect_evaluation_data(model, num_sources, balancer, downsampler, train_loader, valid_loader, report_worst)
    evaluation_metrics.put_on_cpu()
    evaluation_metrics.make_plots(summary_writer, epoch=epoch)

    if report_worst:
        for (true_label, rounded_count), pqueue in worst_offenders_by_label_and_alt_count.items():
            tag = f"True label: {true_label.name}, rounded alt count: {rounded_count}"

            lines = []
            while not pqueue.empty():   # this goes from least to most egregious, FYI
                confidence, var_string = pqueue.get()
                lines.append(f"{var_string} ({confidence:.2f})")

            summary_writer.add_text(tag, "\n".join(lines), global_step=epoch)

    if collect_embeddings:
        embedding_metrics = EmbeddingMetrics()

        # now go over just the validation data and generate feature vectors / metadata for tensorboard projectors
        batch: Batch
        for batch in tqdm(prefetch_generator(valid_loader), mininterval=60, total=len(valid_loader)):
            logits_b, _, alt_means_be, ref_means_be = model.calculate_logits(batch)
            pred_b = logits_b.detach().cpu()
            labels_b = batch.get_training_labels().cpu()
            correct_b = ((pred_b > 0) == (labels_b > 0.5)).tolist()
            is_labeled_list = batch.get_is_labeled_mask().cpu().tolist()

            label_strings = [("artifact" if label > 0.5 else "non-artifact") if is_labeled > 0.5 else "unlabeled"
                             for (label, is_labeled) in zip(labels_b.tolist(), is_labeled_list)]

            correct_strings = [str(correctness) if is_labeled > 0.5 else "-1"
                             for (correctness, is_labeled) in zip(correct_b, is_labeled_list)]

            embedding_metrics.label_metadata.extend(label_strings)
            embedding_metrics.correct_metadata.extend(correct_strings)
            embedding_metrics.type_metadata.extend([Variation(idx).name for idx in batch.get(Data.VARIANT_TYPE).cpu().tolist()])
            embedding_metrics.truncated_count_metadata.extend([alt_count_bin_name(alt_count_bin_index(alt_count)) for alt_count in batch.get(Data.ALT_COUNT).cpu().tolist()])
            embedding_metrics.features.append(alt_means_be.detach().cpu())
            embedding_metrics.ref_features.append(ref_means_be.detach().cpu())
        embedding_metrics.output_to_summary_writer(summary_writer, epoch=epoch)
    # done collecting data
