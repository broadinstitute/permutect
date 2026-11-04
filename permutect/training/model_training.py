from __future__ import annotations

import time
from collections import defaultdict
from queue import PriorityQueue
from typing import Any

import torch
from torch import device
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm import trange

from permutect.architecture.artifact_model import ArtifactModel
from permutect.architecture.artifact_model import record_embeddings
from permutect.data.batch import Batch
from permutect.data.batch import DownsampledBatch
from permutect.data.batches_on_device import batches_on_device
from permutect.data.count_binning import round_alt_count_to_bin_center
from permutect.data.datum import Data
from permutect.data.datum import Datum
from permutect.data.reads_dataset import ReadsDataset
from permutect.metrics.evaluation_metrics import EvaluationMetrics
from permutect.misc_utils import Timer
from permutect.misc_utils import all_reduce_sum_
from permutect.misc_utils import backpropagate
from permutect.misc_utils import check_for_nan
from permutect.misc_utils import is_distributed
from permutect.misc_utils import is_main_process
from permutect.misc_utils import report_memory_usage
from permutect.parameters import TrainingParameters
from permutect.training.balancer import Balancer
from permutect.training.balancer import PlotType
from permutect.training.checkpoint import Checkpoint
from permutect.training.downsampler import Downsampler
from permutect.training.loss_recorder import LossRecorder
from permutect.utils.enums import Epoch
from permutect.utils.enums import Label

WORST_OFFENDERS_QUEUE_SIZE = 100


class ArtifactModelTrainingStep(torch.nn.Module):
    """
    DistributedDataParallel synchronizes gradients only for a backward pass from the output of its forward(), so training
    computes the model's output and losses in this module's forward().  The Balancer is an argument rather than a
    submodule: it has nothing to train, and its own sums over processes keep its counts the same in every process.
    """

    def __init__(self, model: ArtifactModel):
        super().__init__()
        self.model = model

    def forward(self, batch: DownsampledBatch, balancer: Balancer, update_balancer: bool):
        """
        :return: the total loss, the output and the losses.  DDP finds the tensors in its output only in containers such
            as tuples, not in a BatchLosses, and backpropagation must start from the total loss that DDP returns.
        """
        output = self.model.compute_batch_output(batch, balancer, update_balancer=update_balancer)
        losses = self.model.compute_batch_losses(output, batch)
        return losses.total_loss, output, losses


def train_artifact_model(
    model: ArtifactModel,
    train_dataset: ReadsDataset,
    valid_dataset: ReadsDataset | None,
    training_params: TrainingParameters,
    summary_writer: SummaryWriter,
    epochs_per_evaluation: int = 5,
):
    device, dtype = model._device, model._dtype
    balancer = Balancer(num_sources=train_dataset.num_sources(), device=device).to(device=device, dtype=dtype)
    downsampler: Downsampler = Downsampler(num_sources=train_dataset.num_sources()).to(device=device, dtype=dtype)
    downsampler.optimize_downsampling_balance(train_dataset.count_totals_slvcd.to(device=device))

    num_sources = train_dataset.validate_sources()
    if is_main_process():
        train_dataset.report_totals()
    is_cuda = device.type == "cuda"
    print(f"Is CUDA available? {is_cuda}")

    train_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_params.learning_rate,
        weight_decay=training_params.weight_decay,
    )
    scheduler_kwargs = {
        "factor": 0.2,
        "patience": 5,
        "threshold": 0.001,
        "min_lr": (training_params.learning_rate / 100),
    }
    train_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(train_optimizer, **scheduler_kwargs)

    checkpoint = Checkpoint(device, model, train_optimizer)

    # DistributedDataParallel tracks the parameters that require gradients when it is constructed, so they are chosen
    # first.  It also copies the parameters and buffers of process 0 to the other processes.  The reference-read
    # projections of the last gated MLP block never affect the loss, so with static_graph=True DDP finds such unused
    # parameters in the first step rather than expecting a gradient for every parameter.
    model.set_trainable_parameters(training_params.trainable_parameter_sets)
    training_step = ArtifactModelTrainingStep(model)
    if is_distributed():
        device_ids = [device.index] if is_cuda else None
        training_step = DistributedDataParallel(training_step, device_ids=device_ids, static_graph=True)

    train_loader = train_dataset.make_data_loader(training_params.batch_size, is_cuda, training_params.num_workers)
    valid_loader = (
        None
        if valid_dataset is None
        else valid_dataset.make_data_loader(training_params.batch_size, is_cuda, training_params.num_workers)
    )
    report_memory_usage("Loaders created, about to train.")

    first_epoch, last_epoch = 1, training_params.num_epochs
    for epoch in trange(1, last_epoch + 1, desc="Epoch", disable=not is_main_process()):
        start_of_epoch = time.time()
        report_memory_usage(f"Epoch {epoch}.")

        for epoch_type in [Epoch.TRAIN] if valid_loader is None else [Epoch.TRAIN, Epoch.VALID]:
            train_one_epoch(
                balancer,
                checkpoint,
                device,
                downsampler,
                epoch,
                epoch_type,
                epochs_per_evaluation,
                last_epoch,
                model,
                num_sources,
                summary_writer,
                train_loader,
                train_optimizer,
                train_scheduler,
                valid_loader,
                training_step,
            )

        # done with training and validation for this epoch
        report_memory_usage(f"End of epoch {epoch}.")
        print(f"Time elapsed(s): {time.time() - start_of_epoch:.1f}")
        # note that we have not learned the AF spectrum yet
    # done with training
    if is_main_process():  # from this process's shard of the training data
        report_memory_usage("Training complete, recording embeddings for tensorboard.")
        embeddings_timer = Timer("Creating training and validation datasets")
        record_embeddings(model, train_loader, summary_writer)
        embeddings_timer.report("Time to record embeddings for tensorboard.")


def train_one_epoch(
    balancer: Balancer,
    checkpoint: Checkpoint,
    device: device,
    downsampler: Downsampler,
    epoch: int,
    epoch_type: Epoch,
    epochs_per_evaluation: int,
    last_epoch: int,
    model: ArtifactModel,
    num_sources: int,
    summary_writer: SummaryWriter,
    train_loader: DataLoader[Any],
    train_optimizer: AdamW,
    train_scheduler: ReduceLROnPlateau,
    valid_loader: DataLoader[Any],
    training_step: torch.nn.Module = None,
):
    # a single process, eg in tests, needn't wrap the model in DistributedDataParallel
    training_step = ArtifactModelTrainingStep(model) if training_step is None else training_step
    loss_recorder = LossRecorder(device, num_sources)
    model.set_epoch_type(epoch_type)
    loader = train_loader if epoch_type == Epoch.TRAIN else valid_loader
    # very large losses are tracked on the device, so that checking each batch doesn't make the CPU wait for the GPU
    largest_average_loss = torch.zeros((), device=model._device)
    num_very_large_losses = torch.zeros((), device=model._device, dtype=torch.long)

    parent_batch: Batch
    parent_batches = batches_on_device(loader, device=model._device)
    for parent_batch in tqdm(parent_batches, mininterval=60, total=len(loader), disable=not is_main_process()):
        batch: DownsampledBatch
        for downsampling_iteration in range(2):
            ref_fracs_b, alt_fracs_b = downsampler.calculate_downsampling_fractions(parent_batch)
            batch = DownsampledBatch(parent_batch, ref_fracs_b, alt_fracs_b)
            # validation needs no autograd graph
            with torch.set_grad_enabled(epoch_type == Epoch.TRAIN):
                total_loss, output, losses = training_step(batch, balancer, update_balancer=epoch_type == Epoch.TRAIN)
            loss_recorder.record(output, losses, batch)

            if epoch_type == Epoch.TRAIN:
                average_loss = total_loss.detach() / batch.size()
                largest_average_loss = torch.maximum(largest_average_loss, average_loss)
                num_very_large_losses += average_loss > 100.0

                backpropagate(train_optimizer, total_loss, params_to_clip=model.parameters())
        # done with this downsampled batch
    # done with this parent batch
    loss_recorder.all_reduce()
    if epoch_type == Epoch.TRAIN and epoch > 1 and num_very_large_losses.item() > 0:
        print(f"{num_very_large_losses.item()} batches had very large losses, up to {largest_average_loss.item():.2f}.")
    check_for_nan(model)
    if epoch_type == Epoch.TRAIN:
        mean_loss = loss_recorder.mean_loss()
        train_scheduler.step(mean_loss)

    generate_plots = epoch % epochs_per_evaluation == 0 or epoch == last_epoch
    if is_main_process():
        loss_recorder.output_results(epoch_type, epoch, summary_writer, generate_plots)

    if generate_plots and is_main_process():
        weight_prefix = "log(label-balancing weights)" + f"({epoch_type.name})"
        count_prefix = "unweighted data counts after downsampling" + f"({epoch_type.name})"
        balancer.make_plots(summary_writer, weight_prefix, epoch, PlotType.WEIGHTS)
        balancer.make_plots(summary_writer, count_prefix, epoch, PlotType.COUNTS)

    if epoch_type == Epoch.VALID and generate_plots:
        print(f"performing evaluation on epoch {epoch}")
        evaluate_model(
            model,
            epoch,
            num_sources,
            balancer,
            downsampler,
            train_loader,
            valid_loader,
            summary_writer,
            report_worst=False,
        )

    if epoch_type == Epoch.TRAIN:
        checkpoint.save_checkpoint_if_needed(epoch, mean_loss)
        checkpoint.load_checkpoint_if_needed(mean_loss)


@torch.inference_mode()
def collect_evaluation_data(
    model: ArtifactModel,
    num_sources: int,
    balancer: Balancer,
    downsampler: Downsampler,
    train_loader,
    valid_loader,
    report_worst: bool,
):
    # the keys are tuples of (Label; rounded alt count)
    worst_offenders_by_label_and_alt_count = defaultdict(lambda: PriorityQueue(WORST_OFFENDERS_QUEUE_SIZE))

    evaluation_metrics = EvaluationMetrics(num_sources=num_sources, device=model._device)
    epoch_types = [Epoch.TRAIN, Epoch.VALID]
    for epoch_type in epoch_types:
        assert epoch_type == Epoch.TRAIN or epoch_type == Epoch.VALID  # not doing TEST here
        loader = train_loader if epoch_type == Epoch.TRAIN else valid_loader

        parent_batch: Batch
        parent_batches = batches_on_device(loader, device=model._device)
        for parent_batch in tqdm(parent_batches, mininterval=60, total=len(loader), disable=not is_main_process()):
            # TODO: magic constant
            for _ in range(3):
                ref_fracs_b, alt_fracs_b = downsampler.calculate_downsampling_fractions(parent_batch)
                batch = DownsampledBatch(parent_batch, ref_fracs_b=ref_fracs_b, alt_fracs_b=alt_fracs_b)
                output = model.compute_batch_output(batch, balancer, update_balancer=False)

                evaluation_metrics.record_batch(epoch_type, batch, logits=output.logits_b, weights=output.weights)

                if report_worst:
                    for int_array, float_array, predicted_logit in zip(
                        batch.get_int_array_be(),
                        batch.get_float_array_be(),
                        output.logits_b.detach().cpu().tolist(),
                    ):
                        datum = Datum(int_array, float_array)
                        wrong_call = (datum.get(Data.LABEL) == Label.ARTIFACT and predicted_logit < 0) or (
                            datum.get(Data.LABEL) == Label.VARIANT and predicted_logit > 0
                        )
                        if wrong_call:
                            alt_count = datum.get(Data.ALT_COUNT)
                            rounded_count = round_alt_count_to_bin_center(alt_count)
                            confidence = abs(predicted_logit)

                            # the 0th aka highest priority element in the queue is the one with the lowest confidence
                            pqueue = worst_offenders_by_label_and_alt_count[
                                (Label(datum.get(Data.LABEL)), rounded_count)
                            ]

                            # clear space if this confidence is more egregious
                            if pqueue.full() and pqueue.queue[0][0] < confidence:
                                pqueue.get()  # discards the least confident bad call

                            if not pqueue.full():  # if space was cleared or if it wasn't full already
                                pqueue.put(
                                    (
                                        confidence,
                                        str(datum.get(Data.CONTIG))
                                        + ":"
                                        + str(datum.get(Data.POSITION))
                                        + ":"
                                        + datum.get_ref_allele()
                                        + "->"
                                        + datum.get_alt_allele(),
                                    )
                                )
        # done with this epoch type
    # done collecting data
    return evaluation_metrics, worst_offenders_by_label_and_alt_count


@torch.inference_mode()
def evaluate_model(
    model: ArtifactModel,
    epoch: int,
    num_sources: int,
    balancer: Balancer,
    downsampler: Downsampler,
    train_loader,
    valid_loader,
    summary_writer: SummaryWriter,
    report_worst: bool = False,
):
    # self.freeze_all()
    evaluation_metrics, worst_offenders_by_label_and_alt_count = collect_evaluation_data(
        model, num_sources, balancer, downsampler, train_loader, valid_loader, report_worst
    )
    # in distributed training each process evaluates its own shard, in the same order of epoch types
    for epoch_type in (Epoch.TRAIN, Epoch.VALID):
        all_reduce_sum_(evaluation_metrics.accuracy_metrics_by_epoch_type[epoch_type])
    # process 0 makes the plots, and reports the worst offenders of its own shard
    if not is_main_process():
        return
    evaluation_metrics.put_on_cpu()
    evaluation_metrics.make_plots(summary_writer, epoch=epoch)

    if report_worst:
        for (true_label, rounded_count), pqueue in worst_offenders_by_label_and_alt_count.items():
            tag = f"True label: {true_label.name}, rounded alt count: {rounded_count}"

            lines = []
            while not pqueue.empty():  # this goes from least to most egregious, FYI
                confidence, var_string = pqueue.get()
                lines.append(f"{var_string} ({confidence:.2f})")

            summary_writer.add_text(tag, "\n".join(lines), global_step=epoch)

    # done collecting data
