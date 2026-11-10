"""
Fitting the germline model's indel parameters (indel_parameters.py) to indels that are known to be germline, eg because
a matched normal has their alt alleles.

For each repeat class, the shortfall scale, offset and slope and the step function of concentrations maximize the total
germline log likelihood of the class's indels, which the germline model gives each sample's indels, without a normal,
using that sample's emission model as segment_pileups fit it to the sample's SNPs.  The concentrations are parametrized
so that they can't increase with STR length, through a first bin's value and non-negative decrements, and stay above a
smooth floor.

Nothing calls this yet.  IndelParameters' defaults came from it.
"""

import copy
import itertools
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from permutect.segmentation.emissions import EmissionModel
from permutect.segmentation.germline_model import GermlineModel
from permutect.segmentation.germline_model import SampleSegmentation
from permutect.segmentation.indel_parameters import NOT_IN_STR
from permutect.segmentation.indel_parameters import NUM_REPEAT_CLASSES
from permutect.segmentation.indel_parameters import IndelParameters
from permutect.segmentation.indel_parameters import repeat_classes

# every bin of STR lengths but the last has at least this many indels
MIN_INDELS_PER_BIN = 15
# the largest shortfall scale, so that an expected alt fraction never reaches 0, even in the longest STRs
MAX_SHORTFALL_SCALE = 0.9
# after fitting, consecutive bins whose log concentrations differ by less than this merge, keeping the first's
MERGE_TOLERANCE = 0.01
# a smooth floor under the concentrations, so that a long L-BFGS step can't drive one to 0, where the beta binomial is NaN
LOG_MIN_CONCENTRATION = math.log(0.5)
# L-BFGS runs in rounds until the log likelihood improves by less than the tolerance
MAX_ROUNDS = 15
MAX_ITERATIONS_PER_ROUND = 100
TOLERANCE = 1e-3


@dataclass
class GermlineIndels:
    """
    One sample's indels that are known to be germline, with what their germline likelihoods need, and the emission
    model that segment_pileups fit to the sample's SNPs
    """

    emission_model: EmissionModel
    alt_counts: Tensor
    depths: Tensor
    log_allele_frequencies: Tensor
    mafs: Tensor
    repeat_unit_lengths: Tensor
    repeat_lengths: Tensor

    def subset(self, mask: Tensor) -> "GermlineIndels":
        fields = (
            self.alt_counts,
            self.depths,
            self.log_allele_frequencies,
            self.mafs,
            self.repeat_unit_lengths,
            self.repeat_lengths,
        )
        return GermlineIndels(self.emission_model, *(field[mask] for field in fields))


def make_bins(lengths: np.ndarray, min_indels: int = MIN_INDELS_PER_BIN) -> list[int]:
    """
    The upper bounds of bins of consecutive STR lengths, merged until each has at least min_indels indels.  A smaller
    remainder joins the last bin.
    """
    values, counts = np.unique(lengths, return_counts=True)
    upper_bounds, running = [], 0
    for value, count in zip(values, counts):
        running += count
        if running >= min_indels:
            upper_bounds.append(int(value))
            running = 0
    if running > 0:
        if upper_bounds:
            upper_bounds[-1] = int(values[-1])
        else:
            upper_bounds = [int(values[-1])]
    return upper_bounds


def germline_log_likelihood(samples: list[GermlineIndels], parameters: IndelParameters) -> Tensor:
    """the total germline log likelihood, without a normal, of the samples' indels"""
    total = torch.zeros((), dtype=torch.float64)
    for sample in samples:
        model = GermlineModel(SampleSegmentation(emission_model=sample.emission_model), indel_parameters=parameters)
        no_normal = torch.zeros_like(sample.depths)
        log_likelihoods = model.germline_log_likelihoods(
            sample.alt_counts,
            sample.depths,
            no_normal,
            no_normal,
            sample.log_allele_frequencies,
            sample.mafs,
            torch.full_like(sample.mafs, 0.5),
            indels=torch.ones_like(sample.depths, dtype=torch.bool),
            repeat_unit_lengths=sample.repeat_unit_lengths,
            repeat_lengths=sample.repeat_lengths,
        )
        total = total + torch.sum(log_likelihoods)
    return total


def with_class(
    initial: IndelParameters, k: int, scale, offset, slope, upper_bounds: Tensor, log_concentrations: Tensor
):
    """the initial parameters, but with repeat class k's replaced"""

    def replace(tensor: Tensor, value) -> Tensor:
        return torch.stack([value if j == k else tensor[j] for j in range(NUM_REPEAT_CLASSES)])

    bounds, logs = initial.concentration_upper_bounds(), initial.log_concentrations()
    return IndelParameters(
        replace(initial.shortfall_scales, scale),
        replace(initial.shortfall_offsets, offset),
        replace(initial.shortfall_slopes, slope),
        [upper_bounds if j == k else bounds[j] for j in range(NUM_REPEAT_CLASSES)],
        [log_concentrations if j == k else logs[j] for j in range(NUM_REPEAT_CLASSES)],
    )


def fit_repeat_class(samples: list[GermlineIndels], k: int, initial: IndelParameters, min_indels_per_bin: int):
    """repeat class k's parameters that maximize the log likelihood of the samples' indels, all of which are in class k"""
    lengths = np.concatenate([sample.repeat_lengths.numpy() for sample in samples])
    upper_bounds = torch.tensor([0] if k == NOT_IN_STR else make_bins(lengths, min_indels_per_bin), dtype=torch.float64)
    num_bins = len(upper_bounds)
    # the initial scale as a fraction of the largest
    initial_fraction = float(np.clip(initial.shortfall_scales[k].item() / MAX_SHORTFALL_SCALE, 1e-6, 1 - 1e-6))

    def parameters_from(raw: dict) -> IndelParameters:
        # u decreases from bin to bin, and so does the log concentration, LOG_MIN_CONCENTRATION + softplus(u)
        decrements = torch.nn.functional.softplus(raw["log_decrements"])
        u = raw["first_u"] - torch.cat((torch.zeros(1, dtype=torch.float64), torch.cumsum(decrements, 0)))
        log_concentrations = LOG_MIN_CONCENTRATION + torch.nn.functional.softplus(u)
        slope = raw["slope"] if k != NOT_IN_STR else torch.zeros((), dtype=torch.float64)
        scale = MAX_SHORTFALL_SCALE * torch.sigmoid(raw["scale_logit"])
        return with_class(initial, k, scale, raw["offset"], slope, upper_bounds, log_concentrations)

    # from the initial shortfall and from none, with a high and a low first concentration
    initial_shortfall = (
        float(np.log(initial_fraction / (1 - initial_fraction))),
        initial.shortfall_offsets[k].item(),
        initial.shortfall_slopes[k].item(),
    )
    starts = itertools.product([initial_shortfall, (-6.0, -8.0, 0.4)], [np.log(200.0), np.log(20.0)])
    best = None
    for (scale_logit, offset, slope), first_log_concentration in starts:
        raw = {
            "scale_logit": torch.tensor(scale_logit, dtype=torch.float64),
            "offset": torch.tensor(offset, dtype=torch.float64),
            "slope": torch.tensor(slope, dtype=torch.float64),
            "first_u": torch.tensor(
                float(np.log(np.expm1(first_log_concentration - LOG_MIN_CONCENTRATION))), dtype=torch.float64
            ),
            # small decrements between bins
            "log_decrements": torch.full((num_bins - 1,), float(np.log(np.expm1(1e-3))), dtype=torch.float64),
        }
        for tensor in raw.values():
            tensor.requires_grad_(True)
        variables = [tensor for name, tensor in raw.items() if not (name == "slope" and k == NOT_IN_STR)]
        optimizer = torch.optim.LBFGS(variables, max_iter=MAX_ITERATIONS_PER_ROUND, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = -germline_log_likelihood(samples, parameters_from(raw))
            loss.backward()
            return loss

        # an L-BFGS round can wander into parameters whose likelihood is NaN, so each start keeps its best finite ones
        start_best = None
        for _ in range(MAX_ROUNDS):
            optimizer.step(closure)
            with torch.no_grad():
                loss = -germline_log_likelihood(samples, parameters_from(raw)).item()
            if not np.isfinite(loss):
                break
            improvement = np.inf if start_best is None else start_best[0] - loss
            if start_best is None or loss < start_best[0]:
                with torch.no_grad():
                    start_best = (loss, parameters_from(raw))
            if improvement < TOLERANCE:
                break
        if start_best is not None and (best is None or start_best[0] < best[0]):
            best = start_best
    assert best is not None, f"every fit of repeat class {k} diverged"
    return best[1]


def merge_bins(upper_bounds: Tensor, log_concentrations: Tensor, tolerance: float = MERGE_TOLERANCE):
    """
    Merge each bin into the previous one if their log concentrations differ by less than the tolerance, so that a run of
    nearly equal bins becomes one bin with the run's first concentration and last upper bound
    """
    kept_bounds, kept_logs = [upper_bounds[0].item()], [log_concentrations[0].item()]
    for bound, log_concentration in zip(upper_bounds[1:].tolist(), log_concentrations[1:].tolist()):
        if kept_logs[-1] - log_concentration < tolerance:
            kept_bounds[-1] = bound
        else:
            kept_bounds.append(bound)
            kept_logs.append(log_concentration)
    return torch.tensor(kept_bounds, dtype=torch.float64), torch.tensor(kept_logs, dtype=torch.float64)


def fit_indel_parameters(
    samples: list[GermlineIndels], initial: IndelParameters = None, min_indels_per_bin: int = MIN_INDELS_PER_BIN
) -> IndelParameters:
    """
    The indel parameters that maximize the germline log likelihood of the samples' indels, which must be germline,
    class by class.  Classes without indels keep the initial parameters, by default IndelParameters.default().  The
    fit, and the result, are in float64.
    """
    result = copy.deepcopy(IndelParameters.default() if initial is None else initial).double()
    for k in range(NUM_REPEAT_CLASSES):
        class_samples = [sample.subset(repeat_classes(sample.repeat_unit_lengths) == k) for sample in samples]
        class_samples = [sample for sample in class_samples if len(sample.depths) > 0]
        if class_samples:
            fitted = fit_repeat_class(class_samples, k, result, min_indels_per_bin)
            bounds, logs = merge_bins(fitted.concentration_upper_bounds()[k], fitted.log_concentrations()[k])
            result = with_class(
                result,
                k,
                fitted.shortfall_scales[k],
                fitted.shortfall_offsets[k],
                fitted.shortfall_slopes[k],
                bounds,
                logs,
            )
    return result
