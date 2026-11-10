"""
The germline model's parameters for indels, whose alt fractions behave differently from those of the SNPs that
segment_pileups fits its emission models to.

Reads of an allele in a short tandem repeat (STR) often slip by a repeat unit in sequencing, so that a read of a
germline indel's allele looks like another allele.  The fraction of reads with the indel then falls short of the
fraction of the sample's alleles that carry it, more so in longer STRs, and by different amounts at different sites, so
that indels' alt counts are much more dispersed than SNPs'.  Indels outside STRs are more dispersed than SNPs too.

Indels fall into repeat classes by the length of their STR's repeat unit: outside STRs, homopolymers, dinucleotide
repeats, and repeat units of three or more bases.  Given the length L in bases of the STR in the reference, and 0 outside
STRs:
    - an indel's expected alt fraction, after contamination, reference bias and sequencing error, is multiplied by
      1 - a sigmoid(b + c L), where each class has its own shortfall scale a, offset b and slope c.
    - the concentration of the beta binomial of its alt count, for hets and hom alts alike, is its class's step function
      of L, which doesn't increase with L: one concentration for each bin of lengths, whose upper bound is in the bin,
      the last bin's concentration holding beyond it.  It replaces the emission model's concentrations, rather than
      scaling them, so that it doesn't change with them.

The defaults were fit by indel_fitting.fit_indel_parameters to 13,359 indels that the matched normals confirm are
germline, alt allele fraction at least 0.25, in six SEQC2 exome tumors sequenced on HiSeq and NovaSeq.
"""

import torch
from torch import Tensor
from torch import nn

# repeat classes, by the length of the repeat unit of the STR that an indel is in
NOT_IN_STR, HOMOPOLYMER, DINUCLEOTIDE, LONGER_UNIT = 0, 1, 2, 3
NUM_REPEAT_CLASSES = 4
REPEAT_CLASS_NAMES = ("not in an STR", "homopolymer", "dinucleotide", "longer unit")

# the defaults, by repeat class: the shortfall scale, offset and slope, and the concentration bins' upper bounds and log
# concentrations.  Outside STRs the shortfall is 0.005 and the concentration 9.6.  In homopolymers the shortfall is 0.04
# at 10 bases, 0.21 at 15 and 0.45 at 20, and the concentration falls from 34 to 19 at 3-9 bases, 15 at 10, 5.4 at 13
# and 2.1 beyond 21.  Dinucleotide repeats fall short by at most 0.05, with concentration 420 at 2 bases and 11.5 from 6.
# Longer units barely fall short below 50 bases, with concentrations from 35 down to 2.2 beyond 48 bases.
DEFAULT_SHORTFALL_SCALES = (0.0078, 0.5511, 0.0478, 0.9)
DEFAULT_SHORTFALL_OFFSETS = (0.479, -6.615, -7.228, -6.790)
DEFAULT_SHORTFALL_SLOPES = (0.0, 0.4065, 0.6054, 0.0729)
DEFAULT_CONCENTRATION_UPPER_BOUNDS = ((0,), (1, 2, 9, 10, 11, 12, 13, 14, 15, 21, 27), (2, 4, 26), (7, 28, 36, 48, 100))
DEFAULT_LOG_CONCENTRATIONS = (
    (2.260,),
    (3.525, 3.303, 2.928, 2.713, 2.488, 2.351, 1.685, 1.185, 1.022, 0.933, 0.726),
    (6.044, 4.120, 2.446),
    (3.544, 3.024, 2.390, 1.211, 0.788),
)


def repeat_classes(repeat_unit_lengths: Tensor) -> Tensor:
    """each indel's repeat class: its STR's repeat unit length, at most LONGER_UNIT, or NOT_IN_STR (0) outside STRs"""
    return torch.clamp(repeat_unit_lengths, 0, LONGER_UNIT).long()


def step_function(classes: Tensor, lengths: Tensor, upper_bounds: list[Tensor], values: list[Tensor]) -> Tensor:
    """
    Each site's value of its class's step function of length: the value of the first bin whose upper bound is at least
    the length, or of the last bin beyond it.  Differentiable with respect to the values.
    """
    per_class = []
    for bounds, class_values in zip(upper_bounds, values):
        bins = torch.searchsorted(bounds, lengths.to(bounds.dtype).contiguous())
        per_class.append(class_values[torch.clamp(bins, max=len(bounds) - 1)])
    return torch.stack(per_class).gather(0, classes.unsqueeze(0)).squeeze(0)


class IndelParameters(nn.Module):
    """
    The shortfall of indels' alt fractions and their concentrations, by repeat class and STR length.  The tensors are
    buffers rather than parameters, since filter_variants doesn't fit them, but they may require gradients, which
    indel_fitting uses.  Values that aren't tensors get the default dtype, as the emission models' parameters do, and
    tensors keep theirs.
    """

    def __init__(
        self,
        shortfall_scales,
        shortfall_offsets,
        shortfall_slopes,
        concentration_upper_bounds: list,
        log_concentrations: list,
    ):
        super().__init__()

        def as_tensor(values) -> Tensor:
            return values if isinstance(values, Tensor) else torch.as_tensor(values, dtype=torch.get_default_dtype())

        self.register_buffer("shortfall_scales", as_tensor(shortfall_scales))
        self.register_buffer("shortfall_offsets", as_tensor(shortfall_offsets))
        self.register_buffer("shortfall_slopes", as_tensor(shortfall_slopes))
        assert all(
            len(t) == NUM_REPEAT_CLASSES for t in (self.shortfall_scales, self.shortfall_offsets, self.shortfall_slopes)
        )
        scales = self.shortfall_scales.detach()
        assert bool(torch.all((scales >= 0) & (scales < 1))), "a shortfall scale of 1 or more could leave no alt reads"
        assert len(concentration_upper_bounds) == len(log_concentrations) == NUM_REPEAT_CLASSES
        for k, (bounds, logs) in enumerate(zip(concentration_upper_bounds, log_concentrations)):
            bounds, logs = as_tensor(bounds), as_tensor(logs)
            assert len(bounds) == len(logs) > 0 and bool(torch.all(torch.diff(bounds) > 0)), "bins must increase"
            assert bool(torch.all(torch.diff(logs.detach()) <= 0)), "concentrations mustn't increase with STR length"
            self.register_buffer(f"concentration_upper_bounds_{k}", bounds)
            self.register_buffer(f"log_concentrations_{k}", logs)

    def concentration_upper_bounds(self) -> list[Tensor]:
        return [getattr(self, f"concentration_upper_bounds_{k}") for k in range(NUM_REPEAT_CLASSES)]

    def log_concentrations(self) -> list[Tensor]:
        return [getattr(self, f"log_concentrations_{k}") for k in range(NUM_REPEAT_CLASSES)]

    def alt_fraction_factors(self, repeat_unit_lengths: Tensor, repeat_lengths: Tensor) -> Tensor:
        """the factor 1 - a sigmoid(b + c L) of each indel's expected alt fraction"""
        classes = repeat_classes(repeat_unit_lengths)
        lengths = repeat_lengths.to(self.shortfall_scales.dtype)
        logits = self.shortfall_offsets[classes] + self.shortfall_slopes[classes] * lengths
        return 1 - self.shortfall_scales[classes] * torch.sigmoid(logits)

    def concentrations(self, repeat_unit_lengths: Tensor, repeat_lengths: Tensor) -> Tensor:
        """the concentration of each indel's beta binomial, for hets and hom alts alike"""
        classes = repeat_classes(repeat_unit_lengths)
        log_values = step_function(
            classes, repeat_lengths, self.concentration_upper_bounds(), self.log_concentrations()
        )
        return torch.exp(log_values)

    def state(self) -> dict:
        """everything needed to reconstruct it, as Python lists, which torch.load can read with weights_only"""
        return {
            "shortfall_scales": self.shortfall_scales.tolist(),
            "shortfall_offsets": self.shortfall_offsets.tolist(),
            "shortfall_slopes": self.shortfall_slopes.tolist(),
            "concentration_upper_bounds": [bounds.tolist() for bounds in self.concentration_upper_bounds()],
            "log_concentrations": [logs.tolist() for logs in self.log_concentrations()],
        }

    @classmethod
    def from_state(cls, state: dict) -> "IndelParameters":
        return cls(**state)

    @classmethod
    def default(cls) -> "IndelParameters":
        return cls(
            DEFAULT_SHORTFALL_SCALES,
            DEFAULT_SHORTFALL_OFFSETS,
            DEFAULT_SHORTFALL_SLOPES,
            [list(bounds) for bounds in DEFAULT_CONCENTRATION_UPPER_BOUNDS],
            [list(logs) for logs in DEFAULT_LOG_CONCENTRATIONS],
        )
