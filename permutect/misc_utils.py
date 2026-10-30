import time
from typing import Iterator
from typing import Set

import cyvcf2
import numpy as np
import psutil
import torch
from torch import Tensor
from torch import nn
from torch.nn import Parameter
from tqdm.autonotebook import tqdm

from permutect.data.datum import DEFAULT_CPU_FLOAT
from permutect.data.datum import DEFAULT_GPU_FLOAT
from permutect.data.datum import Data
from permutect.data.datum import Datum
from permutect.utils.allele_utils import trim_alleles_on_right
from permutect.utils.allele_utils import truncate_bases_if_necessary


def report_memory_usage(message: str = ""):
    print(f"{message}  Memory usage: {psutil.virtual_memory().percent:.1f}%")


class ConsistentValue:
    """
    Tracks a value that once initialized, is consistent among eg all members of a dataset.  For example, all tensors
    must have the same number of columns.
    """

    def __init__(self, value=None):
        self.value = value

    def check(self, value):
        if self.value is None:
            self.value = value
        else:
            assert self.value == value, "inconsistent values"


def gpu_if_available(exploit_mps=False) -> torch.device:
    if torch.cuda.is_available():
        d = "cuda"
    elif exploit_mps and torch.mps.is_available():
        d = "mps"
    else:
        d = "cpu"
    return torch.device(d)


def resolve_device(device=None) -> torch.device:
    """
    The given device (a torch.device or a string such as "cpu") as a torch.device, or by default a GPU if available.
    Use device=None as a default argument and resolve it with this when called, because a default of
    gpu_if_available() is evaluated only once, at import.
    """
    return gpu_if_available() if device is None else torch.device(device)


def report_device(device: torch.device):
    """
    Print the device a tool runs on, so that a job with a GPU that runs on the CPU instead is visible in its log.  That
    happens, for example, when the GPU driver is older than the CUDA version of PyTorch requires.
    """
    if device.type == "cuda":
        print(f"Running on GPU {torch.cuda.get_device_name(device)} with CUDA {torch.version.cuda}.")
    elif torch.version.cuda is None:
        print(f"Running on {device.type}.")
    else:
        print(
            f"Running on {device.type}.  This PyTorch was built for CUDA {torch.version.cuda}, but CUDA is unavailable: "
            "there is no GPU, or its driver is too old for this CUDA version."
        )


def float_dtype(device=None) -> torch.dtype:
    """
    The floating-point dtype of models and of batches sent to them on a device
    """
    return DEFAULT_CPU_FLOAT if resolve_device(device).type == "cpu" else DEFAULT_GPU_FLOAT


def freeze(parameters):
    for parameter in parameters:
        parameter.requires_grad = False


def unfreeze(parameters):
    for parameter in parameters:
        if parameter.dtype.is_floating_point:  # an integer parameter isn't trainable by gradient descent
            parameter.requires_grad = True


class StreamingAverage:
    def __init__(self):
        self._count = 0.0
        self._sum = 0.0

    def is_empty(self) -> bool:
        return self._count == 0.0

    def get(self) -> float:
        return self._sum / (self._count + 0.0001)

    def record(self, value: float, weight: float = 1):
        self._count += weight
        self._sum += value * weight

    def record_sum(self, value_sum: float | Tensor, count):
        self._count += count
        self._sum += value_sum

    # record only values masked as true
    def record_with_mask(self, values: Tensor, mask: Tensor):
        self._count += torch.sum(mask).item()
        self._sum += torch.sum(values * mask).item()

    # record values with different weights
    # values and mask should live on same device as self._sum
    def record_with_weights(self, values: Tensor, weights: Tensor):
        self._count += torch.sum(weights).item()
        self._sum += torch.sum(values * weights).item()


class Timer:
    def __init__(self, message: str = None):
        if message is not None:
            print(message)
        self._start_time = time.perf_counter()

    def report(self, message: str = ""):
        elapsed_time = time.perf_counter() - self._start_time
        print(f"{message}: {elapsed_time:0.4f} seconds")


def backpropagate(optimizer: torch.optim.Optimizer, loss: Tensor, params_to_clip: Iterator[Parameter] = []):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0)
    optimizer.step()


def get_first_numeric_element(variant, key):
    tuple_or_scalar = variant.INFO[key]
    return tuple_or_scalar[0] if type(tuple_or_scalar) is tuple else tuple_or_scalar


def encode(contig: str, position: int, ref: str, alt: str):
    # contig is a name such as chr1; encode_datum converts a Datum's contig index to its name
    # both alleles are needed to distinguish eg AT>A from ATT>A.  They are truncated like the alleles stored in a Datum.
    trimmed_ref, trimmed_alt = trim_alleles_on_right(ref, alt)
    encoded_alleles = truncate_bases_if_necessary(trimmed_ref) + ":" + truncate_bases_if_necessary(trimmed_alt)
    return contig + ":" + str(position) + ":" + encoded_alleles


def encode_datum(datum: Datum, contig_index_to_name_map):
    contig_name = contig_index_to_name_map[datum.get(Data.CONTIG)]
    return encode(contig_name, datum.get(Data.POSITION), datum.get_ref_allele(), datum.get_alt_allele())


def encode_variant(v: cyvcf2.Variant, zero_based=False):
    alt = v.ALT[0]  # TODO: we're assuming biallelic
    ref = v.REF
    start = (v.start + 1) if zero_based else v.start
    return encode(v.CHROM, start, ref, alt)


def overlapping_filters(v: cyvcf2.Variant, filters_set: Set[str]) -> Set[str]:
    return set([]) if v.FILTER is None else set(v.FILTER.split(";")).intersection(filters_set)


def declare_format_dp(vcf: cyvcf2.VCF):
    # GATK SelectVariants -DGA DP, as in the SplitMultiallelics task of permutect.wdl, removes the FORMAT/DP header line
    # but not the DP values.  htslib flags the undeclared tag in each record, and cyvcf2 refuses to write such records.
    # This must be called before reading any records, and it does nothing if the header already declares DP.
    vcf.add_format_to_header({"ID": "DP", "Description": "depth", "Type": "Integer", "Number": "1"})


def get_tumor_sample_index(vcf: cyvcf2.VCF) -> int:
    # This is specific to Mutect2, which outputs a line:
    # ##tumor_sample=___________
    tumor_sample_name = ""
    all_samples = []
    for header_line in vcf.raw_header.split("\n"):
        if header_line.startswith("##tumor_sample"):
            tumor_sample_name = header_line.split("=")[-1]
            print(tumor_sample_name)
        elif header_line.startswith("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"):
            prefix = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            all_samples = header_line[len(prefix) :].split()
    tumor_sample_index = 0
    for n, sample in enumerate(all_samples):
        if sample == tumor_sample_name:
            print(f"tumor sample found in VCF header: name {sample}, index {n}")
            tumor_sample_index = n
    return tumor_sample_index


def tumor_depths(v: cyvcf2.Variant, tumor_sample_index: int) -> tuple[int, int]:
    # the tumor's ref depth and its total depth of all alt alleles
    tumor_allele_depths = v.format("AD")[tumor_sample_index]
    return tumor_allele_depths[0], np.sum(tumor_allele_depths[1:])


class VcfEditor:
    """
    Copy a VCF record by record, editing records on the way, to an output VCF whose header is that of the input plus the
    given INFO and FILTER lines.  Iterating over the editor yields the input records in order, and each record, with any
    edits, is written when the next one is requested.  Both files are closed after the last record.

    :param info_fields: dict from INFO ID to (Type, Number, Description)
    :param filters: dict from FILTER ID to Description
    """

    def __init__(self, input_vcf, output_vcf, info_fields: dict = None, filters: dict = None):
        self.vcf = cyvcf2.VCF(input_vcf)
        self.tumor_sample_index = get_tumor_sample_index(self.vcf)

        declare_format_dp(self.vcf)
        for info_id, (info_type, number, description) in (info_fields or {}).items():
            info_line = {"ID": info_id, "Description": description, "Type": info_type, "Number": str(number)}
            self.vcf.add_info_to_header(info_line)
        for filter_id, description in (filters or {}).items():
            self.vcf.add_filter_to_header({"ID": filter_id, "Description": description})
        self.writer = cyvcf2.Writer(output_vcf, self.vcf)  # the input VCF is a template for the header

    def __iter__(self) -> Iterator[cyvcf2.Variant]:
        try:
            for v in tqdm(self.vcf, mininterval=60):
                yield v
                self.writer.write_record(v)
        finally:
            self.writer.close()
            self.vcf.close()


def check_for_nan(model: torch.nn.Module):
    nan_found = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"Invalid gradient (NaN or Inf) found in parameter: {name}")
                nan_found = True
    assert not nan_found
