import os
import random
import time
from datetime import timedelta
from typing import Iterator
from typing import Set

import cyvcf2
import numpy as np
import psutil
import torch
import torch.distributed as dist
from torch import Tensor
from torch import nn
from torch.nn import Parameter
from tqdm.autonotebook import tqdm

from permutect.data.datum import DEFAULT_CPU_FLOAT
from permutect.data.datum import DEFAULT_GPU_FLOAT
from permutect.utils.allele_utils import bases_as_base5_int
from permutect.utils.allele_utils import trim_alleles_on_right
from permutect.utils.allele_utils import truncate_bases_if_necessary


def report_memory_usage(message: str = ""):
    print(f"{message}  Memory usage: {psutil.virtual_memory().percent:.1f}%")


def seed_random_number_generators(seed: int = 0):
    """
    Seed the global random number generators of Python, numpy and torch, including torch's of every GPU, so that a tool
    makes the same random choices in every run, such as the order in which a ReadsDataset yields its data.  A DataLoader
    seeds its workers' generators from torch's, so this seeds theirs too.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def distributed_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def distributed_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return distributed_rank() == 0


def init_distributed() -> torch.device:
    """
    Under torchrun with more than one process, join the process group and return this process's own GPU.  Otherwise
    return gpu_if_available() for training in a single process.
    """
    if int(os.environ.get("WORLD_SIZE", "1")) == 1:
        return gpu_if_available()
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    # the other processes wait in their next collective while process 0 makes TensorBoard plots, and NCCL's default
    # timeout is 10 minutes
    dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo", timeout=timedelta(minutes=30))
    return device


def all_reduce_sum_(tensor: Tensor) -> Tensor:
    """
    In distributed training, sum a tensor over processes in place.  In a single process this does nothing.
    """
    if is_distributed():
        dist.all_reduce(tensor)
    return tensor


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
    tuple_or_scalar = variant.INFO.get(key)
    if tuple_or_scalar is None:
        raise ValueError(f"the variant at {variant.CHROM}:{variant.POS} has no {key} INFO field")
    return tuple_or_scalar[0] if type(tuple_or_scalar) is tuple else tuple_or_scalar


def str_repeat_lengths(variant) -> tuple[int, int]:
    """
    The lengths in bases of the repeat unit of the short tandem repeat (STR) that a variant is in and of the STR in the
    reference, from Mutect2's RU (repeat unit) and RPA (repeats per allele, the reference's first) annotations, or
    (0, 0) for a variant outside an STR, which has no RU
    """
    repeat_unit = variant.INFO.get("RU")
    if not repeat_unit:
        return 0, 0
    return len(repeat_unit), len(repeat_unit) * int(get_first_numeric_element(variant, "RPA"))


def str_repeat_count_change(variant) -> int:
    """
    The change from the reference to the first alt allele in the number of repeat units of the short tandem repeat (STR)
    that a variant is in, from Mutect2's RPA (repeats per allele) annotation, eg -1 for the deletion of one repeat unit,
    or 0 for a variant outside an STR, which has no RU
    """
    repeats_per_allele = variant.INFO.get("RPA")
    if not variant.INFO.get("RU") or not isinstance(repeats_per_allele, tuple) or len(repeats_per_allele) < 2:
        return 0
    return int(repeats_per_allele[1]) - int(repeats_per_allele[0])


def allele_key(ref: str, alt: str) -> int | None:
    """
    A variant's alleles as one integer, for sorting and matching variants by position and alleles: the base-5 integers
    of the ref and alt alleles, trimmed on the right and truncated to the 13 bases that a Datum stores, side by side.
    Both alleles are needed to distinguish eg AT>A from ATT>A.  Trimming comes first, since truncating first and then
    trimming can give different alleles.  The key is None for an allele with a base other than A, C, G or T, such as N,
    which a Datum's alleles, decoded from base 5, never have.
    """
    trimmed_ref, trimmed_alt = trim_alleles_on_right(ref, alt)
    ref_bases, alt_bases = truncate_bases_if_necessary(trimmed_ref), truncate_bases_if_necessary(trimmed_alt)
    if not set(ref_bases + alt_bases) <= set("ACGT"):
        return None
    # at most 13 bases, whose base-5 digits are 1 to 4, make an integer below 5^13 < 2^31
    return (bases_as_base5_int(ref_bases) << 32) | bases_as_base5_int(alt_bases)


def vcf_contig_ranks(input_vcf, contig_index_to_name_map) -> dict[str, int]:
    """
    The rank of each contig in the order of a VCF's records: that of the header's contig lines, or, for a header without
    them, that of the contigs table's indices
    """
    vcf = cyvcf2.VCF(input_vcf)
    names = list(vcf.seqnames) or [name for _, name in sorted(contig_index_to_name_map.items())]
    vcf.close()
    return {name: rank for rank, name in enumerate(names)}


def position_key(contig_rank: int, position: int) -> int:
    """
    A variant's contig rank and position as one integer, which sorts like the records of a VCF sorted by position
    """
    return (contig_rank << 32) | position


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


def tumor_alt_orientation_counts(v: cyvcf2.Variant, tumor_sample_index: int) -> tuple[float, float]:
    """
    The tumor's counts of reads of the first alt allele in F1R2 and in F2R1 orientation, from Mutect2's F1R2 and F2R1
    FORMAT annotations, each NaN if the record lacks it or its value is missing
    """
    return tumor_orientation_counts(v, tumor_sample_index, allele=1)


def tumor_ref_orientation_counts(v: cyvcf2.Variant, tumor_sample_index: int) -> tuple[float, float]:
    """likewise the tumor's counts of ref reads in F1R2 and in F2R1 orientation"""
    return tumor_orientation_counts(v, tumor_sample_index, allele=0)


def tumor_orientation_counts(v: cyvcf2.Variant, tumor_sample_index: int, allele: int) -> tuple[float, float]:
    """the tumor's counts of reads of an allele, 0 for the ref, in F1R2 and in F2R1 orientation"""
    counts = []
    for key in ("F1R2", "F2R1"):
        try:
            values = v.format(key)  # by sample, then by allele, ref first, or None if the record lacks the annotation
        except KeyError:  # the header doesn't declare the annotation
            values = None
        count = np.nan if values is None else values[tumor_sample_index][allele]
        # cyvcf2 gives a missing integer as htslib's most negative integer
        counts.append(float(count) if count >= 0 else np.nan)
    return counts[0], counts[1]


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
    params = model.named_parameters()
    invalid = [name for name, param in params if param.grad is not None and not torch.isfinite(param.grad).all()]
    for name in invalid:
        print(f"Invalid gradient (NaN or Inf) found in parameter: {name}")
    assert not invalid
