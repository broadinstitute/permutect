"""
Reading the F1R2 counts that GATK Mutect2 (with --f1r2-tar-gz) and CollectF1R2Counts write: a tar.gz holding, for each
sample,

- <sample>.ref_histogram: a Picard metrics file of histograms of the depths of sites without alt reads, one for each
  3-base reference context, eg ACG
- <sample>.alt_histogram: histograms of the depths of sites with one alt read, one for each reference context, alt base
  and orientation of the alt read, labeled eg ACG_T_F1R2
- <sample>.<context>.alt_table: a GATK table of the sites of one reference context with two or more alt reads of their
  most common alt base: their ref and alt counts, the counts of ref and alt reads in F1R2 orientation, the depth, which
  is the ref count plus the alt count, and the alt base

where <sample> is URL-encoded.  The histograms have a bin for each depth 1, 2, . . . up to a maximum depth (Mutect2's
--f1r2-max-depth, 200 by default), whose bin also counts sites of greater depth.  Depths in the alt tables are exact.

All three become the same thing here: for each sample and reference context, the counts of distinct Sites.
"""

import io
import itertools
import os
import tarfile
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
from operator import itemgetter

import numpy as np
import torch
from torch import Tensor

# in this order the complement of the base with index b has index 3 - b
BASES = "ACGT"
ORIENTATIONS = ("F1R2", "F2R1")
# every 3-base reference context, in lexicographic order
CONTEXTS = tuple(a + b + c for a in BASES for b in BASES for c in BASES)
# the labels of the histograms of sites with one alt read
ALT_HISTOGRAM_LABELS = tuple(
    f"{context}_{alt_base}_{orientation}"
    for context in CONTEXTS
    for alt_base in BASES
    if alt_base != context[1]
    for orientation in ORIENTATIONS
)

REF_HISTOGRAM_EXTENSION = ".ref_histogram"
ALT_HISTOGRAM_EXTENSION = ".alt_histogram"
ALT_TABLE_EXTENSION = ".alt_table"

# lines of a Picard (htsjdk) metrics file
STRING_HEADER = "## htsjdk.samtools.metrics.StringHeader"
HEADER_VALUE_PREFIX = "# "
HISTOGRAM_SECTION_PREFIX = "## HISTOGRAM"

# the metadata line of a GATK table that names its sample, eg #<METADATA>SAMPLE=NA12878
SAMPLE_METADATA_PREFIX = "#<METADATA>SAMPLE="
# columns of an alt table that the read orientation model uses
ALT_TABLE_COLUMNS = ("context", "alt", "alt_count", "alt_f1r2", "depth")


@dataclass(frozen=True)
class Sites:
    """
    Sites of a reference context as their distinct combinations of alt base (an index into BASES), alt count, count of
    alt reads in F1R2 orientation and depth, the columns of keys_n4, with the number of sites of each.  A site without
    alt reads has the ref base as its alt base.  The read orientation model depends on nothing else, so this is exact,
    and there are never more combinations than sites.  The constructor takes keys that are already distinct, with
    positive numbers of sites.
    """

    keys_n4: Tensor
    counts_n: Tensor

    @staticmethod
    def of(keys_n4, counts_n) -> "Sites":
        """
        Sites with the given keys and numbers of sites, in lexicographic order of key, combining identical keys and
        dropping keys without sites
        """
        keys_n4 = torch.as_tensor(keys_n4, dtype=torch.int64).reshape(-1, 4)
        distinct_m4, inverse_n = torch.unique(keys_n4, dim=0, return_inverse=True)
        counts_m = torch.zeros(len(distinct_m4), dtype=torch.int64)
        counts_m.index_add_(0, inverse_n, torch.as_tensor(counts_n, dtype=torch.int64))
        return Sites(distinct_m4[counts_m > 0], counts_m[counts_m > 0])

    @staticmethod
    def empty() -> "Sites":
        return Sites(torch.zeros(0, 4, dtype=torch.int64), torch.zeros(0, dtype=torch.int64))

    @staticmethod
    def concatenate(sites) -> "Sites":
        """the sites of all the given Sites, combining identical keys"""
        sites = [Sites.empty(), *sites]
        return Sites.of(torch.cat([s.keys_n4 for s in sites]), torch.cat([s.counts_n for s in sites]))

    def __add__(self, other: "Sites") -> "Sites":
        return Sites.concatenate([self, other])

    @property
    def alt_bases_n(self) -> Tensor:
        return self.keys_n4[:, 0]

    @property
    def alt_counts_n(self) -> Tensor:
        return self.keys_n4[:, 1]

    @property
    def alt_f1r2_counts_n(self) -> Tensor:
        return self.keys_n4[:, 2]

    @property
    def depths_n(self) -> Tensor:
        return self.keys_n4[:, 3]

    def num_sites(self, selected_n: Tensor | None = None) -> int:
        """the number of sites, or of the selected ones"""
        return int(self.counts_n.sum() if selected_n is None else self.counts_n[selected_n].sum())

    def reverse_complement(self) -> "Sites":
        """
        Sites of the reverse complement of a context in the orientation of the context: their alt bases are
        complementary, and the F1R2 reads of one are the F2R1 reads of the other
        """
        alt_bases_n, alt_counts_n, alt_f1r2_counts_n, depths_n = self.keys_n4.unbind(dim=-1)
        keys_n4 = torch.stack([3 - alt_bases_n, alt_counts_n, alt_counts_n - alt_f1r2_counts_n, depths_n], dim=-1)
        return Sites.of(keys_n4, self.counts_n)


@dataclass
class F1R2Counts:
    """
    The F1R2 counts of one sample, summed over the inputs: the sites of each reference context.  Sites from the
    histograms that are deeper than max_depth count as max_depth.
    """

    sample: str
    max_depth: int
    sites: dict[str, Sites]


@dataclass
class PicardHistograms:
    """
    The histograms of a Picard metrics file, which share their bins, by label, and the values of its string headers,
    which in F1R2 counts name the sample
    """

    string_headers: list[str]
    labels: list[str]
    bins_b: Tensor
    counts_bl: Tensor


def read_picard_histograms(lines) -> PicardHistograms:
    """
    Read the histograms of a Picard metrics file, given as an iterable of its lines.  After a line "## HISTOGRAM", the
    file has a table with a column of bins and a column of counts for each histogram, headed by the histograms' labels.
    """
    lines = [line.rstrip("\n") for line in lines]
    string_headers = [
        value.removeprefix(HEADER_VALUE_PREFIX) for header, value in zip(lines, lines[1:]) if header == STRING_HEADER
    ]
    start = next((n for n, line in enumerate(lines) if line.startswith(HISTOGRAM_SECTION_PREFIX)), None)
    if start is None:
        raise ValueError("Picard metrics file without histograms")
    header, *rows = itertools.takewhile(bool, lines[start + 1 :])
    # Picard formats the counts as decimals
    table_bl = torch.from_numpy(np.loadtxt(rows, delimiter="\t", ndmin=2)).round().long()
    if table_bl.shape[-1] != len(header.split("\t")):
        raise ValueError("Picard histogram rows with the wrong number of columns")
    return PicardHistograms(string_headers, header.split("\t")[1:], table_bl[:, 0], table_bl[:, 1:])


def histogram_sites(histograms: PicardHistograms):
    """
    The reference context and sites of each F1R2 histogram: the depths of sites without alt reads, labeled by context,
    eg ACG, or with one alt read, labeled eg ACG_T_F1R2
    """
    for label, counts_b in zip(histograms.labels, histograms.counts_bl.T):
        context, *alt = label.split("_")
        alt_base, alt_count, alt_f1r2_count = (alt[0], 1, int(alt[1] == "F1R2")) if alt else (context[1], 0, 0)
        keys_b3 = torch.tensor([BASES.index(alt_base), alt_count, alt_f1r2_count]).expand(len(counts_b), 3)
        keys_b4, nonzero_b = torch.column_stack([keys_b3, histograms.bins_b]), counts_b > 0
        yield context, Sites(keys_b4[nonzero_b], counts_b[nonzero_b])


def read_alt_table(lines, name: str) -> tuple[str, str, Sites]:
    """
    The sample, reference context and sites of an alt table <sample>.<context>.alt_table, given as an iterable of its
    lines and its file name
    """
    context = name.removesuffix(ALT_TABLE_EXTENSION).rsplit(".", 1)[-1]
    if context not in CONTEXTS:
        raise ValueError(
            f"{name} is not named <sample>.<reference context>{ALT_TABLE_EXTENSION}.  Older versions of GATK made one "
            "alt table per sample; please regenerate the F1R2 counts with a current Mutect2 or CollectF1R2Counts."
        )

    lines = (line.rstrip("\n") for line in lines)
    sample, header = None, None
    for line in lines:
        if line.startswith(SAMPLE_METADATA_PREFIX):
            sample = line.removeprefix(SAMPLE_METADATA_PREFIX)
        elif not line.startswith("#"):
            header = line.split("\t")
            break
    if sample is None or header is None or not set(ALT_TABLE_COLUMNS) <= set(header):
        raise ValueError(f"{name} is not an alt table with a sample and the columns {ALT_TABLE_COLUMNS}")

    # count the distinct rows of the columns as strings, which is fast and takes memory only for the distinct rows
    columns = itemgetter(*map(header.index, ALT_TABLE_COLUMNS))
    counts = Counter(columns(line.split("\t")) for line in lines)
    if any(row_context != context for row_context, *_ in counts):
        raise ValueError(f"{name} has sites of a reference context other than {context}")
    keys = [
        [BASES.index(alt), int(alt_count), int(alt_f1r2), int(depth)] for _, alt, alt_count, alt_f1r2, depth in counts
    ]
    keys_n4 = torch.tensor(keys, dtype=torch.int64).reshape(-1, 4)
    return sample, context, Sites(keys_n4, torch.tensor(list(counts.values()), dtype=torch.int64))


def read_f1r2_counts(tar_gz_paths) -> dict[str, F1R2Counts]:
    """
    The F1R2 counts of each sample in one or more tar.gz files from Mutect2 or CollectF1R2Counts, eg from the scatters
    of one run, summed over the files.  The files are read in order, without extracting them to disk.
    """
    tar_gz_paths = list(tar_gz_paths)
    sites = defaultdict(lambda: defaultdict(list))  # by sample, then context, the Sites from each file
    num_histogram_files = Counter()  # by sample and extension
    max_depths = defaultdict(set)  # by sample
    for path in tar_gz_paths:
        for name, file in _text_files(path):
            extension = os.path.splitext(name)[1]
            if extension == ALT_TABLE_EXTENSION:
                sample, context, table_sites = read_alt_table(file, name)
                sites[sample][context].append(table_sites)
            elif extension in (REF_HISTOGRAM_EXTENSION, ALT_HISTOGRAM_EXTENSION):
                histograms = read_picard_histograms(file)
                if len(histograms.string_headers) != 1:
                    raise ValueError(f"{name} in {path} should have one string header, which names its sample")
                expected_labels = CONTEXTS if extension == REF_HISTOGRAM_EXTENSION else ALT_HISTOGRAM_LABELS
                if set(histograms.labels) != set(expected_labels):
                    raise ValueError(f"{name} in {path} does not have exactly the histograms of F1R2 counts")
                if not torch.equal(histograms.bins_b, torch.arange(1, len(histograms.bins_b) + 1)):
                    raise ValueError(f"{name} in {path} does not have a bin for each depth from 1 up")

                sample = histograms.string_headers[0]
                num_histogram_files[sample, extension] += 1
                max_depths[sample].add(len(histograms.bins_b))
                for context, histogram in histogram_sites(histograms):
                    sites[sample][context].append(histogram)

    result = {}
    for sample, sites_by_context in sites.items():
        num_ref_files, num_alt_files = (
            num_histogram_files[sample, ext] for ext in (REF_HISTOGRAM_EXTENSION, ALT_HISTOGRAM_EXTENSION)
        )
        if num_ref_files == 0 or num_ref_files != num_alt_files:
            raise ValueError(
                f"Sample {sample} of the F1R2 counts in {tar_gz_paths} does not have one ref histogram file and "
                "one alt histogram file in each input that has it"
            )
        if len(max_depths[sample]) != 1:
            raise ValueError(f"The F1R2 histogram files of sample {sample} have different depth bins")
        (max_depth,) = max_depths[sample]
        sites_by_context = {context: Sites.concatenate(sites_by_context[context]) for context in CONTEXTS}
        result[sample] = F1R2Counts(sample, max_depth, sites_by_context)
    return result


def _text_files(tar_gz_path):
    """the name and text of each file in a tar.gz, in order"""
    with tarfile.open(tar_gz_path, mode="r:gz") as tar:
        for member in tar:
            if member.isfile():
                with io.TextIOWrapper(tar.extractfile(member), encoding="utf-8") as file:
                    yield os.path.basename(member.name), file
