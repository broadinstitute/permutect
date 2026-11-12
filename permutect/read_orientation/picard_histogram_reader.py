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
"""

import io
import os
import tarfile
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from operator import itemgetter

import numpy as np

BASES = "ACGT"
ORIENTATIONS = ("F1R2", "F2R1")
CONTEXT_SIZE = 3
# every 3-base reference context, in lexicographic order
CONTEXTS = tuple(a + b + c for a in BASES for b in BASES for c in BASES)

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


@dataclass
class PicardHistograms:
    """
    The histograms of a Picard metrics file, which share their bins, and the values of its string headers, which in
    F1R2 counts name the sample
    """

    string_headers: list[str]
    bin_label: str
    bins_b: np.ndarray
    counts_by_label: dict[str, np.ndarray]


def read_picard_histograms(lines) -> PicardHistograms:
    """
    Read the histograms of a Picard metrics file, given as an iterable of its lines.  The file's histogram section is a
    table with a column of bins and a column of counts for each histogram, headed by the histograms' labels.
    """
    string_headers, header, rows = [], None, []
    previous_line, in_histograms = None, False
    for line in lines:
        line = line.rstrip("\n")
        if in_histograms:
            if not line:
                break
            if header is None:
                header = line.split("\t")
            else:
                rows.append(line.split("\t"))
        elif line.startswith(HISTOGRAM_SECTION_PREFIX):
            in_histograms = True
        elif previous_line == STRING_HEADER and line.startswith(HEADER_VALUE_PREFIX):
            string_headers.append(line[len(HEADER_VALUE_PREFIX) :])
        previous_line = line

    if header is None:
        raise ValueError("Picard metrics file without histograms")
    if any(len(row) != len(header) for row in rows):
        raise ValueError("Picard histogram rows with the wrong number of columns")
    # Picard formats the counts as decimals
    table = np.array(rows, dtype=np.float64).reshape(len(rows), len(header))
    counts_by_label = {label: np.rint(table[:, column]).astype(np.int64) for column, label in enumerate(header[1:], 1)}
    return PicardHistograms(string_headers, header[0], table[:, 0].astype(np.int64), counts_by_label)


@dataclass
class AltSites:
    """
    Sites with two or more alt reads, collapsed to their distinct combinations of alt base (an index into BASES), alt
    count, count of alt reads in F1R2 orientation and depth, with the number of sites of each.  The read orientation
    model depends on nothing else, and collapsing is exact.  There are never more combinations than sites, and far fewer
    at high depth, where most of these sites have a few alt reads from sequencing errors.
    """

    alt_bases_n: np.ndarray
    alt_counts_n: np.ndarray
    alt_f1r2_counts_n: np.ndarray
    depths_n: np.ndarray
    num_sites_n: np.ndarray

    @staticmethod
    def collapse(alt_bases_n, alt_counts_n, alt_f1r2_counts_n, depths_n, num_sites_n) -> "AltSites":
        """sites with the given columns, of which identical rows are combined, adding their numbers of sites"""
        columns_n4 = np.stack([alt_bases_n, alt_counts_n, alt_f1r2_counts_n, depths_n], axis=1).astype(np.int64)
        distinct_m4, inverse_n = np.unique(columns_n4.reshape(-1, 4), axis=0, return_inverse=True)
        num_sites_m = np.bincount(inverse_n.reshape(-1), weights=num_sites_n, minlength=len(distinct_m4))
        return AltSites(*distinct_m4.T, num_sites_m.astype(np.int64))

    @staticmethod
    def empty() -> "AltSites":
        return AltSites(*(np.zeros(0, dtype=np.int64) for _ in range(5)))

    def __add__(self, other: "AltSites") -> "AltSites":
        concatenated = (np.concatenate(pair) for pair in zip(self.columns(), other.columns()))
        return AltSites.collapse(*concatenated)

    def columns(self):
        return self.alt_bases_n, self.alt_counts_n, self.alt_f1r2_counts_n, self.depths_n, self.num_sites_n

    def num_sites(self) -> int:
        return int(self.num_sites_n.sum())


@dataclass
class F1R2Counts:
    """
    The F1R2 counts of one sample, summed over the inputs.  Each histogram has the number of sites of depth 1, 2, . . .
    max_depth, with sites of greater depth in the last bin.
    """

    sample: str
    max_depth: int
    # by reference context, the histogram of sites without alt reads
    ref_histograms: dict[str, np.ndarray]
    # by reference context, alt base and orientation of the alt read, the histogram of sites with one alt read
    alt_depth_one_histograms: dict[tuple[str, str, str], np.ndarray]
    # by reference context, the sites with two or more alt reads
    alt_sites: dict[str, AltSites]


@dataclass
class _SampleAccumulator:
    """the F1R2 counts of a sample so far, while reading the inputs"""

    num_ref_histogram_files: int = 0
    num_alt_histogram_files: int = 0
    bins_b: np.ndarray = None
    ref_histograms: dict = field(default_factory=dict)
    alt_depth_one_histograms: dict = field(default_factory=dict)
    alt_sites: dict = field(default_factory=lambda: defaultdict(AltSites.empty))

    def add_histograms(self, histograms: PicardHistograms, file_name: str, expected_labels, sums: dict):
        if set(histograms.counts_by_label) != set(expected_labels):
            raise ValueError(f"{file_name} does not have exactly the histograms of F1R2 counts")
        if self.bins_b is None:
            self.bins_b = histograms.bins_b
        elif not np.array_equal(self.bins_b, histograms.bins_b):
            raise ValueError(f"{file_name} has different depth bins from other F1R2 counts of its sample")
        for label, counts_b in histograms.counts_by_label.items():
            sums[label] = sums[label] + counts_b if label in sums else counts_b


def _alt_histogram_labels():
    return [
        "_".join((context, alt_base, orientation))
        for context in CONTEXTS
        for alt_base in BASES
        if alt_base != context[1]
        for orientation in ORIENTATIONS
    ]


def read_f1r2_counts(tar_gz_paths) -> dict[str, F1R2Counts]:
    """
    The F1R2 counts of each sample in one or more tar.gz files from Mutect2 or CollectF1R2Counts, eg from the scatters
    of one run, summed over the files.  The files are read in order, without extracting them to disk.
    """
    tar_gz_paths = list(tar_gz_paths)
    accumulators = defaultdict(_SampleAccumulator)
    alt_histogram_labels = _alt_histogram_labels()
    for path in tar_gz_paths:
        with tarfile.open(path, mode="r:gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                name = os.path.basename(member.name)
                with io.TextIOWrapper(tar.extractfile(member), encoding="utf-8") as file:
                    if name.endswith(REF_HISTOGRAM_EXTENSION):
                        histograms = read_picard_histograms(file)
                        accumulator = accumulators[_histogram_sample(histograms, path, name)]
                        accumulator.add_histograms(histograms, name, CONTEXTS, accumulator.ref_histograms)
                        accumulator.num_ref_histogram_files += 1
                    elif name.endswith(ALT_HISTOGRAM_EXTENSION):
                        histograms = read_picard_histograms(file)
                        accumulator = accumulators[_histogram_sample(histograms, path, name)]
                        sums = accumulator.alt_depth_one_histograms
                        accumulator.add_histograms(histograms, name, alt_histogram_labels, sums)
                        accumulator.num_alt_histogram_files += 1
                    elif name.endswith(ALT_TABLE_EXTENSION):
                        sample, context, alt_sites = read_alt_table(file, name)
                        accumulator = accumulators[sample]
                        accumulator.alt_sites[context] = accumulator.alt_sites[context] + alt_sites

    result = {}
    for sample, accumulator in accumulators.items():
        if (
            accumulator.num_ref_histogram_files == 0
            or accumulator.num_ref_histogram_files != accumulator.num_alt_histogram_files
        ):
            raise ValueError(
                f"Sample {sample} of the F1R2 counts in {tar_gz_paths} does not have one ref histogram file and "
                "one alt histogram file in each input that has it"
            )
        bins_b = accumulator.bins_b
        max_depth = len(bins_b)
        if not np.array_equal(bins_b, np.arange(1, max_depth + 1)):
            raise ValueError(f"The F1R2 histograms of sample {sample} do not have a bin for each depth from 1 up")
        alt_depth_one_histograms = {
            tuple(label.split("_")): counts_b for label, counts_b in accumulator.alt_depth_one_histograms.items()
        }
        alt_sites = {context: accumulator.alt_sites.get(context, AltSites.empty()) for context in CONTEXTS}
        result[sample] = F1R2Counts(sample, max_depth, accumulator.ref_histograms, alt_depth_one_histograms, alt_sites)
    return result


def _histogram_sample(histograms: PicardHistograms, path, name: str) -> str:
    # GATK records the sample of a histogram file as its only string header
    if len(histograms.string_headers) != 1:
        raise ValueError(f"{name} in {path} should have one string header, which names its sample")
    return histograms.string_headers[0]


def read_alt_table(lines, name: str) -> tuple[str, str, AltSites]:
    """
    The sample, reference context and sites of an alt table <sample>.<context>.alt_table, given as an iterable of its
    lines and its file name
    """
    context = name[: -len(ALT_TABLE_EXTENSION)].rsplit(".", 1)[-1]
    if context not in CONTEXTS:
        raise ValueError(
            f"{name} is not named <sample>.<reference context>{ALT_TABLE_EXTENSION}.  Older versions of GATK made one "
            "alt table per sample; please regenerate the F1R2 counts with a current Mutect2 or CollectF1R2Counts."
        )

    lines = iter(lines)
    sample, header = None, None
    for line in lines:
        line = line.rstrip("\n")
        if line.startswith(SAMPLE_METADATA_PREFIX):
            sample = line[len(SAMPLE_METADATA_PREFIX) :]
        elif not line.startswith("#"):
            header = line.split("\t")
            break
    if sample is None or header is None or not set(ALT_TABLE_COLUMNS) <= set(header):
        raise ValueError(f"{name} is not an alt table with a sample and the columns {ALT_TABLE_COLUMNS}")

    # count the distinct combinations of the columns as strings, which is fast and takes memory only for the
    # combinations, then convert them
    columns = itemgetter(*(header.index(column) for column in ALT_TABLE_COLUMNS))
    counts = Counter(columns(line.rstrip("\n").split("\t")) for line in lines)
    if any(row_context != context for row_context, *_ in counts):
        raise ValueError(f"{name} has sites of a reference context other than {context}")
    if not counts:
        return sample, context, AltSites.empty()
    _, alt_bases, alt_counts, alt_f1r2_counts, depths = zip(*counts)
    alt_bases_n = np.array([BASES.index(alt_base) for alt_base in alt_bases], dtype=np.int64)
    numeric = (np.array(column, dtype=np.int64) for column in (alt_counts, alt_f1r2_counts, depths))
    alt_sites = AltSites.collapse(alt_bases_n, *numeric, np.array(list(counts.values()), dtype=np.int64))
    return sample, context, alt_sites
