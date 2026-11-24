"""
Reading the F1R2 counts that GATK Mutect2 (with --f1r2-tar-gz) and CollectF1R2Counts write: a tar.gz holding, for each
sample, GATK tables (see permutect.utils.gatk_table) of

- <sample>.ref_histogram: the depths of sites without alt reads, by block of the genome and 3-base reference context: a
  row for each 1 Mb block and context, eg ACG, with sites, and a column of the number of sites of each depth
- <sample>.alt_histogram: likewise for sites with one alt read, with a row for each block, context, and alt base and
  orientation of the alt read
- <sample>.<context>.alt_table: the sites of one reference context with two or more alt reads of their most common alt
  base, a row for each: their ref and alt counts, the counts of ref and alt reads in F1R2 orientation, the alt base, and
  their contig and position

where <sample> is URL-encoded.  The depths of the histograms are those of their columns, eg 1, 2, . . . 24, 30, 40, . . .
200: Mutect2 keeps depths up to 24 exact and rounds greater depths to multiples of 10, and the column of the maximum
depth (Mutect2's --f1r2-max-depth, 200 by default) also counts sites of greater depth.  The depth of a site in an alt
table is its ref count plus its alt count, exactly.

All three become the same thing here: for each sample and reference context, the counts of distinct Sites.  Given a
segmentation of the sample by minor allele fraction (MAF), the positions of the alt tables' sites and the blocks of the
histograms give each site its MAF mixture (see permutect.read_orientation.maf_mixtures).  Each alt table site keeps the
F1R2 count of its ref reads, so that its number of F1R2 reads, ref and alt, is known, and the alt tables' sites' depths
and numbers of F1R2 reads are kept together too, for the sample's capture bias (see
permutect.read_orientation.capture_bias).
"""

import io
import os
import tarfile
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from permutect.read_orientation.maf_mixtures import BALANCED_MIXTURE
from permutect.read_orientation.maf_mixtures import MafMixtures
from permutect.read_orientation.maf_mixtures import balanced_mixtures_mg
from permutect.segmentation.germline_model import SampleSegmentation
from permutect.utils.gatk_table import read_gatk_table

# in this order the complement of the base with index b has index 3 - b
BASES = "ACGT"
BASE_INDICES = {base: n for n, base in enumerate(BASES)}
# every 3-base reference context, in lexicographic order
CONTEXTS = tuple(a + b + c for a in BASES for b in BASES for c in BASES)

REF_HISTOGRAM_EXTENSION = ".ref_histogram"
ALT_HISTOGRAM_EXTENSION = ".alt_histogram"
ALT_TABLE_EXTENSION = ".alt_table"

# the columns of a ref histogram table before its columns of depths, and those of an alt histogram table
REF_HISTOGRAM_KEY_COLUMNS = ("contig", "block_start", "context")
ALT_HISTOGRAM_KEY_COLUMNS = (*REF_HISTOGRAM_KEY_COLUMNS, "alt", "orientation")
# rows of a histogram table read at once, which bounds memory: a genome has about 1.3 million if Mutect2 isn't scattered
HISTOGRAM_CHUNK_ROWS = 100_000
# columns of an alt table that the read orientation model uses, with the ref count for the depth and the ref F1R2 count
# for the number of F1R2 reads, and the columns of a site's position, which a segmentation needs
ALT_TABLE_COLUMNS = ("ref_count", "alt_count", "ref_f1r2", "alt_f1r2", "alt")
ALT_TABLE_POSITION_COLUMNS = ("contig", "position")

# the columns of the distinct sites of the tables: the reference context, then the keys of Sites, the MAF mixture and
# the ref F1R2 count
SITE_COLUMNS = ("context", "alt_base", "alt_count", "alt_f1r2", "depth", "maf_mixture", "ref_f1r2")
# the ref F1R2 count of the sites of the histograms, which don't record it
UNKNOWN_REF_F1R2 = -1


@dataclass(frozen=True)
class Sites:
    """
    Sites of a reference context as their distinct combinations of alt base (an index into BASES), alt count, count of
    alt reads in F1R2 orientation and depth, the columns of keys_n4, of MAF mixture (an index into the F1R2 counts'
    table of mixtures) and of count of ref reads in F1R2 orientation (UNKNOWN_REF_F1R2 for the histograms' sites), with
    the number of sites of each.  A site without alt reads has the ref base as its alt base, and its depth is its ref
    count.  The read orientation model depends on nothing else, so this is exact, and there are never more combinations
    than sites.  The constructor takes combinations that are already distinct, with positive numbers of sites.
    """

    keys_n4: Tensor
    counts_n: Tensor
    maf_mixtures_n: Tensor
    ref_f1r2_counts_n: Tensor

    @staticmethod
    def of(keys_n4, counts_n, maf_mixtures_n=None, ref_f1r2_counts_n=None) -> "Sites":
        """
        Sites with the given keys, numbers of sites, MAF mixtures, by default MAF 1/2, and ref F1R2 counts, by default
        unknown, in lexicographic order of key, then MAF mixture, then ref F1R2 count, combining identical combinations
        and dropping those without sites
        """
        keys_n4 = torch.as_tensor(keys_n4, dtype=torch.int64).reshape(-1, 4)
        if maf_mixtures_n is None:
            maf_mixtures_n = torch.full((len(keys_n4),), BALANCED_MIXTURE, dtype=torch.int64)
        if ref_f1r2_counts_n is None:
            ref_f1r2_counts_n = torch.full((len(keys_n4),), UNKNOWN_REF_F1R2, dtype=torch.int64)
        extra_n2 = torch.stack(
            [torch.as_tensor(x, dtype=torch.int64).reshape(-1) for x in (maf_mixtures_n, ref_f1r2_counts_n)], 1
        )
        distinct_m6, inverse_n = torch.unique(torch.cat([keys_n4, extra_n2], dim=-1), dim=0, return_inverse=True)
        counts_m = torch.zeros(len(distinct_m6), dtype=torch.int64)
        counts_m.index_add_(0, inverse_n, torch.as_tensor(counts_n, dtype=torch.int64))
        distinct_m6 = distinct_m6[counts_m > 0]
        return Sites(distinct_m6[:, :4], counts_m[counts_m > 0], distinct_m6[:, 4], distinct_m6[:, 5])

    @staticmethod
    def empty() -> "Sites":
        return Sites(torch.zeros(0, 4, dtype=torch.int64), *(torch.zeros(0, dtype=torch.int64) for _ in range(3)))

    @staticmethod
    def concatenate(sites) -> "Sites":
        """the sites of all the given Sites, combining identical combinations"""
        sites = [Sites.empty(), *sites]
        names = ("keys_n4", "counts_n", "maf_mixtures_n", "ref_f1r2_counts_n")
        return Sites.of(*(torch.cat([getattr(s, name) for s in sites]) for name in names))

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


@dataclass(frozen=True)
class CaptureCounts:
    """
    The depths and numbers of F1R2 reads, ref and alt, of the sites of the alt tables, as their distinct combinations,
    with the number of sites of each
    """

    depths_j: Tensor
    f1r2_counts_j: Tensor
    num_sites_j: Tensor

    @staticmethod
    def empty() -> "CaptureCounts":
        return CaptureCounts(*(torch.zeros(0, dtype=torch.int64) for _ in range(3)))

    def tensors(self) -> tuple[Tensor, Tensor, Tensor]:
        return self.depths_j, self.f1r2_counts_j, self.num_sites_j

    def num_sites(self) -> int:
        return int(self.num_sites_j.sum())


@dataclass
class F1R2Counts:
    """
    The F1R2 counts of one sample, summed over the inputs: the sites of each reference context, the weight of each grid
    MAF in each of the sites' MAF mixtures, and the depths and F1R2 counts of the reads at the alt tables' sites.  Sites
    from the histograms have the depths of their columns, and those that are deeper than max_depth count as max_depth.
    """

    sample: str
    max_depth: int
    sites: dict[str, Sites]
    maf_mixtures_mg: Tensor = field(default_factory=balanced_mixtures_mg)
    capture_counts: CaptureCounts = field(default_factory=CaptureCounts.empty)


def histogram_sites(rows: pd.DataFrame, alt: bool, maf_mixtures: MafMixtures) -> pd.DataFrame:
    """
    The distinct sites of rows of a ref histogram table, or an alt histogram table if alt is true: a DataFrame of the
    SITE_COLUMNS and the number of each in the column count.  Sites without alt reads have the ref base as their alt base,
    and every site of a row has the MAF mixture of its block.
    """
    num_key_columns = len(ALT_HISTOGRAM_KEY_COLUMNS if alt else REF_HISTOGRAM_KEY_COLUMNS)
    depths_d = np.array(rows.columns[num_key_columns:], dtype=np.int64)
    counts_rd = rows.iloc[:, num_key_columns:].to_numpy(dtype=np.int64)
    r, d = np.nonzero(counts_rd)
    alt_bases = rows["alt"] if alt else rows["context"].str[1]
    sites = pd.DataFrame(
        {
            "context": rows["context"].to_numpy()[r],
            "alt_base": alt_bases.map(BASE_INDICES).to_numpy(dtype=np.int64)[r],
            "alt_count": int(alt),
            "alt_f1r2": (rows["orientation"] == "F1R2").to_numpy(dtype=np.int64)[r] if alt else 0,
            "depth": depths_d[d],
            "maf_mixture": maf_mixtures.of_blocks(rows["contig"].to_numpy(), rows["block_start"].to_numpy())[r],
            "ref_f1r2": UNKNOWN_REF_F1R2,
            "count": counts_rd[r, d],
        }
    )
    return sites.groupby(list(SITE_COLUMNS), as_index=False)["count"].sum()


def alt_table_sites(rows: pd.DataFrame, context: str, maf_mixtures: MafMixtures) -> pd.DataFrame:
    """
    The distinct sites of the rows of an alt table of a reference context: a DataFrame of the SITE_COLUMNS and the number
    of each in the column count.  Each site has the MAF mixture of its position, which takes the columns of
    ALT_TABLE_POSITION_COLUMNS if the MAF mixtures have segments.  An alt table is small, and there are 64 for each
    sample and input, so numpy combines identical sites, which costs less than a pandas groupby.
    """
    alt_counts = rows["alt_count"].to_numpy(dtype=np.int64)
    depths = rows["ref_count"].to_numpy(dtype=np.int64) + alt_counts
    alt_f1r2_counts = rows["alt_f1r2"].to_numpy(dtype=np.int64)
    if maf_mixtures.has_segments():
        positions = rows["position"].to_numpy(dtype=np.int64)
        mixtures = maf_mixtures.of_positions(rows["contig"].to_numpy(), positions)
    else:
        mixtures = np.full(len(rows), BALANCED_MIXTURE, dtype=np.int64)
    alt_bases = rows["alt"].map(BASE_INDICES).to_numpy(dtype=np.int64)
    ref_f1r2_counts = rows["ref_f1r2"].to_numpy(dtype=np.int64)
    keys_n6 = np.column_stack([alt_bases, alt_counts, alt_f1r2_counts, depths, mixtures, ref_f1r2_counts])
    distinct_m6, counts_m = np.unique(keys_n6, axis=0, return_counts=True)
    return pd.DataFrame({"context": context, **dict(zip(SITE_COLUMNS[1:], distinct_m6.T)), "count": counts_m})


def alt_table_capture_counts(rows: pd.DataFrame) -> pd.DataFrame:
    """the distinct (depth, number of F1R2 reads) of the sites of an alt table, and the number of each"""
    depths = (rows["ref_count"] + rows["alt_count"]).to_numpy(dtype=np.int64)
    f1r2_counts = (rows["ref_f1r2"] + rows["alt_f1r2"]).to_numpy(dtype=np.int64)
    distinct, counts = np.unique(np.column_stack([depths, f1r2_counts]), axis=0, return_counts=True)
    return pd.DataFrame({"depth": distinct[:, 0], "f1r2": distinct[:, 1], "count": counts})


def alt_table_context(name: str) -> str:
    """the reference context of an alt table <sample>.<context>.alt_table, given its file name"""
    context = name.removesuffix(ALT_TABLE_EXTENSION).rsplit(".", 1)[-1]
    if context not in CONTEXTS:
        raise ValueError(
            f"{name} is not named <sample>.<reference context>{ALT_TABLE_EXTENSION}.  Older versions of GATK made one "
            "alt table per sample; please regenerate the F1R2 counts with a current Mutect2 or CollectF1R2Counts."
        )
    return context


def histogram_depths(columns: list[str], alt: bool, name: str) -> tuple[int, ...]:
    """the depths of the columns of a ref histogram table, or an alt histogram table if alt is true"""
    key_columns = ALT_HISTOGRAM_KEY_COLUMNS if alt else REF_HISTOGRAM_KEY_COLUMNS
    depth_columns = columns[len(key_columns) :]
    if tuple(columns[: len(key_columns)]) != key_columns or not all(column.isdigit() for column in depth_columns):
        raise ValueError(f"{name} does not have the columns {key_columns} and then depths")
    depths = tuple(map(int, depth_columns))
    if not depths or depths[0] < 1 or any(a >= b for a, b in zip(depths, depths[1:])):
        raise ValueError(f"{name} does not have increasing, positive depths as its columns")
    return depths


def read_f1r2_counts(tar_gz_paths, segmentation: SampleSegmentation = None) -> dict[str, F1R2Counts]:
    """
    The F1R2 counts of each sample in one or more tar.gz files from Mutect2 or CollectF1R2Counts, eg from the scatters
    of one run, summed over the files.  The files are read in order, without extracting them to disk.  Given a
    segmentation by MAF of the counts' sample, the sites get the MAF mixtures of their positions and blocks; otherwise
    all have MAF 1/2.
    """
    tar_gz_paths = list(tar_gz_paths)
    maf_mixtures = MafMixtures(segmentation)
    alt_table_columns = [*ALT_TABLE_COLUMNS, *(ALT_TABLE_POSITION_COLUMNS if maf_mixtures.has_segments() else ())]
    sites = defaultdict(list)  # by sample, DataFrames of the distinct sites of each table
    capture_counts = defaultdict(list)  # by sample, DataFrames of the depths and F1R2 counts of each alt table
    num_histogram_tables = Counter()  # by sample and extension
    depths = defaultdict(set)  # by sample, the depths of its histogram tables
    for path in tar_gz_paths:
        for name, file in _text_files(path):
            extension = os.path.splitext(name)[1]
            if extension == ALT_TABLE_EXTENSION:
                context = alt_table_context(name)
                table = read_gatk_table(file, usecols=alt_table_columns, dtype={"alt": str, "contig": str})
                sample = _sample(table, name, path)
                sites[sample].append(alt_table_sites(table.rows, context, maf_mixtures))
                capture_counts[sample].append(alt_table_capture_counts(table.rows))
            elif extension in (REF_HISTOGRAM_EXTENSION, ALT_HISTOGRAM_EXTENSION):
                alt = extension == ALT_HISTOGRAM_EXTENSION
                key_columns = ALT_HISTOGRAM_KEY_COLUMNS if alt else REF_HISTOGRAM_KEY_COLUMNS
                string_columns = {column: str for column in key_columns if column != "block_start"}
                table = read_gatk_table(file, chunksize=HISTOGRAM_CHUNK_ROWS, dtype=string_columns)
                sample = _sample(table, name, path)
                num_histogram_tables[sample, extension] += 1
                depths[sample].add(histogram_depths(table.columns, alt, f"{name} in {path}"))
                sites[sample].extend(histogram_sites(chunk, alt, maf_mixtures) for chunk in table.rows)

    result = {}
    for sample, sample_sites in sites.items():
        num_ref_tables, num_alt_tables = (
            num_histogram_tables[sample, ext] for ext in (REF_HISTOGRAM_EXTENSION, ALT_HISTOGRAM_EXTENSION)
        )
        if num_ref_tables == 0 or num_ref_tables != num_alt_tables:
            raise ValueError(
                f"Sample {sample} of the F1R2 counts in {tar_gz_paths} does not have one ref histogram table and one alt "
                "histogram table in each input that has it"
            )
        if len(depths[sample]) != 1:
            raise ValueError(f"The F1R2 histogram tables of sample {sample} have different depths")
        (sample_depths,) = depths[sample]

        # combine identical sites, in lexicographic order, as Sites keeps them
        counts = pd.concat(sample_sites).groupby(list(SITE_COLUMNS))["count"].sum()
        if unknown := set(counts.index.get_level_values("context")) - set(CONTEXTS):
            raise ValueError(f"The F1R2 counts of sample {sample} have sites of unknown contexts {sorted(unknown)}")
        sites_by_context = dict.fromkeys(CONTEXTS, Sites.empty())
        for context, context_counts in counts.groupby(level="context"):
            keys = torch.tensor(context_counts.index.droplevel("context").to_frame().to_numpy(dtype=np.int64))
            counts_n = torch.tensor(context_counts.to_numpy(dtype=np.int64))
            sites_by_context[context] = Sites(keys[:, :4], counts_n, keys[:, 4], keys[:, 5])
        capture = CaptureCounts.empty()
        if capture_counts[sample]:
            summed = pd.concat(capture_counts[sample]).groupby(["depth", "f1r2"])["count"].sum()
            capture = CaptureCounts(
                *(torch.tensor(summed.index.get_level_values(level).to_numpy(dtype=np.int64)) for level in (0, 1)),
                torch.tensor(summed.to_numpy(dtype=np.int64)),
            )
        result[sample] = F1R2Counts(sample, sample_depths[-1], sites_by_context, maf_mixtures.weights_mg(), capture)
    if maf_mixtures.has_segments():
        print(f"MAF mixtures of the F1R2 counts' sites: {maf_mixtures.summary()}")
    return result


def _sample(table, name: str, path) -> str:
    if table.sample is None:
        raise ValueError(f"{name} in {path} does not name its sample")
    return table.sample


def _text_files(tar_gz_path):
    """the name and text of each file in a tar.gz, in order"""
    with tarfile.open(tar_gz_path, mode="r:gz") as tar:
        for member in tar:
            if member.isfile():
                with io.TextIOWrapper(tar.extractfile(member), encoding="utf-8") as file:
                    yield os.path.basename(member.name), file
