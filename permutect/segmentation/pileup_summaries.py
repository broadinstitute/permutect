"""
Tables of GATK GetPileupSummaries: the ref, alt and other-alt read counts of a sample at common SNPs, with the population
allele frequency of each SNP's alt allele.
"""

from dataclasses import dataclass

import numpy as np

from permutect.utils.gatk_table import SAMPLE_METADATA_PREFIX
from permutect.utils.gatk_table import read_gatk_table

# the line of a table of pileup summaries that names its sample, for writing one
SAMPLE_HEADER = SAMPLE_METADATA_PREFIX
COUNT_COLUMNS = ("ref_count", "alt_count", "other_alt_count")


@dataclass
class PileupSummaries:
    sample: str
    contigs_n: np.ndarray
    positions_n: np.ndarray
    ref_counts_n: np.ndarray
    alt_counts_n: np.ndarray
    other_alt_counts_n: np.ndarray
    allele_frequencies_n: np.ndarray

    def total_counts(self) -> np.ndarray:
        """GATK's total count of a site, which includes the reads of other alt alleles"""
        return self.ref_counts_n + self.alt_counts_n + self.other_alt_counts_n

    def subset(self, indices) -> "PileupSummaries":
        return PileupSummaries(
            self.sample,
            self.contigs_n[indices],
            self.positions_n[indices],
            self.ref_counts_n[indices],
            self.alt_counts_n[indices],
            self.other_alt_counts_n[indices],
            self.allele_frequencies_n[indices],
        )

    def __len__(self) -> int:
        return len(self.positions_n)


def read_pileup_summaries(table) -> PileupSummaries:
    """all the sites of a GetPileupSummaries table, in its order"""
    with open(table) as file:
        pileups = read_gatk_table(
            file, usecols=["contig", "position", *COUNT_COLUMNS, "allele_frequency"], dtype={"contig": str}
        )
    rows = pileups.rows
    return PileupSummaries(
        pileups.sample or "",
        rows["contig"].to_numpy(dtype=object),
        rows["position"].to_numpy(dtype=np.int64),
        *(rows[column].to_numpy(dtype=np.int64) for column in COUNT_COLUMNS),
        rows["allele_frequency"].to_numpy(dtype=np.float64),
    )
