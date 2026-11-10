"""
Tables of GATK GetPileupSummaries: the ref, alt and other-alt read counts of a sample at common SNPs, with the population
allele frequency of each SNP's alt allele.
"""

from dataclasses import dataclass

import numpy as np

SAMPLE_HEADER = "#<METADATA>SAMPLE="


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
    sample = ""
    with open(table) as file:
        for line in file:
            if line.startswith(SAMPLE_HEADER):
                sample = line[len(SAMPLE_HEADER) :].strip()
            elif not line.startswith("#"):
                columns = line.split()
                break
        names = ("contig", "position", "ref_count", "alt_count", "other_alt_count", "allele_frequency")
        contig_column, *number_columns = (columns.index(name) for name in names)
        contigs, numbers = [], []
        for line in file:
            tokens = line.split()
            contigs.append(tokens[contig_column])
            numbers.append([float(tokens[column]) for column in number_columns])

    positions, ref_counts, alt_counts, other_alt_counts, allele_frequencies = (
        np.array(numbers, dtype=np.float64).reshape(-1, 5).T
    )
    return PileupSummaries(
        sample,
        np.array(contigs, dtype=object),
        positions.astype(np.int64),
        ref_counts.astype(np.int64),
        alt_counts.astype(np.int64),
        other_alt_counts.astype(np.int64),
        allele_frequencies,
    )
