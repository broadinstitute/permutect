"""
Reading the STR slippage counts of GATK CollectSTRSlippageCounts, the training data of the slippage model: a GATK table
(see permutect.utils.gatk_table) with a row for each short tandem repeat (STR) of the reference that at least one fragment
spans, giving its location, repeat unit and number of whole units, the number of fragments that change its length by each
number of units, and the population allele frequencies of the two most common non-zero changes.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from permutect.utils.gatk_table import read_gatk_table

# the columns of fragment counts by change in an STR's length: at most -4 units, -3, ..., at least +4 units, and a change
# that isn't a whole number of units
COUNT_COLUMNS = (
    "minus_4_or_more",
    "minus_3",
    "minus_2",
    "minus_1",
    "no_change",
    "plus_1",
    "plus_2",
    "plus_3",
    "plus_4_or_more",
    "not_whole_units",
)
MAX_BINNED_CHANGE = 4
NO_CHANGE_BIN = COUNT_COLUMNS.index("no_change")
NOT_WHOLE_UNITS_BIN = COUNT_COLUMNS.index("not_whole_units")

# the two most common non-zero changes of whole units, a change of 0 if there is no such change, and their population
# allele frequencies in the germline resource, NaN if it doesn't have the allele or no resource was given
ALT_CHANGE_COLUMNS = ("top_alt_change", "second_alt_change")
ALT_ALLELE_FREQUENCY_COLUMNS = ("top_alt_af", "second_alt_af")

LOCATION_COLUMNS = ("contig", "start", "unit", "units")


@dataclass
class SlippageCounts:
    """
    The STRs of one sample's slippage counts table that read_slippage_counts keeps, indexed by s, with the count bins of
    COUNT_COLUMNS indexed by k and the two most common changes indexed by a
    """

    sample: str
    contigs: list[str]
    starts_s: Tensor  # the 1-based position of each STR's first base
    unit_lengths_s: Tensor  # the bases of each STR's repeat unit
    repeat_lengths_s: Tensor  # the bases of each STR's whole units, the reference STR length of filter_variants
    counts_sk: Tensor
    alt_changes_sa: Tensor
    alt_allele_frequencies_sa: Tensor

    def num_strs(self) -> int:
        return len(self.contigs)

    def depths_s(self) -> Tensor:
        """the number of fragments that span each STR"""
        return self.counts_sk.sum(dim=-1)


def read_slippage_counts(path) -> SlippageCounts:
    """
    The slippage counts table at path, without the STRs that may have two non-reference alleles: those where the
    reference length doesn't have the most or second-most fragments.
    """
    columns = [*LOCATION_COLUMNS, *COUNT_COLUMNS, *ALT_CHANGE_COLUMNS, *ALT_ALLELE_FREQUENCY_COLUMNS]
    with open(path) as file:
        table = read_gatk_table(file, usecols=columns, dtype={"contig": str, "unit": str})
    if table.sample is None:
        raise ValueError(f"The slippage counts table {path} doesn't name its sample")
    # drop the STRs where two count bins, including not_whole_units, have more fragments than the reference; a tie keeps
    # the STR.  This also drops hom alts with more fragments of another length, eg their own slippage away from the
    # reference, than of the reference's.
    counts = table.rows[list(COUNT_COLUMNS)]
    rows = table.rows[counts.gt(counts["no_change"], axis=0).sum(axis=1) <= 1]
    unit_lengths_s = torch.tensor(rows["unit"].str.len().to_numpy())
    units_s = torch.tensor(rows["units"].to_numpy())
    return SlippageCounts(
        sample=table.sample,
        contigs=rows["contig"].tolist(),
        starts_s=torch.tensor(rows["start"].to_numpy()),
        unit_lengths_s=unit_lengths_s,
        repeat_lengths_s=unit_lengths_s * units_s,
        counts_sk=torch.tensor(rows[list(COUNT_COLUMNS)].to_numpy()),
        alt_changes_sa=torch.tensor(rows[list(ALT_CHANGE_COLUMNS)].to_numpy()),
        # the reader keeps fields as strings, so GATK's NaN arrives as a string
        alt_allele_frequencies_sa=torch.tensor(rows[list(ALT_ALLELE_FREQUENCY_COLUMNS)].astype(float).to_numpy()),
    )
