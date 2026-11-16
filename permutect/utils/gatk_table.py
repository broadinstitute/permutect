"""
Reading tables in the format of GATK's TableWriter, eg pileup summaries and the F1R2 counts of Mutect2: lines of
metadata such as #<METADATA>SAMPLE=NA12878, then a header of tab-separated column names, then a tab-separated row for
each record.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TextIO

import pandas as pd

METADATA_PREFIX = "#<METADATA>"
SAMPLE_METADATA_KEY = "SAMPLE"
# the line of a table's metadata that names its sample, eg #<METADATA>SAMPLE=NA12878
SAMPLE_METADATA_PREFIX = f"{METADATA_PREFIX}{SAMPLE_METADATA_KEY}="


@dataclass
class GatkTable:
    """
    A GATK table: its metadata by key, its column names, and its rows, in one DataFrame or, if it is read in chunks, an
    iterator of DataFrames
    """

    metadata: dict[str, str]
    columns: list[str]
    rows: pd.DataFrame | Iterator[pd.DataFrame]

    @property
    def sample(self) -> str | None:
        return self.metadata.get(SAMPLE_METADATA_KEY)


def read_gatk_table(file: TextIO, chunksize: int | None = None, **read_csv_arguments) -> GatkTable:
    """
    Read a GATK table from a text file, from its current position.  With chunksize, the rows come in chunks of up to
    that many, which are read while the file stays open.  Other keyword arguments go to pandas.read_csv, eg usecols for
    only some columns and dtype for their types.  Fields are kept as they are, without pandas' conversion of NA and
    other strings to missing values, and decimals are parsed exactly as Python parses them.
    """
    name = getattr(file, "name", "the GATK table")
    metadata = {}
    while (line := file.readline()).startswith("#"):
        if line.startswith(METADATA_PREFIX):
            key, _, value = line.removeprefix(METADATA_PREFIX).rstrip("\n").partition("=")
            metadata[key] = value
    if not line:
        raise ValueError(f"{name} is not a GATK table: it has no header of column names")
    columns = line.rstrip("\n").split("\t")
    if missing := set(read_csv_arguments.get("usecols", ())) - set(columns):
        raise ValueError(f"{name} does not have the columns {sorted(missing)}")
    rows = pd.read_csv(
        file,
        sep="\t",
        names=columns,
        header=None,
        na_filter=False,
        float_precision="round_trip",
        chunksize=chunksize,
        **read_csv_arguments,
    )
    return GatkTable(metadata, columns, rows)
