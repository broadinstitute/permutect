import argparse
import math

import numpy as np
import torch

from permutect import constants
from permutect.segmentation.emissions import Pileups
from permutect.segmentation.plotting import plot_segmentation
from permutect.segmentation.segmenter import Segment
from permutect.segmentation.segmenter import make_segments
from permutect.segmentation.segmenter import segment

SAMPLE_HEADER = "#<METADATA>SAMPLE="
SEGMENTS_COLUMNS = (
    "contig",
    "start",
    "end",
    "minor_allele_fraction",
    "secondary_minor_allele_fraction",
    "minor_allele_fraction_low",
    "minor_allele_fraction_high",
)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Segment a sample by minor allele fraction (MAF), using the population allele frequencies of the "
        "common SNPs in its GATK GetPileupSummaries table.  The output has the format of the segmentation of GATK "
        "CalculateContamination plus three columns.  The secondary MAF: between two segments a gap segment has the "
        "MAF of the first and the secondary MAF of the second, and within a segment the secondary MAF equals the MAF.  "
        "The low and high MAFs: the bounds of the narrowest interval with 80% of the posterior of a segment's MAF, or "
        "NaN for a gap segment."
    )
    parser.add_argument("--" + constants.INPUT_NAME, required=True, help="GetPileupSummaries table of the sample")
    parser.add_argument("--" + constants.OUTPUT_NAME, required=True, help="path to output segments table")
    parser.add_argument(
        "--" + constants.NUM_MAF_STATES_NAME,
        type=int,
        default=51,
        help="number of hidden states of the hidden Markov model, ie of MAFs evenly spaced from 0 to 1/2",
    )
    parser.add_argument(
        "--" + constants.CONTAMINATION_NAME,
        type=float,
        default=0.0,
        help="fraction of the sample's reads that come from another person",
    )
    parser.add_argument(
        "--" + constants.MAX_BREAKPOINT_RATE_NAME,
        type=float,
        default=1e-7,
        help="maximum rate per base of breakpoints between segments",
    )
    parser.add_argument(
        "--" + constants.MAF_PRIOR_CONCENTRATION_NAME,
        type=float,
        default=0.1,
        help="concentration of the symmetric Dirichlet prior on the prior distribution of MAFs.  Below 1 it prunes "
        "MAFs that few segments have.",
    )
    parser.add_argument(
        "--" + constants.MAX_EM_ITERATIONS_NAME,
        type=int,
        default=100,
        help="maximum number of iterations of expectation maximization",
    )
    parser.add_argument(
        "--" + constants.PLOT_NAME,
        help="path to an optional output plot of the sites' alt fractions and the segments' MAFs along the genome, like "
        "GATK PlotModeledSegments'.  Matplotlib infers the format from the extension, eg .png.",
    )
    return parser.parse_args(argv)


def main_without_parsing(args):
    segment_pileups(
        input_table=getattr(args, constants.INPUT_NAME),
        output_table=getattr(args, constants.OUTPUT_NAME),
        num_maf_states=getattr(args, constants.NUM_MAF_STATES_NAME),
        contamination=getattr(args, constants.CONTAMINATION_NAME),
        max_breakpoint_rate=getattr(args, constants.MAX_BREAKPOINT_RATE_NAME),
        maf_prior_concentration=getattr(args, constants.MAF_PRIOR_CONCENTRATION_NAME),
        max_em_iterations=getattr(args, constants.MAX_EM_ITERATIONS_NAME),
        plot=getattr(args, constants.PLOT_NAME),
    )


def read_pileup_summaries(table) -> tuple[str, list[str], np.ndarray, np.ndarray, Pileups]:
    """
    The sample name, contigs, number of sites of each contig, positions and pileups of a GetPileupSummaries table, with
    the sites of each contig consecutive and in order of position.  Sites without coverage are left out, as are sites
    whose population allele frequency isn't strictly between 0 and 1.
    """
    sample, contigs = "", []
    with open(table) as file:
        for line in file:
            if line.startswith(SAMPLE_HEADER):
                sample = line[len(SAMPLE_HEADER) :].strip()
            elif not line.startswith("#"):
                columns = line.split()
                break
        names = ("contig", "position", "ref_count", "alt_count", "allele_frequency")
        contig_column, *number_columns = (columns.index(name) for name in names)
        numbers = []
        for line in file:
            tokens = line.split()
            contigs.append(tokens[contig_column])
            numbers.append([float(tokens[column]) for column in number_columns])

    positions_n, ref_counts_n, alt_counts_n, allele_frequencies_n = np.array(numbers, dtype=np.float64).reshape(-1, 4).T
    depths_n = ref_counts_n + alt_counts_n
    kept_n = (depths_n > 0) & (allele_frequencies_n > 0) & (allele_frequencies_n < 1)
    kept_contigs = [contig for contig, kept in zip(contigs, kept_n) if kept]

    # contigs in order of first appearance, which in GetPileupSummaries is the reference order
    contigs_c = list(dict.fromkeys(kept_contigs))
    contig_indices = {contig: index for index, contig in enumerate(contigs_c)}
    contigs_n = np.array([contig_indices[contig] for contig in kept_contigs], dtype=np.int64)
    order_n = np.lexsort((positions_n[kept_n], contigs_n))
    pileups = Pileups(
        *(torch.from_numpy(array[kept_n][order_n]) for array in (alt_counts_n, depths_n, allele_frequencies_n))
    )
    lengths_c = np.bincount(contigs_n, minlength=len(contigs_c))
    return sample, contigs_c, lengths_c, positions_n[kept_n][order_n].astype(np.int64), pileups


def write_segments(table, sample: str, segments: list[Segment]):
    with open(table, "w") as file:
        file.write(SAMPLE_HEADER + sample + "\n")
        file.write("\t".join(SEGMENTS_COLUMNS) + "\n")
        for s in segments:
            mafs = "\t".join(_format_maf(maf) for maf in (s.maf, s.secondary_maf, s.maf_low, s.maf_high))
            file.write(f"{s.contig}\t{s.start}\t{s.end}\t{mafs}\n")


def _format_maf(maf: float) -> str:
    # NaN as GATK writes it
    return "NaN" if math.isnan(maf) else f"{maf:.4f}"


def segment_pileups(
    input_table,
    output_table,
    num_maf_states: int = 51,
    contamination: float = 0.0,
    max_breakpoint_rate: float = 1e-7,
    maf_prior_concentration: float = 0.1,
    max_em_iterations: int = 100,
    plot=None,
):
    sample, contigs_c, lengths_c, positions_n, pileups = read_pileup_summaries(input_table)
    segments = []
    if len(positions_n) > 0:
        segmentation = segment(
            pileups,
            positions_n,
            lengths_c,
            num_maf_states=num_maf_states,
            contamination=contamination,
            max_jump_rate=max_breakpoint_rate,
            prior_concentration=maf_prior_concentration,
            max_iterations=max_em_iterations,
        )
        segments = make_segments(contigs_c, lengths_c, positions_n, segmentation.run_starts_r, segmentation.run_mafs_r3)
    write_segments(output_table, sample, segments)
    if plot is not None:
        plot_segmentation(plot, sample, contigs_c, lengths_c, positions_n, pileups, segments)


def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == "__main__":
    main()
