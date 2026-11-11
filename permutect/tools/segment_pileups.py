import argparse
import math
from dataclasses import dataclass

import numpy as np
import torch

from permutect import constants
from permutect.segmentation.contamination import calculate_contamination
from permutect.segmentation.contamination import write_contamination_table
from permutect.segmentation.emissions import EmissionModel
from permutect.segmentation.emissions import Pileups
from permutect.segmentation.germline_model import GermlineModel
from permutect.segmentation.germline_model import SampleSegmentation
from permutect.segmentation.pileup_summaries import SAMPLE_HEADER
from permutect.segmentation.pileup_summaries import PileupSummaries
from permutect.segmentation.pileup_summaries import read_pileup_summaries
from permutect.segmentation.plotting import plot_segmentation
from permutect.segmentation.segmenter import Segment
from permutect.segmentation.segmenter import make_segments
from permutect.segmentation.segmenter import segment

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
        description="Segment a sample, and optionally its matched normal, by minor allele fraction (MAF), using the "
        "population allele frequencies of the common SNPs in their GATK GetPileupSummaries tables, and estimate the "
        "sample's contamination as GATK CalculateContamination does.  Segmentation and contamination alternate: the "
        "first segmentation assumes no contamination and each later one uses the latest estimate.  The segments "
        "tables have the format of the segmentation of GATK CalculateContamination plus three columns.  The secondary "
        "MAF: between two segments a gap segment has the MAF of the first and the secondary MAF of the second, and "
        "within a segment the secondary MAF equals the MAF.  The low and high MAFs: the bounds of the narrowest interval "
        "with 80% of the posterior of a segment's MAF, or NaN for a gap segment."
    )
    parser.add_argument("--" + constants.INPUT_NAME, required=True, help="GetPileupSummaries table of the sample")
    parser.add_argument(
        "--" + constants.NORMAL_INPUT_NAME,
        help="optional GetPileupSummaries table of a matched normal, which is segmented separately, assuming no "
        "contamination, and genotypes the sites for the contamination estimate",
    )
    parser.add_argument("--" + constants.OUTPUT_NAME, required=True, help="path to output segments table")
    parser.add_argument("--" + constants.NORMAL_OUTPUT_NAME, help="path to optional output segments of the normal")
    parser.add_argument(
        "--" + constants.CONTAMINATION_TABLE_NAME,
        help="path to optional output table of the contamination, in the format of GATK CalculateContamination",
    )
    parser.add_argument(
        "--" + constants.PLOT_NAME,
        help="path to an optional output plot of the sites' alt fractions and the segments' MAFs along the genome, like "
        "GATK PlotModeledSegments'.  Matplotlib infers the format from the extension, eg .png.",
    )
    parser.add_argument("--" + constants.NORMAL_PLOT_NAME, help="path to an optional output plot of the normal")
    parser.add_argument(
        "--" + constants.GERMLINE_MODEL_NAME,
        help="path to an optional output GermlineModel for filter_variants: the segments of the sample and of the "
        "normal, the parameters of the hidden Markov models that found them, and the contamination",
    )
    parser.add_argument(
        "--" + constants.NUM_CONTAMINATION_ITERATIONS_NAME,
        type=int,
        default=2,
        help="number of rounds of segmenting the sample and then estimating its contamination",
    )
    parser.add_argument(
        "--" + constants.NUM_MAF_STATES_NAME,
        type=int,
        default=51,
        help="number of hidden states of the hidden Markov model, ie of MAFs evenly spaced from 0 to 1/2",
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
    return parser.parse_args(argv)


def main_without_parsing(args):
    segment_pileups(
        input_table=getattr(args, constants.INPUT_NAME),
        output_table=getattr(args, constants.OUTPUT_NAME),
        normal_input_table=getattr(args, constants.NORMAL_INPUT_NAME),
        normal_output_table=getattr(args, constants.NORMAL_OUTPUT_NAME),
        contamination_table=getattr(args, constants.CONTAMINATION_TABLE_NAME),
        plot=getattr(args, constants.PLOT_NAME),
        normal_plot=getattr(args, constants.NORMAL_PLOT_NAME),
        germline_model=getattr(args, constants.GERMLINE_MODEL_NAME),
        num_contamination_iterations=getattr(args, constants.NUM_CONTAMINATION_ITERATIONS_NAME),
        num_maf_states=getattr(args, constants.NUM_MAF_STATES_NAME),
        max_breakpoint_rate=getattr(args, constants.MAX_BREAKPOINT_RATE_NAME),
        maf_prior_concentration=getattr(args, constants.MAF_PRIOR_CONCENTRATION_NAME),
        max_em_iterations=getattr(args, constants.MAX_EM_ITERATIONS_NAME),
    )


@dataclass
class SegmentationSites:
    """
    The sites to segment: the contigs, the number of sites of each contig, their positions and their pileups, with the
    sites of each contig consecutive and in order of position
    """

    contigs_c: list[str]
    lengths_c: np.ndarray
    positions_n: np.ndarray
    pileups: Pileups


def segmentation_sites(pileup_summaries: PileupSummaries) -> SegmentationSites:
    """
    The sites of a GetPileupSummaries table to segment, leaving out sites without coverage and sites whose population
    allele frequency isn't strictly between 0 and 1.  Depth is ref plus alt count, without other alt alleles.
    """
    depths_n = pileup_summaries.ref_counts_n + pileup_summaries.alt_counts_n
    allele_frequencies_n = pileup_summaries.allele_frequencies_n
    kept = pileup_summaries.subset((depths_n > 0) & (allele_frequencies_n > 0) & (allele_frequencies_n < 1))

    # contigs in order of first appearance, which in GetPileupSummaries is the reference order
    contigs_c = list(dict.fromkeys(kept.contigs_n))
    contig_indices = {contig: index for index, contig in enumerate(contigs_c)}
    contigs_n = np.array([contig_indices[contig] for contig in kept.contigs_n], dtype=np.int64)
    ordered = kept.subset(np.lexsort((kept.positions_n, contigs_n)))
    arrays = (ordered.alt_counts_n, ordered.ref_counts_n + ordered.alt_counts_n, ordered.allele_frequencies_n)
    pileups = Pileups(*(torch.from_numpy(np.asarray(array, dtype=np.float64)) for array in arrays))
    lengths_c = np.bincount(contigs_n, minlength=len(contigs_c))
    return SegmentationSites(contigs_c, lengths_c, ordered.positions_n, pileups)


def segment_sites(
    sample: str, sites: SegmentationSites, contamination: float, **segmentation_arguments
) -> SampleSegmentation:
    """the segments of a sample's sites and the parameters learned in finding them"""
    if len(sites.positions_n) == 0:
        return SampleSegmentation(sample, emission_model=EmissionModel(contamination=contamination))
    segmentation = segment(
        sites.pileups, sites.positions_n, sites.lengths_c, contamination=contamination, **segmentation_arguments
    )
    args = (sites.contigs_c, sites.lengths_c, sites.positions_n, segmentation.run_starts_r, segmentation.run_mafs_r3)
    hmm_parameters = (torch.from_numpy(segmentation.mafs_k), torch.from_numpy(segmentation.prior_k))
    return SampleSegmentation(
        sample, make_segments(*args), segmentation.emission_model, *hmm_parameters, segmentation.jump_rate
    )


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
    normal_input_table=None,
    normal_output_table=None,
    contamination_table=None,
    plot=None,
    normal_plot=None,
    germline_model=None,
    num_contamination_iterations: int = 2,
    num_maf_states: int = 51,
    max_breakpoint_rate: float = 1e-7,
    maf_prior_concentration: float = 0.1,
    max_em_iterations: int = 100,
):
    assert num_contamination_iterations >= 1
    assert normal_input_table is not None or (normal_output_table is None and normal_plot is None)
    segmentation_arguments = dict(
        num_maf_states=num_maf_states,
        max_jump_rate=max_breakpoint_rate,
        prior_concentration=maf_prior_concentration,
        max_iterations=max_em_iterations,
    )

    pileup_summaries = read_pileup_summaries(input_table)
    sample = pileup_summaries.sample
    sites = segmentation_sites(pileup_summaries)
    normal_pileup_summaries, normal_segmentation, normal_segments = None, None, None
    if normal_input_table is not None:
        normal_pileup_summaries = read_pileup_summaries(normal_input_table)
        normal_sites = segmentation_sites(normal_pileup_summaries)
        normal_segmentation = segment_sites(normal_pileup_summaries.sample, normal_sites, 0.0, **segmentation_arguments)
        normal_segments = normal_segmentation.segments

    contamination = 0.0
    for iteration in range(num_contamination_iterations):
        tumor_segmentation = segment_sites(sample, sites, contamination, **segmentation_arguments)
        contamination, error = calculate_contamination(
            pileup_summaries, tumor_segmentation.segments, normal_pileup_summaries, normal_segments
        )
        print(f"contamination iteration {iteration}: contamination {contamination:.6f}, error {error:.6f}")
    segments = tumor_segmentation.segments

    write_segments(output_table, sample, segments)
    if contamination_table is not None:
        write_contamination_table(contamination_table, sample, contamination, error)
    if plot is not None:
        plot_segmentation(plot, sample, sites.contigs_c, sites.lengths_c, sites.positions_n, sites.pileups, segments)
    if normal_output_table is not None:
        write_segments(normal_output_table, normal_pileup_summaries.sample, normal_segments)
    if normal_plot is not None:
        args = (normal_sites.contigs_c, normal_sites.lengths_c, normal_sites.positions_n, normal_sites.pileups)
        plot_segmentation(normal_plot, normal_pileup_summaries.sample, *args, normal_segments)
    if germline_model is not None:
        # the sample's emission model has the final contamination, not the one that its segmentation assumed
        tumor_segmentation.emission_model.contamination = contamination
        GermlineModel(tumor_segmentation, normal_segmentation, contamination, error).save(germline_model)


def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == "__main__":
    main()
