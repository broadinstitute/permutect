import argparse

import torch
from torch import Tensor

from permutect import constants
from permutect.segmentation.germline_model import GermlineModel
from permutect.segmentation.germline_model import SampleSegmentation
from permutect.slippage.slip_rates import DEFAULT_ALLELE_FREQUENCY
from permutect.slippage.slip_rates import MAX_UNIT_LENGTH
from permutect.slippage.slip_rates import SLIPS
from permutect.slippage.slip_rates import SlippageData
from permutect.slippage.slip_rates import fit_slip_log_odds
from permutect.slippage.slip_rates import slip_probabilities
from permutect.slippage.slippage_counts import SlippageCounts
from permutect.slippage.slippage_counts import read_slippage_counts
from permutect.slippage.slippage_model import SlippageModel

# allele lengths, in bases, at which to print the learned slip rates
REPORTED_LENGTHS = (8, 10, 12, 15, 20, 30)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Learn a sample's model of polymerase slippage, the PCR and sequencing errors that add or remove "
        "repeat units of short tandem repeats (STRs), from its STR slippage counts from GATK CollectSTRSlippageCounts.  "
        "The sample's germline model from segment_pileups gives the minor allele fractions of germline STR alleles, "
        "to tell slippage from germline variation in the STRs' lengths.  The output is a SlippageModel with the "
        "maximum-likelihood rates of slips by one and two repeat units, by repeat unit length and allele length; "
        "filter_variants doesn't use them yet."
    )
    parser.add_argument(
        "--" + constants.INPUT_NAME,
        required=True,
        help="STR slippage counts table of one sample from GATK CollectSTRSlippageCounts",
    )
    parser.add_argument(
        "--" + constants.GERMLINE_MODEL_NAME,
        required=True,
        help="GermlineModel from segment_pileups whose tumor or matched normal is the sample of the slippage counts",
    )
    parser.add_argument("--" + constants.OUTPUT_NAME, required=True, help="path to the output SlippageModel")
    parser.add_argument(
        "--" + constants.DEFAULT_ALLELE_FREQUENCY_NAME,
        type=float,
        default=DEFAULT_ALLELE_FREQUENCY,
        help="population allele frequency of STR alleles that the germline resource of CollectSTRSlippageCounts lacks",
    )
    return parser.parse_args(argv)


def main_without_parsing(args):
    learn_slippage_model(
        input_table=getattr(args, constants.INPUT_NAME),
        germline_model=GermlineModel.load(getattr(args, constants.GERMLINE_MODEL_NAME)),
        output_path=getattr(args, constants.OUTPUT_NAME),
        default_allele_frequency=getattr(args, constants.DEFAULT_ALLELE_FREQUENCY_NAME),
    )


def learn_slippage_model(
    input_table, germline_model: GermlineModel, output_path, default_allele_frequency: float = DEFAULT_ALLELE_FREQUENCY
):
    """
    The slippage model of the sample of the STR slippage counts in input_table, saved to output_path.  germline_model is
    segment_pileups' model of the sample, as its tumor or as its matched normal.  Alt alleles without a population allele
    frequency get default_allele_frequency.
    """
    counts = read_slippage_counts(input_table)
    print(
        f"Slippage counts of sample {counts.sample}: {counts.num_strs()} STRs spanned by "
        f"{int(counts.depths_s().sum())} fragments"
    )
    mafs_s = str_mafs(sample_segmentation(germline_model, counts.sample), counts)
    data = SlippageData.of(counts, mafs_s, default_allele_frequency)
    log_odds_jul, log_likelihood, num_iterations = fit_slip_log_odds(data)
    print(f"Fit the slip rates in {num_iterations} L-BFGS iterations: log likelihood {log_likelihood:.6f} per fragment")
    print_slip_rates(log_odds_jul)
    SlippageModel(slip_log_odds_jul=log_odds_jul).save(output_path)


def sample_segmentation(germline_model: GermlineModel, sample: str) -> SampleSegmentation:
    """the germline model's segmentation of the sample, as the tumor or as the matched normal"""
    for segmentation in (germline_model.tumor, germline_model.normal):
        if segmentation.sample == sample:
            return segmentation
    samples = [
        segmentation.sample for segmentation in (germline_model.tumor, germline_model.normal) if segmentation.sample
    ]
    raise ValueError(
        f"The slippage counts are of sample {sample}, but the germline model is of samples {', '.join(samples) or 'none'}"
    )


def str_mafs(segmentation: SampleSegmentation, counts: SlippageCounts) -> Tensor:
    """the MAF of the segment of each STR's first base, or 1/2 outside the segments, as filter_variants looks them up"""
    intervals = segmentation.maf_intervals()

    def maf(contig, start) -> float:
        overlaps = intervals[contig][start]
        return next(iter(overlaps)).data if overlaps else 0.5

    return torch.tensor([maf(c, s) for c, s in zip(counts.contigs, counts.starts_s.tolist())], dtype=torch.float64)


def print_slip_rates(log_odds_jul: Tensor):
    """print the probability of each slip at some unit lengths and allele lengths"""
    rates_jul = slip_probabilities(log_odds_jul)
    columns = "".join(f"{length:>9}" for length in REPORTED_LENGTHS)
    for slip in (-1, 1, -2, 2):
        print(f"Rates of slips by {slip:+d} units, by unit length (rows) and allele length in bases (columns)")
        print(f"{'':>5}{columns}")
        for u in range(1, MAX_UNIT_LENGTH + 1):
            rates = [rates_jul[SLIPS.index(slip), u - 1, length - 1].item() for length in REPORTED_LENGTHS]
            print(f"{u:>5}" + "".join(f"{rate:9.1e}" for rate in rates))


def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == "__main__":
    main()
