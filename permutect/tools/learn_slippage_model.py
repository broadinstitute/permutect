import argparse

from permutect import constants
from permutect.segmentation.germline_model import GermlineModel
from permutect.segmentation.germline_model import SampleSegmentation
from permutect.slippage.slippage_counts import read_slippage_counts
from permutect.slippage.slippage_model import SlippageModel


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Learn a sample's model of polymerase slippage, the PCR and sequencing errors that add or remove "
        "repeat units of short tandem repeats (STRs), from its STR slippage counts from GATK CollectSTRSlippageCounts.  "
        "The sample's germline model from segment_pileups helps to tell slippage from germline variation in the STRs' "
        "lengths.  The output is a SlippageModel.  For now this is a skeleton: nothing is learned, and the output "
        "has the default parameters, a port of FilterMutectCalls' slippage filter."
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
    return parser.parse_args(argv)


def main_without_parsing(args):
    learn_slippage_model(
        input_table=getattr(args, constants.INPUT_NAME),
        germline_model=GermlineModel.load(getattr(args, constants.GERMLINE_MODEL_NAME)),
        output_path=getattr(args, constants.OUTPUT_NAME),
    )


def learn_slippage_model(input_table, germline_model: GermlineModel, output_path):
    """
    The slippage model of the sample of the STR slippage counts in input_table, saved to output_path.  germline_model is
    segment_pileups' model of the sample, as its tumor or as its matched normal.

    TODO: learn the model.  For now it has the default parameters.
    """
    counts = read_slippage_counts(input_table)
    print(
        f"Slippage counts of sample {counts.sample}: {counts.num_strs()} STRs spanned by "
        f"{int(counts.depths_s().sum())} fragments"
    )
    # the segmentation will give the allele fractions of germline STR alleles
    sample_segmentation(germline_model, counts.sample)
    SlippageModel().save(output_path)


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


def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == "__main__":
    main()
