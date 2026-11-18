import argparse

from permutect import constants
from permutect.read_orientation.f1r2_counts_reader import read_f1r2_counts
from permutect.read_orientation.read_orientation_model import ReadOrientationModel
from permutect.read_orientation.read_orientation_model import learn_priors
from permutect.segmentation.germline_model import GermlineModel


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Learn a sample's read orientation artifact model, as GATK LearnReadOrientationModel does, from its "
        "F1R2 counts from Mutect2 --f1r2-tar-gz or CollectF1R2Counts, eg those of each scatter of a run.  Each 3-base "
        "reference context and its reverse complement get a prior over 8 artifact states, F1R2 or F2R1 for each alt "
        "base, and 4 non-artifact states, learned by expectation maximization.  The output is a ReadOrientationModel "
        "of the priors and of the distributions of each state's alt count and alt F1R2 count, for filter_variants "
        "--read_orientation_model."
    )
    parser.add_argument(
        "--" + constants.INPUT_NAME,
        nargs="+",
        required=True,
        help="one or more tar.gz files of the F1R2 counts of one sample, eg from the scatters of a Mutect2 run, which "
        "are summed",
    )
    parser.add_argument("--" + constants.OUTPUT_NAME, required=True, help="path to the output ReadOrientationModel")
    parser.add_argument(
        "--" + constants.GERMLINE_MODEL_NAME,
        help="optional GermlineModel from segment_pileups of the same sample, whose segmentation of the sample by "
        "minor allele fraction and emission model give the likelihoods of germline hets, accounting for hets in regions "
        "of loss of heterozygosity.  Without it, germline hets have GATK's likelihoods.",
    )
    parser.add_argument(
        "--" + constants.CONVERGENCE_THRESHOLD_NAME,
        type=float,
        default=1e-4,
        help="EM stops when the prior changes by at most this much (Euclidean distance) in an iteration",
    )
    parser.add_argument(
        "--" + constants.MAX_EM_ITERATIONS_NAME,
        type=int,
        default=20,
        help="maximum number of iterations of expectation maximization",
    )
    return parser.parse_args(argv)


def main_without_parsing(args):
    germline_model_path = getattr(args, constants.GERMLINE_MODEL_NAME)
    learn_read_orientation_model(
        input_tar_gzs=getattr(args, constants.INPUT_NAME),
        output_path=getattr(args, constants.OUTPUT_NAME),
        convergence_threshold=getattr(args, constants.CONVERGENCE_THRESHOLD_NAME),
        max_em_iterations=getattr(args, constants.MAX_EM_ITERATIONS_NAME),
        germline_model=None if germline_model_path is None else GermlineModel.load(germline_model_path),
    )


def learn_read_orientation_model(
    input_tar_gzs,
    output_path,
    convergence_threshold: float = 1e-4,
    max_em_iterations: int = 20,
    germline_model: GermlineModel = None,
):
    """
    Given germline_model, segment_pileups' model of the counts' sample, germline hets have the likelihoods of its emission
    model at the minor allele fractions of its segmentation
    """
    counts_by_sample = read_f1r2_counts(input_tar_gzs, None if germline_model is None else germline_model.tumor)
    # Mutect2 collects the F1R2 counts of tumor samples, and Permutect calls one tumor
    if len(counts_by_sample) != 1:
        raise ValueError(
            f"A read orientation model is of one sample, but the F1R2 counts are of {len(counts_by_sample)} samples: "
            + ", ".join(sorted(counts_by_sample))
        )
    ((sample, counts),) = counts_by_sample.items()
    if germline_model is not None and germline_model.tumor.sample != sample:
        raise ValueError(
            f"The germline model is of sample {germline_model.tumor.sample}, but the F1R2 counts are of sample {sample}"
        )
    emission_model = None if germline_model is None else germline_model.tumor.emission_model
    model = ReadOrientationModel(sample, learn_priors(counts, convergence_threshold, max_em_iterations, emission_model))
    model.save(output_path)


def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == "__main__":
    main()
