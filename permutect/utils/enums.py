import enum


class Variation(enum.IntEnum):
    SNV = 0
    INSERTION = 1
    DELETION = 2
    BIG_INSERTION = 3
    BIG_DELETION = 4

    @staticmethod
    def get_type(ref_allele: str, alt_allele: str):
        diff = len(alt_allele) - len(ref_allele)
        if diff == 0:
            return Variation.SNV
        elif diff > 0:
            return Variation.BIG_INSERTION if diff > 1 else Variation.INSERTION
        else:
            return Variation.BIG_DELETION if diff < -1 else Variation.DELETION


class Call(enum.IntEnum):
    SOMATIC = 0
    ARTIFACT = 1
    SEQ_ERROR = 2
    GERMLINE = 3
    NORMAL_ARTIFACT = 4


class Epoch(enum.IntEnum):
    TRAIN = 0
    VALID = 1
    TEST = 2


class ParameterSet(enum.Enum):
    """
    Types of parameter sets to be re-fit at test time.  Generally for domain adaptation but not necessarily so.

    These are additive; hence the idea is to be able to specify different sets of parameter sets with a
    --parameter_set flag that can be specified multiple times.
    """

    # the artifact model's read_embedding parameters
    INITIAL_READ_EMBEDDING = "INITIAL_READ_EMBEDDING"

    # the artifact model's info_embedding parameters
    INFO = "INFO"

    # the artifact model's haplotypes_cnn parameter
    HAPLOTYPES = "HAPLOTYPES"

    # the artifact model's ref_alt_reads_encoder parameter
    GATED_MLP = "GATED_MLP"

    # the artifact model's reducer parameter
    REDUCER = "REDUCER"

    # the clustering model's global read translation parameter read_translation_e
    TRANSLATION = "TRANSLATION"

    # the clustering model's global rotation parameter read_rotation_ee -- when unfreezing this or setting its use_gradient = True etc.
    # be aware that it is parameterized via orthogonal and might need special care
    ROTATION = "ROTATION"

    # the shape of the presumed Gaussian distribution of nonartifact reads in the clustering model.  Currently
    # the covariance is presumed diagonal, but that could change.  Careful -- the associated parameter nonartifact_stdev_e
    # is handled via parametrize.register_parametrization.
    NONARTIFACT_COVARIANCE = "NONARTIFACT_COVARIANCE"

    # the directions that artifacts point away from the centroid, associated with parameter artifact_directions_ke.
    # Careful, it is also handled via parametrize.register_parametrization.
    ARTIFACT_DIRECTIONS = "ARTIFACT_DIRECTIONS"

    # the exponentially modified Gaussian shape parameters for the 1D distribution of artifact reads along the projection
    # onto the artifact direction vectors.  This is associated with the cluster model parameters mu_k, sigma_k, and
    # lambda_k, of which lambda_k and sigma_k are handled via parametrize.register_parametrization.
    ARTIFACT_PROJECTIONS = "ARTIFACT_PROJECTIONS"

    # the standard deviation (this is necessarily isotropic) of artifact reads in the subspace of dimensions other
    # than the artifact vectors.  Also handled via parametrize.register_parametrization.
    ARTIFACT_STDEV = "ARTIFACT_STDEV"

    # predefined combinations of different sets for convenience

    # all parameters of the clustering model
    ALL_CLUSTERING = "ALL_CLUSTERING"

    # the entire artifact model -- this could be convenient
    WHOLE_MODEL = "WHOLE_MODEL"

    @staticmethod
    def get_parameter_set(set_str: str):
        for parameter_set in ParameterSet:
            if set_str == parameter_set.name:
                return parameter_set

        raise ValueError("parameter set type is invalid: %s" % set_str)


class Label(enum.IntEnum):
    ARTIFACT = 0
    VARIANT = 1
    UNLABELED = 2

    @staticmethod
    def get_label(label_str: str):
        for label in Label:
            if label_str == label.name:
                return label

        raise ValueError("label is invalid: %s" % label_str)

    @staticmethod
    def is_label(label_str: str):
        for label in Label:
            if label_str == label.name:
                return True

        return False
