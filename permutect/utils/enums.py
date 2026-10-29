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

    # the value of each enum element is a one-element tuple holding a lambda whose argument is an artifact model,
    # saying how to extract the iterable of relevant parameters from that artifact model.
    # Wrapping the lambdas in a tuple is necessary; a bare function is treated as a method, not an element of the enum,
    # and so is a functools.partial from Python 3.13 on.

    # the artifact model's read_embedding parameters
    INITIAL_READ_EMBEDDING = (lambda model: model.read_embedding.parameters(),)

    # the artifact model's info_embedding parameters
    INFO = (lambda model: model.info_embedding.parameters(),)

    # the artifact model's haplotypes_cnn parameter
    HAPLOTYPES = (lambda model: model.haplotypes_cnn.parameters(),)

    # the artifact model's ref_alt_reads_encoder parameter
    GATED_MLP = (lambda model: model.ref_alt_reads_encoder.parameters(),)

    # the artifact model's reducer parameter
    REDUCER = (lambda model: model.reducer.parameters(),)

    # the artifact model's Euclidean transformation prior to clustering
    PRECLUSTERING = (lambda model: model.pre_clustering_transform.parameters(),)

    # the shape of the presumed Gaussian distribution of nonartifact reads in the clustering model.  Currently
    # the covariance is presumed diagonal, but that could change.  Careful -- the associated parameter nonartifact_stdev_e
    # is handled via parametrize.register_parametrization.
    NONARTIFACT_COVARIANCE = (lambda model: [model.feature_clustering.parametrizations.nonartifact_stdev_e.original],)

    # the directions that artifacts point away from the centroid, associated with parameter artifact_directions_ke.
    # Careful, it is also handled via parametrize.register_parametrization.
    ARTIFACT_DIRECTIONS = (lambda model: [model.feature_clustering.parametrizations.artifact_directions_ke.original],)

    # the exponentially modified Gaussian shape parameters for the 1D distribution of artifact reads along the projection
    # onto the artifact direction vectors.  This is associated with the cluster model parameters mu_k, sigma_k, and
    # lambda_k, of which lambda_k and sigma_k are handled via parametrize.register_parametrization.
    ARTIFACT_EMG = (lambda model: model.feature_clustering.artifact_emg.parameters(),)

    # the standard deviation (this is necessarily isotropic) of artifact reads in the subspace of dimensions other
    # than the artifact vectors.  Also handled via parametrize.register_parametrization.
    ARTIFACT_STDEV = (lambda model: [model.feature_clustering.parametrizations.artifact_stdev_k.original],)

    # predefined combinations of different sets for convenience

    # all parameters of the clustering model
    CLUSTERING = (lambda model: model.feature_clustering.parameters(),)

    # the entire artifact model -- this could be convenient
    WHOLE_MODEL = (lambda model: model.parameters(),)

    def __init__(self, parameters_function):
        self.parameters_function = parameters_function

    def get_parameters(self, artifact_model):
        return self.parameters_function(artifact_model)


class Label(enum.IntEnum):
    ARTIFACT = 0
    VARIANT = 1
    UNLABELED = 2
