import numpy as np

from permutect.data.position_table import PositionTable
from permutect.utils.enums import Call


class PosteriorResult:
    """
    simple container class for holding results of the posterior model and other things that get output to the VCF and
    tensorboard analysis
    """

    def __init__(
        self,
        artifact_logit: float,
        posterior_probabilities,
        error_probability: float,
        log_priors,
        spectra_lls,
        normal_lls,
        label,
        alt_count,
        depth,
        var_type,
        embedding,
        simulated_embedding,
        simulated_source: str,
        without_damage: tuple[list, float] = None,
    ):
        self.artifact_logit = artifact_logit
        self.posterior_probabilities = posterior_probabilities
        self.error_probability = error_probability
        self.log_priors = log_priors
        self.spectra_lls = spectra_lls
        self.normal_lls = normal_lls
        self.label = label
        self.alt_count = alt_count
        self.depth = depth
        self.variant_type = var_type
        self.embedding = embedding
        # the embedding of a twin simulated from the feature clustering model, and the component it came from
        self.simulated_embedding = simulated_embedding
        self.simulated_source = simulated_source
        # given a read orientation model, the posterior probabilities and error probability as if SEQ_ERROR had Mutect2's
        # likelihood alone, without damage
        self.without_damage = without_damage


class PosteriorResultTable(PositionTable):
    """
    The posterior results of the data, one row per datum, on disk.  The rows are sorted like a VCF sorted by position,
    and each record's row is found in one pass alongside it, the last appended of several with the same keys, as when the
    results were kept in a dict.

    The values that go into the VCF are float64, which holds the float32 or float64 values of the posterior model
    exactly, so that the VCF is the same as when they were kept in memory as Python floats.
    """

    def __init__(self, embedding_dimension: int):
        num_calls = len(Call)
        super().__init__(
            [
                ("posterior_probabilities", np.float64, (num_calls,)),
                ("log_priors", np.float64, (num_calls,)),
                ("spectra_lls", np.float64, (num_calls,)),
                ("normal_lls", np.float64, (num_calls,)),
                ("error_probability", np.float64),
                ("artifact_logit", np.float64),
                # given a read orientation model; otherwise unused
                ("probabilities_without_damage", np.float64, (num_calls,)),
                ("error_probability_without_damage", np.float64),
                ("embedding", np.float32, (embedding_dimension,)),
                ("simulated_embedding", np.float32, (embedding_dimension,)),
                ("simulated_component", np.int64),
                ("label", np.int64),
                ("alt_count", np.int64),
                ("depth", np.int64),
                ("variant_type", np.int64),
            ]
        )
