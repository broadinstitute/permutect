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
