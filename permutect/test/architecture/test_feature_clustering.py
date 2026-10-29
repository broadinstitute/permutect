from permutect.architecture.exponentially_modified_gaussian import STDEV_CONSTRAINT
from permutect.architecture.feature_clustering import FeatureClustering


def test_standard_deviations_share_one_constraint():
    # the Gaussian and exponentially-modified Gaussian standard deviations used to have separate, identical bounds
    clustering = FeatureClustering(feature_dimension=3, num_artifact_clusters=2)
    stdevs = [
        (clustering, "nonartifact_stdev_e"),
        (clustering, "artifact_stdev_k"),
        (clustering.artifact_emg, "sigma_k"),
    ]
    for module, name in stdevs:
        assert module.parametrizations[name][0] is STDEV_CONSTRAINT, name
