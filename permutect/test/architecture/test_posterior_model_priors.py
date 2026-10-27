import numpy as np
import torch

from permutect.architecture.posterior_model_priors import PosteriorModelPriors
from permutect.data.batch import Batch
from permutect.data.datum import RAW_READS_ARRAY_DTYPE
from permutect.data.datum import Data
from permutect.data.datum import Datum
from permutect.utils.enums import Call


def make_batch(allele_frequencies, values: dict = None) -> Batch:
    """
    a batch with the given allele frequencies, and other data fields set to the given values or else 0
    """
    data = []
    for _ in allele_frequencies:
        # ref and alt haplotypes of length 3 follow the scalar int elements
        datum = Datum(
            int_array=np.zeros(Data.HAPLOTYPES_START_IDX + 6),
            float_array=np.zeros(Data.NUM_SCALAR_FLOAT_ELEMENTS),
            reads_re=np.zeros((0, 2), dtype=RAW_READS_ARRAY_DTYPE),
        )
        for data_field, value in (values or {}).items():
            datum.set(data_field, value)
        data.append(datum)
    batch = Batch(data)
    # set the batch's float32 log allele frequencies directly, since a Datum would round them to float16
    log_allele_frequencies = torch.log(torch.tensor(allele_frequencies, dtype=torch.float32))
    batch.float_tensor[:, Data.LOG_ALLELE_FREQUENCY.idx] = log_allele_frequencies
    return batch


def test_germline_log_prior_is_accurate_for_small_allele_frequencies():
    allele_frequencies = [2e-8, 5e-8, 1e-7, 1e-6, 1e-5, 1e-3, 0.1, 0.5, 1.0]
    priors = PosteriorModelPriors(
        variant_log_prior=-10, artifact_log_prior=-10, no_germline_mode=False, device=torch.device("cpu")
    )
    log_priors_bc = priors.log_priors_bc(make_batch(allele_frequencies))

    # before the softmax normalization, the sequencing error log prior is 0 and the germline log prior is the log
    # probability of not being hom ref, 1 - (1 - af)^2
    afs = torch.tensor(allele_frequencies, dtype=torch.float32).double()
    expected = torch.log1p(-torch.square(1 - afs))
    germline_log_prior = log_priors_bc[:, Call.GERMLINE] - log_priors_bc[:, Call.SEQ_ERROR]
    torch.testing.assert_close(germline_log_prior.double(), expected, rtol=0, atol=1e-4)
