import numpy as np
import torch

from permutect.architecture.posterior_model import PosteriorModel
from permutect.architecture.posterior_model_priors import get_ref_contexts_and_alt_bases
from permutect.data.batch import Batch
from permutect.data.compact_posterior_data import CompactPosteriorData
from permutect.data.datum import COMPRESSED_READS_ARRAY_DTYPE
from permutect.data.datum import Data
from permutect.data.datum import Datum
from permutect.data.memory_mapped_data import MemoryMappedData
from permutect.utils.enums import Variation

SEQ_LENGTH = 21  # the length of each haplotype, as in GATK data
INFO_LENGTH = 5
META = torch.device("meta")


def posterior_datum(position: int, rng) -> Datum:
    """
    A datum like those of posterior data, with plausible values of the fields that the posterior model reads, random
    haplotypes and INFO vector, and no reads
    """
    int_array = np.zeros(Data.HAPLOTYPES_START_IDX + 2 * SEQ_LENGTH, dtype=np.int16)
    int_array[Data.HAPLOTYPES_START_IDX :] = rng.integers(0, 4, size=2 * SEQ_LENGTH)
    float_array = np.zeros(Data.NUM_SCALAR_FLOAT_ELEMENTS + INFO_LENGTH, dtype=np.float16)
    float_array[Data.INFO_START_IDX :] = rng.normal(size=INFO_LENGTH)
    empty_reads = np.zeros((0, 0), dtype=COMPRESSED_READS_ARRAY_DTYPE)
    datum = Datum(int_array=int_array, float_array=float_array, reads_re=empty_reads, compressed=True)
    depth, normal_depth = int(rng.integers(20, 200)), int(rng.integers(0, 100))
    values = {
        Data.VARIANT_TYPE: int(rng.choice([Variation.SNV, Variation.INSERTION, Variation.DELETION])),
        Data.POSITION: position,
        Data.ORIGINAL_DEPTH: depth,
        Data.ORIGINAL_ALT_COUNT: int(rng.integers(1, depth)),
        Data.ORIGINAL_NORMAL_DEPTH: normal_depth,
        Data.ORIGINAL_NORMAL_ALT_COUNT: int(rng.integers(0, 3)) if normal_depth > 2 else 0,
        Data.SEQ_ERROR_LOG_LK: -rng.uniform(1, 50),
        Data.NORMAL_SEQ_ERROR_LOG_LK: -rng.uniform(1, 50),
        Data.LOG_ALLELE_FREQUENCY: np.log(rng.uniform(1e-6, 0.5)),
        Data.MAF: rng.uniform(0.1, 0.5),
        Data.NORMAL_MAF: 0.5,
        Data.CACHED_ARTIFACT_LOGIT: rng.normal(),
        Data.REPEAT_UNIT_LENGTH: 1,
        Data.REPEAT_LENGTH: int(rng.integers(0, 15)),
        Data.REPEAT_COUNT_CHANGE: -1,
    }
    for field, value in values.items():
        datum.set(field, value)
    return datum


def make_data(num_data: int) -> tuple[MemoryMappedData, list[Datum]]:
    rng = np.random.default_rng(seed=0)
    data = [posterior_datum(position=1000 + n, rng=rng) for n in range(num_data)]
    return MemoryMappedData.from_generator(iter(data), estimated_num_data=num_data, estimated_num_reads=0), data


def one_batch(batches) -> Batch:
    """the batches as one, sorted by position"""
    batches = list(batches)
    int_tensor, float_tensor = torch.cat([b.int_tensor for b in batches]), torch.cat([b.float_tensor for b in batches])
    order = torch.argsort(Batch.without_reads(int_tensor, float_tensor).get(Data.POSITION))
    return Batch.without_reads(int_tensor[order], float_tensor[order])


def test_a_pass_has_every_variant_once_with_the_fields_the_posterior_model_reads():
    memory_mapped_data, data = make_data(10)
    compact = CompactPosteriorData.from_memory_mapped_data(memory_mapped_data, batch_size=4, device="cpu")
    assert len(compact) == 3
    assert [batch.size() for batch in compact] == [4, 4, 2]

    full, compacted = one_batch([Batch(data)]), one_batch(compact)
    for field in Data:
        assert torch.equal(compacted.get(field), full.get(field)), field.name
    contexts = zip(get_ref_contexts_and_alt_bases(full), get_ref_contexts_and_alt_bases(compacted))
    for full_column, compacted_column in contexts:
        assert torch.equal(compacted_column, full_column)

    # int16 and float16, without the rest of the haplotypes or the INFO vector
    num_columns = Data.NUM_SCALAR_INT_ELEMENTS + 6 + Data.NUM_SCALAR_FLOAT_ELEMENTS
    assert compact.nbytes == 2 * num_columns * len(data)


def test_the_posterior_model_gives_the_same_posteriors_from_compact_data():
    memory_mapped_data, data = make_data(10)
    compact = CompactPosteriorData.from_memory_mapped_data(memory_mapped_data, batch_size=4, device="cpu")
    model = PosteriorModel(variant_log_prior=-10, artifact_log_prior=-10, device="cpu")
    expected = model.log_relative_posteriors_bc(one_batch([Batch(data)]))
    assert torch.equal(model.log_relative_posteriors_bc(one_batch(compact)), expected)


def test_each_pass_has_a_new_order_that_the_seed_determines():
    memory_mapped_data, _ = make_data(20)
    compact = CompactPosteriorData.from_memory_mapped_data(memory_mapped_data, batch_size=8, device="cpu")

    def positions() -> list[int]:
        return torch.cat([batch.get(Data.POSITION) for batch in compact]).tolist()

    torch.manual_seed(1)
    first, second = positions(), positions()
    assert sorted(first) == sorted(second) == list(range(1000, 1020)) and first != second
    torch.manual_seed(1)
    assert positions() == first


def test_data_too_big_for_the_device_stay_on_the_host():
    memory_mapped_data, _ = make_data(5)
    too_big = CompactPosteriorData.from_memory_mapped_data(memory_mapped_data, 2, device=META, max_device_bytes=0)
    assert too_big.int_tensor_ve.device.type == too_big.float_tensor_ve.device.type == "cpu"
    # the batches are copied to the device
    assert all(batch.int_tensor.device == batch.float_tensor.device == META for batch in too_big)

    fits = CompactPosteriorData.from_memory_mapped_data(memory_mapped_data, 2, device=META)
    assert fits.int_tensor_ve.device == fits.float_tensor_ve.device == META
