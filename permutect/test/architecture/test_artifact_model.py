import itertools

import pytest
import torch

from permutect.architecture.artifact_model import load_model
from permutect.data.batch import Batch
from permutect.data.datum import Data
from permutect.data.memory_mapped_data import MemoryMappedData
from permutect.test.test_file_names import PERMUTECT_V0_9_2_ARTIFACT_MODEL
from permutect.test.test_file_names import PREPROCESSED_DATA


def test_reads_must_match_the_counts():
    # RaggedSets don't check their lengths, since on a GPU that makes the CPU wait, so the reads of a batch are checked
    # against its counts once, when its sets are made
    model = load_model(PERMUTECT_V0_9_2_ARTIFACT_MODEL, device=torch.device("cpu"))
    batch = Batch(list(itertools.islice(MemoryMappedData.load_from_tarfile(PREPROCESSED_DATA).generate(), 8)))
    model.calculate_features(batch)

    batch.int_tensor[0, Data.ALT_COUNT.idx] += 1
    with pytest.raises(AssertionError, match="reads don't match"):
        model.calculate_features(batch)
