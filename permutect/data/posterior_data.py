from __future__ import annotations

import copy
from typing import List

import numpy as np
import torch
from torch import Tensor

from permutect.data.batch import Batch
from permutect.data.datum import Datum, Data


class PosteriorDatum(Datum):

    def __init__(self, int16_array, float16_array, embedding: Tensor):
        super().__init__(int16_array, float16_array)
        self.embedding = embedding

    @classmethod
    def create(cls, int16_array, float16_array, allele_frequency: float, artifact_logit: float, maf: float, normal_maf: float, embedding: Tensor):
        result = cls(int16_array, float16_array, embedding)
        result.set(Data.ALLELE_FREQUENCY, allele_frequency)
        result.set(Data.MAF, maf)
        result.set(Data.NORMAL_MAF, normal_maf)
        result.set(Data.CACHED_ARTIFACT_LOGIT, artifact_logit)
        return result

class PosteriorBatch(Batch):

    def __init__(self, data: List[PosteriorDatum]):
        super().__init__(data)
        self.embeddings = torch.from_numpy(np.vstack([item.embedding for item in data])).float()

    def pin_memory(self):
        super().pin_memory()
        self.embeddings = self.embeddings.pin_memory()
        return self

    # dtype is just for floats!!! Better not convert the int tensor to a float accidentally!
    def copy_to(self, device, dtype):
        is_cuda = device.type == 'cuda'
        new_batch = copy.copy(self)
        new_batch.int16_data = self.int16_data.to(device, non_blocking=is_cuda)  # don't cast dtype -- needs to stay integral!
        new_batch.float16_data = self.float16_data.to(device, dtype=dtype, non_blocking=is_cuda)  # don't cast dtype -- needs to stay integral!
        new_batch.embeddings = self.embeddings.to(device=device, dtype=dtype, non_blocking=is_cuda)
        return new_batch




