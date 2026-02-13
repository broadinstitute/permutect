from __future__ import annotations

import copy
from typing import List

import numpy as np
import torch
from torch import IntTensor, Tensor

from permutect.data.batch import Batch
from permutect.data.datum import Datum, Data


class PosteriorDatum(Datum):

    ALLELE_FREQUENCY = 0
    ARTIFACT_LOGIT = 1
    MAF = 2
    NORMAL_MAF = 3

    def __init__(self, datum_array, float_array, embedding: Tensor):
        super().__init__(datum_array)
        self.embedding = embedding
        self.float_array = float_array


    @classmethod
    def create(cls, datum_array, allele_frequency: float, artifact_logit: float, maf: float, normal_maf: float, embedding: Tensor):
        float_array = np.zeros(4, dtype=np.float16)
        float_array[PosteriorDatum.ALLELE_FREQUENCY] = allele_frequency
        float_array[PosteriorDatum.ARTIFACT_LOGIT] = artifact_logit
        float_array[PosteriorDatum.MAF] = maf
        float_array[PosteriorDatum.NORMAL_MAF] = normal_maf
        return cls(datum_array, float_array, embedding)

    def get_artifact_logit(self) -> float:
        return self.float_array[self.__class__.ARTIFACT_LOGIT]


class PosteriorBatch(Batch):

    def __init__(self, data: List[PosteriorDatum]):
        super().__init__(data)
        self.embeddings = torch.from_numpy(np.vstack([item.embedding for item in data])).float()
        self.float_tensor = torch.from_numpy(np.vstack([item.float_array for item in data])).float()

    def pin_memory(self):
        super().pin_memory()
        self.embeddings = self.embeddings.pin_memory()
        self.float_tensor = self.float_tensor.pin_memory()
        return self

    # dtype is just for floats!!! Better not convert the int tensor to a float accidentally!
    def copy_to(self, device, dtype):
        is_cuda = device.type == 'cuda'
        new_batch = copy.copy(self)
        new_batch.data = self.data.to(device, non_blocking=is_cuda)  # don't cast dtype -- needs to stay integral!
        new_batch.embeddings = self.embeddings.to(device=device, dtype=dtype, non_blocking=is_cuda)
        new_batch.float_tensor = self.float_tensor.to(device=device, dtype=dtype, non_blocking=is_cuda)
        return new_batch

    def get_allele_frequencies(self) -> Tensor:
        return self.float_tensor[:, PosteriorDatum.ALLELE_FREQUENCY]

    def get_artifact_logits(self) -> Tensor:
        return self.float_tensor[:, PosteriorDatum.ARTIFACT_LOGIT]

    def get_mafs(self) -> Tensor:
        return self.float_tensor[:, PosteriorDatum.MAF]

    def get_normal_mafs(self) -> Tensor:
        return self.float_tensor[:, PosteriorDatum.NORMAL_MAF]

    def get_original_normal_ref_counts(self) -> IntTensor:
        return self.get(Data.ORIGINAL_NORMAL_DEPTH) - self.get(Data.ORIGINAL_NORMAL_ALT_COUNT)


