from typing import Optional, Callable

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.settings import PAD_TOKEN, UNKNOWN_TOKEN
from src.data.datasets import BaseDataset


class MullenbachDataset(BaseDataset):
    def __init__(
        self,
        examples: list[tuple[str, set[str], dict]],
        transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
        max_length: Optional[int] = None,
        pad_index: int = 0,
        split_name: str = "train",
        code2description: dict[str, str] = None,
    ) -> None:
        super().__init__(
            examples, transform, label_transform, max_length, pad_index, split_name
        )
        self.code2description = code2description

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data, label, metadata = self.examples[idx]
        icd9_description = self.code2description.get(label, UNKNOWN_TOKEN)
        if self._transform is not None:
            data = self._transform(data)
            icd9_description = self._transform(icd9_description)
        if self._label_transform is not None:
            label = self._label_transform(label)
        if self.max_length_tokens is not None:
            data = data[: self.max_length_tokens]
        return data, label, icd9_description, metadata

    def collate_fn(self, batch) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
        (
            data,
            label,
            icd9_description,
            metadata,
        ) = zip(*batch)
        data = pad_sequence(data, padding_value=self.pad_index, batch_first=True)
        icd9_description = pad_sequence(
            icd9_description, padding_value=self.pad_index, batch_first=True
        )
        label = torch.stack(label)
        return data, label, icd9_description, metadata
