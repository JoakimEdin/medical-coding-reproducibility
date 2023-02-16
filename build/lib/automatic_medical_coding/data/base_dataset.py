from typing import Optional, Callable

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.settings import PAD_TOKEN


class BaseDataset(Dataset):
    def __init__(
        self,
        examples: list[tuple[str, set[str], dict]],
        transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
        max_length: Optional[int] = None,
        pad_index: int = 0,
        split_name: str = "train",
    ) -> None:
        super().__init__()
        self.split_name = split_name
        self._examples = examples
        self._transform = transform
        self._label_transform = label_transform
        self._max_length = max_length
        self.pad_index = pad_index

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data, label, metadata = self._examples[idx]
        if self._transform is not None:
            data = self._transform(data)
        if self._label_transform is not None:
            label = self._label_transform(label)
        if self._max_length is not None:
            data = data[: self._max_length]
        return data, label, metadata

    def collate_fn(self, batch) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
        data, label, metadata = zip(*batch)
        data = pad_sequence(data, padding_value=self.pad_index, batch_first=True)
        label = torch.stack(label)
        return data, label, metadata
