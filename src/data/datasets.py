from typing import Callable
import torch
from torch.utils.data import Dataset
import pyarrow as pa

from src.data.datatypes import (
    Batch,
    Lookups,
)


class BaseDataset(Dataset):
    def __init__(
        self,
        data: list[pa.RecordBatch],
        text_transform: Callable,
        label_transform: Callable,
        lookups: Lookups,
        split_name: str = "train",
    ) -> None:
        super().__init__()
        self.data = data
        self.split_name = split_name
        self.text_transform = text_transform
        self.label_transform = label_transform
        self.lookups = lookups

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        token_ids, targets, id, num_tokens, _ = self.data[idx]
        targets = self.label_transform(targets)
        return token_ids, targets, id, num_tokens

    def collate_fn(self, batch: tuple[list, list, list, list]) -> Batch:
        data, targets, ids, num_tokens = zip(*batch)
        data = self.text_transform.seq2batch(data)
        targets = self.label_transform.seq2batch(targets)
        ids = torch.tensor(ids)
        num_tokens = torch.tensor(num_tokens)
        return Batch(data=data, targets=targets, ids=ids, num_tokens=num_tokens)


class HuggingfaceDataset(Dataset):
    def __init__(
        self,
        data: list[pa.RecordBatch],
        text_transform: Callable,
        label_transform: Callable,
        lookups: Lookups,
        chunk_size: int = 128,
        split_name: str = "train",
    ) -> None:
        super().__init__()
        self.data = data
        self.split_name = split_name
        self.text_transform = text_transform
        self.label_transform = label_transform
        self.chunk_size = chunk_size
        self.lookups = lookups

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor]:
        token_ids, targets, id, num_tokens, attention_mask = self.data[idx]
        targets = self.label_transform(targets)
        return token_ids, targets, id, num_tokens, attention_mask

    def collate_fn(self, batch: tuple[list, list, list, list]) -> Batch:
        token_ids, targets, ids, num_tokens, attention_mask = zip(*batch)
        data = self.text_transform.seq2batch(token_ids, chunk_size=self.chunk_size)
        attention_mask = self.text_transform.seq2batch(
            attention_mask, chunk_size=self.chunk_size
        )
        targets = self.label_transform.seq2batch(targets)
        ids = torch.tensor(ids)
        num_tokens = torch.tensor(num_tokens)
        return Batch(
            data=data,
            targets=targets,
            ids=ids,
            num_tokens=num_tokens,
            attention_mask=attention_mask,
        )


# class CodeDescriptionDataset(BaseDataset):
#     def __getitem__(
#         self, idx: int
#     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, MetaData]:
#         example = self.split_data[idx]
#         data, targets, metadata = example.text, example.targets, example.metadata
#         code_descriptions = self.get_code_descriptions(targets)
#         data = self.text_transform(data)
#         # tokenize and concatinate the icd9 descriptions
#         code_descriptions = self.tokenise_code_descriptions(code_descriptions)
#         targets = self.label_transform(targets)
#         if self._max_length is not None:
#             data = data[: self._max_length]
#         return data, targets, code_descriptions, metadata

#     def get_code_descriptions(self, targets: Iterable[str]) -> list[str]:
#         if self.lookups.code2description is None:
#             return []
#         return [
#             self.lookups.code2description.get(target, UNKNOWN_TOKEN)
#             for target in targets
#         ]

#     def tokenise_code_descriptions(self, code_descriptions: list[str]) -> torch.Tensor:
#         if len(code_descriptions) == 0:
#             return torch.tensor([])
#         code_descriptions_tokenised = [
#             self.text_transform(code_description)
#             for code_description in code_descriptions
#         ]
#         return self.text_transform.seq2batch(code_descriptions_tokenised)

#     def collate_fn(self, batch: tuple[list, list, list, list]) -> Batch:
#         (
#             data,
#             targets,
#             icd9_description,
#             metadata,
#         ) = zip(*batch)
#         data = self.text_transform.seq2batch(data)
#         icd9_description = list(icd9_description)
#         targets = self.label_transform.seq2batch(targets)
#         return Batch(
#             data=data,
#             targets=targets,
#             code_descriptions=icd9_description,
#             metadata=metadata,
#         )

#     def get_tokenized_icd9_descriptions(self, targets: Iterable[str]) -> torch.Tensor:
#         icd9_descriptions = [
#             self.lookups.code2description.get(target, UNKNOWN_TOKEN)
#             for target in targets
#         ]
#         icd9_descriptions_tokenized = [
#             self.text_transform(icd9_description)
#             for icd9_description in icd9_descriptions
#         ]
#         return self.text_transform.seq2batch(icd9_descriptions_tokenized)
