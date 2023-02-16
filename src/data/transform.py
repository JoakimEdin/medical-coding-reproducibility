import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer

from src.data.tokenizers import word_tokenizer
from src.settings import PAD_TOKEN, UNKNOWN_TOKEN


class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.unknown_token = UNKNOWN_TOKEN
        self.pad_token = PAD_TOKEN
        self.pad_index = 0
        self.unknown_index = None

    def forward(self, x: Any):
        return self.transform(x)

    def transform(self, x):
        raise NotImplementedError()

    def inverse_transform(self, x):
        raise NotImplementedError()

    def seq2batch(self, sequence: Sequence[torch.Tensor]) -> torch.Tensor:
        """A batching function for the Transform classes. Used in the collate function of the dataset.

        Args:
            sequence (Sequence[torch.Tensor]): A Sequence of vectors of equal lengths.

        Returns:
            torch.Tensor: A tensor of vectors.
        """
        return torch.stack(sequence)

    def batch_transform(self, texts: list[str]) -> list[list[int]]:
        """Transform a batch of texts into a batch of indices.

        Args:
            texts (list[str]): A batch of texts.

        Returns:
            torch.Tensor: A batch of indices.
        """
        raise NotImplementedError


class TokenSequence(Transform):
    def __init__(
        self,
        min_frequency: int = 0,
    ):
        """A class to encode text into indices and back.

        Args:
            min_frequency (int, optional): The minimum frequency of a token to be included in the vocabulary. Defaults to 0.
        """
        super().__init__()
        self.min_frequency = min_frequency
        self.tokenizer = word_tokenizer
        self.token2index = {}
        self.index2token = {}
        self.file_name = "token2index.json"

    def transform(self, text: str) -> torch.Tensor:
        """Transform a text into a list of indices.

        Args:
            text (str): The text to be transformed.

        Returns:
            torch.Tensor: A tensor of indices.
        """
        tokens = self.tokenizer(text)
        return torch.tensor([self.token_to_index(token) for token in tokens])

    def inverse_transform(self, indices: list[str]) -> list[str]:
        """Transform a list of indices into a list of tokens.

        Args:
            indices (torch.Tensor): A tensor of indices.

        Returns:
            list[str]: A list of tokens.
        """
        return [self.index2token[index] for index in indices]

    def token_to_index(self, token: str) -> Optional[int]:
        """Transform a token into an index.

        Args:
            token (str): The token to be transformed.

        Returns:
            int: The index of the token.
        """
        if self.unknown_token is not None:
            return self.token2index.get(token, self.token2index[self.unknown_token])

        return self.token2index.get(token)

    def index_to_token(self, index: int) -> str:
        """Transform an index into a token.

        Args:
            index (int): The index to be transformed.

        Returns:
            str: The token of the index.
        """
        return self.index2token[index]

    def set_tokenmap(
        self, token2index: dict[str, int], index2token: dict[int, str]
    ) -> None:
        self.token2index = token2index
        self.index2token = index2token
        self.unknown_index = self.token2index[UNKNOWN_TOKEN]
        self.pad_index = self.token2index[PAD_TOKEN]

    def fit(
        self,
        texts: list[str],
        special_tokens: Optional[list[str]] = [UNKNOWN_TOKEN, PAD_TOKEN],
    ) -> None:
        """Fit the text encoder to a list of texts.

        Args:
            texts (list[str]): A list of texts.
            special_tokens (Optional[list[str]], optional): A list of special tokens. Defaults to [UNKNOWN_TOKEN, PAD_TOKEN].
        """
        counter = Counter()
        for text in texts:
            counter.update(self.tokenizer(text))

        for index, (token, count) in enumerate(counter.items()):
            if count >= self.min_frequency:
                self.token2index[token] = index
                self.index2token[index] = token

        # Add special tokens
        self.add_special_token(UNKNOWN_TOKEN)
        self.add_special_token(PAD_TOKEN)
        self.unknown_index = self.token2index[UNKNOWN_TOKEN]
        self.pad_index = self.token2index[PAD_TOKEN]

    def add_special_token(self, token: str) -> None:
        """Add a special token to the vocabulary.

        Args:
            token (str): The token to be added.
        """
        self.token2index[token] = len(self.token2index)
        self.index2token[len(self.index2token)] = token

    @property
    def vocab_size(self) -> int:
        """The size of the vocabulary.

        Returns:
            int: The size of the vocabulary.
        """
        return len(self.token2index)

    def seq2batch(self, sequence: Sequence[torch.Tensor]) -> torch.Tensor:
        """Batch a sequence of vectors of different lengths. Use the pad_index to pad the vectors.

        Args:
            sequences (Sequence[torch.Tensor]): A sequence of sequences.

        Returns:
            torch.Tensor: A batched tensor.
        """
        return torch.nn.utils.rnn.pad_sequence(
            sequence, batch_first=True, padding_value=self.pad_index
        )

    def save(self, path: str) -> None:
        """Saves the token2index as a json file"""
        path = Path(path) / self.file_name
        with open(path, "w") as f:
            json.dump(self.token2index, f)

    def load(self, path: str) -> None:
        """Loads the token2index from a json file"""
        path = Path(path) / self.file_name
        with open(path, "r") as f:
            self.token2index = json.load(f)
            self.index2token = {v: k for k, v in self.token2index.items()}

    def batch_transform(self, texts: list[str]) -> list[list[int]]:
        """Transform a batch of texts into a batch of indices.

        Args:
            texts (list[str]): A batch of texts.

        Returns:
            list[list[int]]: A batch of indices.
        """
        return [
            [self.token_to_index(token) for token in self.tokenizer(text)]
            for text in texts
        ]


class HuggingFaceTokenizer(Transform):
    def __init__(
        self,
        model_path: str,
        add_special_token: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        **kwargs
    ):
        """A class to encode text into indices and back.

        Args:
            model_name (str): The name of the huggingface tokenizer.
        """
        super().__init__()
        self.add_special_token = add_special_token
        self.padding = padding
        self.max_length = max_length
        self.truncation = truncation
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)

    def transform(self, text: str) -> torch.Tensor:
        """Transform a text into a list of indices.

        Args:
            text (str): The text to be transformed.

        Returns:
            torch.Tensor: A tensor of indices.
        """
        return self.tokenizer(
            text,
            padding=self.padding,
            truncation=self.truncation,
            add_special_tokens=self.add_special_token,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def batch_transform(self, texts: list[str]) -> list[list[int]]:
        """Transform a batch of texts into a batch of indices.

        Args:
            texts (list[str]): A batch of texts.

        Returns:
            list[list[int]]: A batch of indices.
        """
        return self.tokenizer(
            texts,
            padding=self.padding,
            truncation=self.truncation,
            add_special_tokens=self.add_special_token,
            max_length=self.max_length,
        )["input_ids"]

    def inverse_transform(self, indices: list[str]) -> list[str]:
        """Transform a list of indices into a list of tokens.

        Args:
            indices (torch.Tensor): A tensor of indices.

        Returns:
            list[str]: A list of tokens.
        """

        return self.tokenizer.decode(indices)

    def seq2batch(
        self, sequence: Sequence[torch.Tensor], chunk_size: int = 0
    ) -> torch.Tensor:
        """Batch a sequence of vectors of different lengths. Use the pad_index to pad the vectors.

        Args:
            sequences (Sequence[torch.Tensor]): A sequence of sequences.

        Returns:
            torch.Tensor: A batched tensor.
        """
        if chunk_size == 0:
            return torch.nn.utils.rnn.pad_sequence(
                sequence, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
        sequence = list(sequence)
        batch_size = len(sequence)
        max_length = max([len(x) for x in sequence])
        if max_length % chunk_size != 0:
            max_length = max_length + (chunk_size - max_length % chunk_size)

        # pad first sequence to the desired length
        sequence[0] = torch.nn.functional.pad(
            sequence[0],
            (0, max_length - len(sequence[0])),
            value=self.tokenizer.pad_token_id,
        )
        return (
            torch.nn.utils.rnn.pad_sequence(
                sequence, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            .contiguous()
            .view((batch_size, -1, chunk_size))
        )

    @property
    def vocab_size(self) -> int:
        """The size of the vocabulary.

        Returns:
            int: The size of the vocabulary.
        """
        return len(self.tokenizer)

    def fit(self, texts: list[str]) -> None:
        pass

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass


class BOW(Transform):
    def __init__(
        self,
        binary: bool = True,
        max_features: int = None,
        lowercase: bool = True,
        max_df: float = 1.0,
        **kwargs: dict
    ):
        super().__init__()
        """A class to encode text into bag of words vectors.

        Args:
            unk_token (str, optional): Token used for all out of vocabulary words. Defaults to UNKNOWN_TOKEN.
        """
        self.tokenizer = word_tokenizer
        self.vectorizer = CountVectorizer(
            tokenizer=self.tokenizer,
            binary=binary,
            max_features=max_features,
            lowercase=lowercase,
            max_df=max_df,
        )
        self.file_name = "vectorizer.pkl"

    def transform(self, x: str) -> torch.Tensor:
        """Transform a text into a bag of words vector.

        Args:
            x (str): The text to be transformed.

        Returns:
            torch.Tensor: A bag of words vector.
        """
        bow_sparse = self.vectorizer.transform([x])
        bow = bow_sparse.toarray()[0]
        return torch.from_numpy(bow).float()

    def fit(self, texts: list[str], **kwargs: dict) -> None:
        """Fit the vectorizer to a list of texts.

        Args:
            texts (list[str]): A list of texts.
        """
        self.vectorizer.fit(texts, **kwargs)

    @property
    def vocab_size(self) -> int:
        """The size of the vocabulary.

        Returns:
            int: The size of the vocabulary.
        """
        return len(self.vectorizer.vocabulary_)

    def save(self, path: str) -> None:
        """Save the vectorizer to a pickle file."""
        path = Path(path) / self.file_name
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self, path: str) -> None:
        """Load the vectorizer from a pickle file."""
        path = Path(path) / self.file_name
        with open(path, "rb") as f:
            self.vectorizer = pickle.load(f)


class OneHotEncoder(Transform):
    def __init__(self) -> None:
        """One hot encoder for targets"""
        super().__init__()
        self.target2index = {}
        self.index2target = {}
        self.file_name = "target2index.json"

    def fit(self, targets: set[str]) -> None:
        """Fit the encoder to all the targets in the dataset. That also includes the validation and test set.

        Args:
            targets (set[str]): List of targets
        """
        for index, target in enumerate(targets):
            self.target2index[target] = index
            self.index2target[index] = target

    @property
    def num_classes(self) -> int:
        """Number of classes supported by the encoder

        Returns:
            int: Number of classes
        """
        return len(self.target2index)

    def get_classes(self) -> list[str]:
        """Get the list of classes supported by the encoder. The classes are sorted by their index."""
        return [self.index2target[index] for index in range(self.num_classes)]

    def transform(self, targets: Iterable[str]) -> torch.Tensor:
        """Transform a set of targets into a one-hot encoded tensor

        Args:
            targets (set[str]): Set of targets

        Returns:
            torch.Tensor: One-hot encoded tensor
        """
        output_tensor = torch.zeros(self.num_classes)
        for label in targets:
            if label in self.target2index:
                output_tensor[self.target2index[label]] = 1
        return output_tensor

    def inverse_transform(self, output_tensor: torch.Tensor) -> set[str]:
        """Transform a one-hot encoded tensor into a set of targets

        Args:
            output_tensor (torch.Tensor): One-hot encoded tensor

        Returns:
            set[str]: Set of targets
        """

        indices = torch.nonzero(output_tensor).squeeze(0).numpy()
        return set([self.index2target[int(index)] for index in indices])

    def get_indices(self, targets: Iterable[str]) -> torch.Tensor:
        """Get the indices of the targets

        Args:
            targets (Iterable[str]): Set of targets

        Returns:
            torch.Tensor: Indices of the targets
        """
        indices = torch.zeros(len(targets), dtype=torch.long)
        for index, label in enumerate(targets):
            if label in self.target2index:
                indices[index] = self.target2index[label]
        return indices

    def save(self, path: str) -> None:
        """Save target2index as a json file

        Args:
            path (str): path to save the json file
        """
        path = Path(path) / self.file_name
        with open(path, "w") as f:
            json.dump(self.target2index, f)

    def load(self, path: str) -> None:
        """Load target2index from a json file

        Args:
            path (str): path of the directory containing the json file
        """
        path = Path(path) / self.file_name
        with open(path, "r") as f:
            self.target2index = json.load(f)
            self.index2target = {v: k for k, v in self.target2index.items()}
