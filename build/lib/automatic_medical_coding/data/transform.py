import math
from typing import Callable, Optional, Any
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import numpy as np
import gensim.models.word2vec as w2v

from src.settings import UNKNOWN_TOKEN, PAD_TOKEN


class Transform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Any):
        return self.transform(x)

    def transform(self, x):
        raise NotImplementedError()


class TextEncoder(Transform):
    def __init__(self, tokenizer: Callable, unk_token: str = UNKNOWN_TOKEN):
        """A class to encode text into indices and back.

        Args:
            tokenizer (Callable): A function that takes a string and returns a list of tokens.
            unk_token (str, optional): Token used for all out of vocabulary words. Defaults to UNKNOWN_TOKEN.
        """
        super().__init__()
        self.min_frequency = 0
        self.tokenizer = tokenizer
        self.token2index = {}
        self.index2token = {}
        self.unknown_token = unk_token

    def transform(self, text: str) -> torch.Tensor:
        """Transform a text into a list of indices.

        Args:
            text (str): The text to be transformed.

        Returns:
            torch.Tensor: A tensor of indices.
        """
        tokens = self.tokenizer(text)
        return torch.tensor([self.token_to_index(token) for token in tokens])

    def inverse_transform(self, indices: torch.Tensor) -> list[str]:
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

    def use_gensim_word2vec_tokenmap(self, word2vec: w2v.Word2Vec) -> None:
        """Use the token map from a gensim word2vec model.

        Args:
            word2vec (w2v.Word2Vec): A gensim word2vec model.
        """
        self.token2index = word2vec.wv.key_to_index
        self.index2token = word2vec.wv.index_to_key

    def fit(
        self,
        texts: list[str],
        special_tokens: Optional[list[str]] = [UNKNOWN_TOKEN, PAD_TOKEN],
        min_frequency: int = 0,
    ) -> None:
        """Fit the text encoder to a list of texts.

        Args:
            texts (list[str]): A list of texts.
            special_tokens (Optional[list[str]], optional): A list of special tokens. Defaults to [UNKNOWN_TOKEN, PAD_TOKEN].
            min_frequency (int, optional): The minimum frequency of a token to be included in the vocabulary. Defaults to 0.
        """
        self.min_frequency = min_frequency
        counter = Counter()
        for text in texts:
            counter.update(self.tokenizer(text))

        for index, (token, count) in enumerate(counter.items()):
            if count >= min_frequency:
                self.token2index[token] = index
                self.index2token[index] = token

        for token in special_tokens:
            self.add_special_token(token)

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


class OneHotEncoder(Transform):
    def __init__(self) -> None:
        """One hot encoder for labels"""
        super().__init__()
        self.label2index = {}
        self.index2label = {}

    def fit(self, labels_list: set[str]) -> None:
        """Fit the encoder to the labels in the dataset

        Args:
            labels_list (set[str]): List of labels
        """
        for index, label in enumerate(labels_list):
            self.label2index[label] = index
            self.index2label[index] = label

    @property
    def num_classes(self) -> int:
        """Number of classes supported by the encoder

        Returns:
            int: Number of classes
        """
        return len(self.label2index)

    def get_classes(self) -> list[str]:
        """Get the list of classes supported by the encoder"""
        return list(self.label2index.keys())

    def transform(self, labels: set[str]) -> torch.Tensor:
        """Transform a set of labels into a one-hot encoded tensor

        Args:
            labels (set[str]): Set of labels

        Returns:
            torch.Tensor: One-hot encoded tensor
        """
        output_tensor = torch.zeros(self.num_classes)
        for label in labels:
            if label in self.label2index:
                output_tensor[self.label2index[label]] = 1
        return output_tensor

    def inverse_transform(self, output_tensor: torch.Tensor) -> set[str]:
        """Transform a one-hot encoded tensor into a set of labels

        Args:
            output_tensor (torch.Tensor): One-hot encoded tensor

        Returns:
            set[str]: Set of labels
        """
        labels = set()
        for index, value in enumerate(output_tensor):
            if value == 1:
                labels.add(self.index2label[index])
        return labels
