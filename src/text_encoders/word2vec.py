from typing import Callable, Iterable, Optional
from pathlib import Path

import gensim.models.word2vec as w2v
import numpy as np
from omegaconf import OmegaConf

from src.settings import PAD_TOKEN, UNKNOWN_TOKEN
from src.text_encoders import BaseTextEncoder
from src.data.tokenizers import word_tokenizer


class Word2Vec(BaseTextEncoder):
    """The word2vec text encoder."""

    def __init__(self, config: OmegaConf, model: Optional[w2v.Word2Vec] = None) -> None:
        super().__init__(config)
        self.model = model
        self.tokeniser = word_tokenizer

    def save(self, path: Path) -> None:
        """Save the word2vec model.

        Args:
            path (Path): The path to save the word2vec model to.
        """
        self.model.save(str(path))
        OmegaConf.save(config=self.config, f=path.with_suffix(".yaml"))

    @classmethod
    def load(cls, path: Path) -> "Word2Vec":
        """Load the word2vec model.

        Args:
            path (Path): The path to load the word2vec model from.

        Returns:
            Word2Vec: The word2vec model.
        """
        model = w2v.Word2Vec.load(str(path))
        config = OmegaConf.load(path.with_suffix(".yaml"))
        return cls(config, model=model)

    @property
    def weights(self) -> np.ndarray:
        return self.model.wv.vectors

    @property
    def token2index(self) -> dict:
        """The token to index mapping.

        Returns:
            dict: The token to index mapping.
        """
        return self.model.wv.key_to_index

    @property
    def index2token(self) -> dict:
        """The index to token mapping.

        Returns:
            dict: The index to token mapping.
        """
        return self.model.wv.index_to_key

    @property
    def embedding_size(self) -> int:
        """The embedding size.

        Returns:
            int: The embedding size.
        """
        return self.model.vector_size

    def fit(self, texts: Iterable[str]) -> None:
        """Fit the word2vec model.

        Args:
            texts (Iterable[str]): The texts to fit the word2vec model to.
        """
        sentences = [self.tokeniser(sentence) for sentence in texts]
        self.model = w2v.Word2Vec(sentences, **self.config.model_configs)
        self.remove_rare_words(texts, self.config.min_document_count)
        vec = np.random.randn(self.embedding_size)
        self.model.wv.add_vector(UNKNOWN_TOKEN, vec)
        self.normalize_weights()
        self.model.wv.add_vector(PAD_TOKEN, np.zeros(self.embedding_size))

    def normalize_weights(self) -> None:
        """Normalize the word2vec model."""
        self.model.init_sims(replace=True)

    def remove_rare_words(self, texts: Iterable[str], min_document_count: int) -> None:
        """Remove the rare words from the word2vec model.

        Args:
            texts (Iterable[str]): The data to train on. Each element is a sentence.
            min_document_count (int): The minimum number of documents a word occurs in.
        """

        word_counts = dict()
        for sentence in texts:
            words_in_sentence = set()
            for word in sentence.split():
                if word not in word_counts:
                    word_counts[word] = 0
                if word not in words_in_sentence:
                    word_counts[word] += 1
                words_in_sentence.add(word)

        words_to_remove = [
            word
            for word, count in word_counts.items()
            if (count < min_document_count) and (word in self.model.wv.key_to_index)
        ]
        ids_to_remove = [self.model.wv.key_to_index[word] for word in words_to_remove]

        for word in words_to_remove:
            del self.model.wv.key_to_index[word]

        self.model.wv.vectors = np.delete(self.model.wv.vectors, ids_to_remove, axis=0)

        for i in sorted(ids_to_remove, reverse=True):
            del self.model.wv.index_to_key[i]

        for i, key in enumerate(self.model.wv.index_to_key):
            self.model.wv.key_to_index[key] = i
