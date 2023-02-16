from typing import Callable, Iterable, Optional
from pathlib import Path

import gensim.models.word2vec as w2v
import numpy as np
from omegaconf import OmegaConf

from src.settings import PAD_TOKEN, UNKNOWN_TOKEN


class BaseTextEncoder:
    """The base class for text encoders."""

    def __init__(self, config: OmegaConf):
        self.config = config

    def save(self, path: Path) -> None:
        """Save the text encoder.

        Args:
            path (Path): The path to save the text encoder to.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> "BaseTextEncoder":
        """Load the text encoder.

        Args:
            path (Path): The path to load the text encoder from.

        Returns:
            TextEncoder: The text encoder.
        """
        raise NotImplementedError

    def fit(self, texts: Iterable[str]) -> None:
        """Fit the text encoder.

        Args:
            texts (Iterable[str]): The texts to fit the text encoder to.
        """
        raise NotImplementedError

    @property
    def token2index(self) -> dict:
        """The token to index mapping.

        Returns:
            dict: The token to index mapping.
        """
        raise NotImplementedError

    @property
    def index2token(self) -> dict:
        """The index to token mapping.

        Returns:
            dict: The index to token mapping.
        """
        raise NotImplementedError

    @property
    def weights(self) -> np.ndarray:
        """The weights of the text encoder.

        Returns:
            np.ndarray: The weights of the text encoder.
        """
        raise NotImplementedError
