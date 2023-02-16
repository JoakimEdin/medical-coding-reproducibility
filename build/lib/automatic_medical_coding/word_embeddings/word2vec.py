from typing import Callable, Iterable, Optional
from pathlib import Path

import gensim.models.word2vec as w2v
import numpy as np

from src.settings import PAD_TOKEN


def train_gensim_word2vec(
    data: Iterable[str], model_path: Path, tokenizer: Callable, **kwargs
):
    """Split the sentences into words and train a word2vec model.

    Args:
        data (Iterable[str]): The data to train on. Each element is a sentence.
        model_path (Path): The path to save the model.
        tokenizer (Callable): The tokenizer to split the sentences into words.
        **kwargs: The arguments for the word2vec model.
    """
    words = [tokenizer(sentence) for sentence in data]
    model = w2v.Word2Vec(words, **kwargs)
    model.save(str(model_path))
    return model


def load_gensim_word2vec(model_path: Path) -> w2v.Word2Vec:
    """Load a word2vec model.

    Args:
        model_path (Path): The path to the model.

    Returns:
        gensim.models.word2vec.Word2Vec: The word2vec model.
    """
    return w2v.Word2Vec.load(str(model_path))


def normalize_gensim_word2vec_model(model: w2v.Word2Vec) -> None:
    """Normalize the word2vec model.

    Args:
        model (w2v.Word2Vec): The word2vec model.
    """
    model.init_sims(replace=True)


def get_gensim_word2vec_model(
    data: Iterable[str],
    model_path: Path,
    tokenizer: Callable,
    pad_token: Optional[str] = PAD_TOKEN,
    special_tokens: Optional[list] = None,
    **kwargs
) -> w2v.Word2Vec:
    """Get the word2vec model. If the model does not exist, train it.

    Args:
        data (Iterable[str]): The data to train on. Each element is a sentence.
        model_path (Path): The path to the model.
        tokenizer (Callable): The tokenizer to split the sentences into words.
        pad_token (Optional[str], optional): The token to use for padding. Defaults to PAD_TOKEN.
        special_tokens (Optional[list], optional): The special tokens to add to the model. Defaults to None.
        **kwargs: The arguments for the word2vec model.

    Returns:
        gensim.models.word2vec.Word2Vec: The word2vec model.
    """
    if model_path.exists():
        word2vec = load_gensim_word2vec(model_path)
    else:
        word2vec = train_gensim_word2vec(data, model_path, tokenizer, **kwargs)

    embedding_dim = word2vec.vector_size

    if special_tokens is not None:
        for token in special_tokens:
            vec = np.random.randn(embedding_dim)
            word2vec.wv.add_vector(token, vec)

    normalize_gensim_word2vec_model(word2vec)

    if pad_token is not None:
        word2vec.wv.add_vector(pad_token, np.zeros(embedding_dim))

    return word2vec
