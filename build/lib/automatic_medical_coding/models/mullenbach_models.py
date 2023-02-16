from pathlib import Path
from typing import Callable, Iterable, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.autograd import Variable
from gensim.models.word2vec import Word2Vec

from src.models import BaseModel
from src.metrics import (
    Recall,
    Precision,
    F1Score,
    LossMetric,
    ExactMatchRatio,
    AUC,
    Recall_K,
    Precision_K,
)


class MullenbachBaseModel(BaseModel):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        word2vec: Optional[Word2Vec] = None,
        lmbda: float = 0,
        embed_dropout: float = 0.5,
        embed_size: int = 100,
        padding_idx: int = 0,
    ):
        super(BaseModel, self).__init__()
        self.embed_size = embed_size
        self.embed_drop = nn.Dropout(p=embed_dropout)
        self.lmbda = lmbda

        self.loss = F.binary_cross_entropy_with_logits

        # make embedding layer
        if word2vec is not None:
            print("loading pretrained embeddings...")
            weights = torch.FloatTensor(word2vec.wv.vectors)
            self.embed = nn.Embedding.from_pretrained(weights, padding_idx=padding_idx)
        else:
            self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)

    def encoder(self, x):
        x = self.embed(x)
        x = self.embed_drop(x)
        return x.transpose(1, 2)

    def training_step(self, x, targets, icd9_description=None):
        logits = self(x)
        loss = self.loss(logits, targets)
        prob = torch.sigmoid(logits)
        metrics = [
            LossMetric(loss),
            Recall(prob, targets, average="micro", threshold=0.5),
            Precision(prob, targets, average="micro", threshold=0.5),
            F1Score(prob, targets, average="micro", threshold=0.5),
            Recall(prob, targets, average="macro", threshold=0.5),
            Precision(prob, targets, average="macro", threshold=0.5),
            F1Score(prob, targets, average="macro", threshold=0.5),
            ExactMatchRatio(prob, targets, threshold=0.5),
        ]
        return loss, metrics, prob

    def validation_step(self, x, targets, icd9_description=None):
        logits = self(x)
        loss = self.loss(logits, targets)
        prob = torch.sigmoid(logits)
        metrics = [
            LossMetric(loss),
            F1Score(prob, targets, average="micro", threshold=0.5),
            F1Score(prob, targets, average="macro", threshold=0.5),
            Recall(prob, targets, average="micro", threshold=0.5),
            Recall(prob, targets, average="macro", threshold=0.5),
            Precision(prob, targets, average="micro", threshold=0.5),
            Precision(prob, targets, average="macro", threshold=0.5),
            # AUC(prob, targets, average="micro"),
            # AUC(prob, targets, average="macro"),
            ExactMatchRatio(prob, targets, threshold=0.5),
            Recall_K(prob, targets, k=8),
            Recall_K(prob, targets, k=15),
            Precision_K(prob, targets, k=8),
            Precision_K(prob, targets, k=15),
        ]
        return loss, metrics, prob

    def test_step(self, x, targets, icd9_description=None):
        logits = self(x)
        loss = self.loss(logits, targets)
        prob = torch.sigmoid(logits)
        metrics = [
            LossMetric(loss),
            F1Score(prob, targets, average="micro", threshold=0.5),
            F1Score(prob, targets, average="macro", threshold=0.5),
            Recall(prob, targets, average="micro", threshold=0.5),
            Recall(prob, targets, average="macro", threshold=0.5),
            Precision(prob, targets, average="micro", threshold=0.5),
            Precision(prob, targets, average="macro", threshold=0.5),
            AUC(prob, targets, average="micro"),
            AUC(prob, targets, average="macro"),
            ExactMatchRatio(prob, targets, threshold=0.5),
            Recall_K(prob, targets, k=8),
            Recall_K(prob, targets, k=15),
            Precision_K(prob, targets, k=8),
            Precision_K(prob, targets, k=15),
        ]
        return loss, metrics, prob


class VanillaConv(MullenbachBaseModel):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        word2vec: Optional[Word2Vec] = None,
        lmbda: float = 0,
        embed_dropout: float = 0.2,
        embed_size: int = 100,
        padding_idx: int = 0,
        num_filters: int = 500,
        kernel_size: int = 4,
    ):
        super().__init__(
            vocab_size=vocab_size,
            num_classes=num_classes,
            word2vec=word2vec,
            lmbda=lmbda,
            embed_dropout=embed_dropout,
            embed_size=embed_size,
            padding_idx=padding_idx,
        )

        self.conv = nn.Conv1d(self.embed_size, num_filters, kernel_size=kernel_size)
        xavier_uniform_(self.conv.weight)

        self.fc = nn.Linear(num_filters, num_classes)
        xavier_uniform_(self.fc.weight)

    def decoder(self, x):
        x = self.conv(x)
        x = torch.tanh(x)
        x = F.max_pool1d(x, kernel_size=x.size(2))
        x = x.squeeze(dim=2)
        return self.fc(x)

    def forward(self, x):
        representations = self.encoder(x)
        return self.decoder(representations)


class CAML(MullenbachBaseModel):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        word2vec: Optional[Word2Vec] = None,
        lmbda: float = 0,
        embed_dropout: float = 0.2,
        embed_size: int = 100,
        padding_idx: int = 0,
        num_filters: int = 500,
        kernel_size: int = 4,
    ):
        super().__init__(
            vocab_size=vocab_size,
            num_classes=num_classes,
            word2vec=word2vec,
            lmbda=lmbda,
            embed_dropout=embed_dropout,
            embed_size=embed_size,
            padding_idx=padding_idx,
        )

        self.conv = nn.Conv1d(
            self.embed_size,
            num_filters,
            kernel_size=kernel_size,
            padding=int(math.floor(kernel_size / 2)),
        )
        xavier_uniform_(self.conv.weight)

        # weight matrix for attention
        self.U = nn.Linear(num_filters, num_classes)
        xavier_uniform_(self.U.weight)

        self.fc = nn.Linear(num_filters, num_classes)
        xavier_uniform_(self.fc.weight)

    def decoder(self, x):
        x = self.conv(x)
        x = torch.tanh(x)
        x = self.attention(x)
        return self.fc.weight.mul(x).sum(2).add(self.fc.bias)

    def forward(self, x):
        representations = self.encoder(x)
        return self.decoder(representations)

    def attention(self, x):
        alpha = torch.softmax(self.U.weight.matmul(x), dim=2)
        return alpha.matmul(x.transpose(1, 2))


class DRCAML(CAML):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        word2vec: Optional[Word2Vec] = None,
        lmbda: float = 0,
        embed_dropout: float = 0.2,
        embed_size: int = 100,
        padding_idx: int = 0,
        num_filters: int = 500,
        kernel_size: int = 4,
    ):
        super().__init__(
            vocab_size=vocab_size,
            num_classes=num_classes,
            word2vec=word2vec,
            lmbda=lmbda,
            embed_dropout=embed_dropout,
            embed_size=embed_size,
            padding_idx=padding_idx,
        )

        self.conv = nn.Conv1d(
            self.embed_size,
            num_filters,
            kernel_size=kernel_size,
            padding=int(math.floor(kernel_size / 2)),
        )
        xavier_uniform_(self.conv.weight)

        # weight matrix for attention
        self.U = nn.Linear(num_filters, num_classes)
        xavier_uniform_(self.U.weight)

        self.fc = nn.Linear(num_filters, num_classes)
        xavier_uniform_(self.fc.weight)

    def decoder(self, x):
        x = self.conv(x)
        x = torch.tanh(x)
        x = self.attention(x)
        return self.fc.weight.mul(x).sum(2).add(self.fc.bias)

    def forward(self, x):
        representations = self.encoder(x)
        return self.decoder(representations)
