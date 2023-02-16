from typing import Optional
import math

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from gensim.models.word2vec import Word2Vec
import numpy as np

from src.models import BaseModel
from src.models.modules.attention import CAMLAttention
from src.text_encoders import BaseTextEncoder


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dropout: float,
    ):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=int(math.floor(kernel_size / 2)),
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.Tanh(),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=int(math.floor(kernel_size / 2)),
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            ),
            nn.BatchNorm1d(out_channels),
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = torch.tanh(out)
        out = self.dropout(out)
        return out


class MultiResCNN(BaseModel):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        num_filters: int,
        kernel_sizes: list[int],
        text_encoder: BaseTextEncoder,
        embed_dropout: float,
        pad_index: int = 0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_size = text_encoder.embedding_size
        self.embed_drop = nn.Dropout(p=embed_dropout)
        self.device = torch.device("cpu")
        self.num_classes = num_classes

        self.loss = F.binary_cross_entropy_with_logits

        print("loading pretrained embeddings...")
        weights = torch.FloatTensor(text_encoder.weights)
        self.embed = nn.Embedding.from_pretrained(
            weights, padding_idx=pad_index, freeze=False
        )

        self.convs = nn.ModuleList()
        self.channels = nn.ModuleList()  # convolutional layers in parallel
        for kernel_size in kernel_sizes:
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        self.embed_size,
                        self.embed_size,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=int(math.floor(kernel_size / 2)),
                        bias=False,
                    ),
                    nn.Tanh(),
                    ResidualBlock(
                        self.embed_size, num_filters, kernel_size, 1, embed_dropout
                    ),
                )
            )

        self.attention = CAMLAttention(
            input_size=num_filters * len(self.convs), num_classes=num_classes
        )

    def forward(self, text):
        embedded = self.embed(text)
        embedded = self.embed_drop(embedded)
        embedded = embedded.transpose(1, 2)
        outputs = []
        for conv in self.convs:
            x = conv(embedded)
            outputs.append(x)
        x = torch.cat(outputs, dim=1)
        x = self.attention(x)
        return x

    def get_loss(self, logits, targets):
        return self.loss(logits, targets)
