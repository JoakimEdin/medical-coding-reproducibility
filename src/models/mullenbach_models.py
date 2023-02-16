import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models.word2vec import Word2Vec
from torch.nn.init import xavier_uniform_

from src.models import BaseModel
from src.models.modules.attention import CAMLAttention
from src.text_encoders import BaseTextEncoder


class MullenbachBaseModel(BaseModel):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        text_encoder: BaseTextEncoder,
        embed_dropout: float = 0.5,
        pad_index: int = 0,
        **kwargs
    ):
        super(BaseModel, self).__init__()
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

    def encoder(self, x):
        x = self.embed(x)
        x = self.embed_drop(x)
        return x.transpose(1, 2)

    def get_loss(self, logits, targets):
        return self.loss(logits, targets)


class VanillaConv(MullenbachBaseModel):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        text_encoder: BaseTextEncoder,
        embed_dropout: float = 0.2,
        pad_index: int = 0,
        num_filters: int = 500,
        kernel_size: int = 4,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            num_classes=num_classes,
            text_encoder=text_encoder,
            embed_dropout=embed_dropout,
            pad_index=pad_index,
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
        text_encoder: Optional[Word2Vec] = None,
        embed_dropout: float = 0.2,
        pad_index: int = 0,
        num_filters: int = 500,
        kernel_size: int = 4,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            num_classes=num_classes,
            text_encoder=text_encoder,
            embed_dropout=embed_dropout,
            pad_index=pad_index,
        )

        self.conv = nn.Conv1d(
            self.embed_size,
            num_filters,
            kernel_size=kernel_size,
            padding=int(math.floor(kernel_size / 2)),
        )
        xavier_uniform_(self.conv.weight)

        self.attention = CAMLAttention(num_filters, num_classes)

    def decoder(self, x):
        x = self.conv(x)
        return self.attention(x)

    def forward(self, x):
        representations = self.encoder(x)
        return self.decoder(representations)


class VanillaRNN(MullenbachBaseModel):
    """
    General RNN - can be LSTM or GRU, uni/bi-directional
    """

    def __init__(
        self,
        rnn_dim: int,
        cell_type: str,
        num_layers: int,
        vocab_size: int,
        num_classes: int,
        bidirectional=False,
        text_encoder: Optional[Word2Vec] = None,
        embed_dropout: float = 0.2,
        pad_index: int = 0,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            num_classes=num_classes,
            text_encoder=text_encoder,
            embed_dropout=embed_dropout,
            pad_index=pad_index,
        )

        self.rnn_dim = rnn_dim
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        # recurrent unit
        if self.cell_type == "lstm":
            self.rnn = nn.LSTM(
                self.embed_size,
                math.floor(self.rnn_dim / self.num_directions),
                self.num_layers,
                bidirectional=bidirectional,
            )
        else:
            self.rnn = nn.GRU(
                self.embed_size,
                math.floor(self.rnn_dim / self.num_directions),
                self.num_layers,
                bidirectional=bidirectional,
            )
        # linear output
        self.final = nn.Linear(self.rnn_dim, num_classes)

    def forward(self, x):
        # clear hidden state, reset batch size at the start of each batch
        init_hidden = self.init_hidden(x.size(0))
        # embed
        embeds = self.embed(x).transpose(0, 1)
        # apply RNN
        _, hidden = self.rnn(embeds, init_hidden)

        # get final hidden state in the appropriate way
        last_hidden = hidden[0] if self.cell_type == "lstm" else hidden
        last_hidden = (
            last_hidden[-1]
            if self.num_directions == 1
            else last_hidden[-2:].transpose(0, 1).contiguous().view(x.size(0), -1)
        )

        return self.final(last_hidden)

    def init_hidden(self, batch_size):
        h_0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            math.floor(self.rnn_dim / self.num_directions),
        ).to(self.device)

        if self.cell_type != "lstm":
            return h_0

        c_0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            math.floor(self.rnn_dim / self.num_directions),
        ).to(self.device)
        return (h_0, c_0)
