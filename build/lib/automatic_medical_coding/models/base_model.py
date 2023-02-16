import logging

import torch.nn as nn
import torch


LOGGER = logging.getLogger(name=__file__)


class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self._kwargs = kwargs

    def encoder(self, x):
        raise NotImplementedError

    def decoder(self, x):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError
