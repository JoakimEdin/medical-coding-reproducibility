import logging
from typing import Union

import torch.nn as nn
import torch


LOGGER = logging.getLogger(name=__file__)


class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.device = None

    def encoder(self, x):
        raise NotImplementedError

    def decoder(self, x):
        raise NotImplementedError

    def get_loss(self, logits, targets):
        raise NotImplementedError

    def training_step(self, batch) -> dict[str, torch.Tensor]:
        data, targets = batch.data, batch.targets
        logits = self(data)
        loss = self.get_loss(logits, targets)
        logits = torch.sigmoid(logits)
        return {"logits": logits, "loss": loss, "targets": targets}

    def validation_step(self, batch) -> dict[str, torch.Tensor]:
        data, targets = batch.data, batch.targets
        logits = self(data)
        loss = self.get_loss(logits, targets)
        logits = torch.sigmoid(logits)
        return {"logits": logits, "loss": loss, "targets": targets}

    def to(self, device: Union[torch.device, str]) -> nn.Module:
        self.device = device
        return super().to(device)
