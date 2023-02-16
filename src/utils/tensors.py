import math
from typing import Any, Union

import torch
from omegaconf import OmegaConf


def detach(x: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
    """Detach a tensor from the computational graph"""
    if isinstance(x, torch.Tensor):
        return x.detach()
    return x


def detach_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Detach a batch from the computational graph"""
    return {k: detach(v) for k, v in batch.items()}


def get_dummy_batch(config: OmegaConf, device: torch.device) -> torch.Tensor:
    """Get a dummy batch to initialize the model weights.

    Args:
        config (OmegaConf): Config object.
        device (torch.device): Device to use.

    Returns:
        torch.Tensor: Dummy batch.
    """
    if config.data.max_length is not None:
        max_length = config.data.max_length
    elif hasattr(config.text_transform.configs, "max_length"):
        if config.text_transform.configs.max_length is not None:
            max_length = config.text_transform.configs.max_length
    else:
        max_length = 10000

    if hasattr(config.dataset.configs, "chunk_size"):
        return torch.zeros(
            (
                config.dataloader.max_batch_size,
                math.ceil(max_length / config.dataset.configs.chunk_size),
                config.dataset.configs.chunk_size,
            ),
            device=device,
        ).long()

    return torch.zeros(
        (config.dataloader.max_batch_size, max_length), device=device
    ).long()
