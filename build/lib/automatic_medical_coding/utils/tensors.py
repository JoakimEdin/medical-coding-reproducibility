from typing import Any, Union
import torch


def detach(x: Union[torch.Tensor, Any]):
    """Detach a tensor from the computational graph"""
    if isinstance(x, torch.Tensor):
        return x.detach()
    return x
