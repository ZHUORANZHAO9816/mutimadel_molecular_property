"""
Device utilities for GTpro.

Automatically selects the best available device:
CUDA > MPS (Apple Silicon) > CPU.
"""

import torch
from typing import Optional


def get_device() -> torch.device:
    """Get the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def to_device(tensor_or_module, device: Optional[torch.device] = None):
    """Move a tensor or module to the best available device."""
    if device is None:
        device = get_device()
    return tensor_or_module.to(device)