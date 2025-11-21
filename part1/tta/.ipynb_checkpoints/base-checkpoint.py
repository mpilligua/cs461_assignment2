from __future__ import annotations

import abc

import torch.nn as nn


class TTAMethod(nn.Module, metaclass=abc.ABCMeta):
    """Base class for all Test-Time Adaptation (TTA) strategies.

    Subclasses should override :meth:`forward` and optionally :meth:`reset`.

    The reset method is called whenever the TTA method needs to clear any
    accumulated state (between different corruption types).
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        """Default forward simply calls the wrapped model."""
        return self.model(x)

    def reset(self) -> None:
        """Reset any state accumulated during adaptation (optional)."""
        # Default implementation is a no-op
        return None
