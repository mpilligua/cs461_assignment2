from __future__ import annotations

from copy import deepcopy

import torch.nn as nn

from tta.base import TTAMethod


class TestTimeNorm(TTAMethod):
    """Test-time normalization with batch statistics."""

    def __init__(
        self,
        model,
        eps: float = 1e-5,
        momentum: float = 0.1,
        reset_stats: bool = False,
        no_stats: bool = False,
    ):
        model = configure_model(model, eps, momentum, reset_stats, no_stats)
        super().__init__(model)
        self.model_state = deepcopy(self.model.state_dict())

    def reset(self) -> None:
        self.model.load_state_dict(self.model_state, strict=True)

def configure_model(
    model: nn.Module, eps: float, momentum: float, reset_stats: bool, no_stats: bool
) -> nn.Module:
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            m.eps = eps
            m.momentum = momentum
            if reset_stats:
                m.reset_running_stats()
            if no_stats:
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    return model
