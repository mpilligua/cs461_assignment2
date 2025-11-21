from __future__ import annotations

import torch

from tta.base import TTAMethod


class Unadapted(TTAMethod):
    """Baseline model without any adaptation."""

    def __init__(self, model, use_eval_mode: bool = True):
        super().__init__(model)
        if use_eval_mode:
            self.model.eval()

    def forward(self, x):
        with torch.no_grad():
            return self.model(x)
