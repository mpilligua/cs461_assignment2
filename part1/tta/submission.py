from __future__ import annotations

from tta.base import TTAMethod
    
from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

# Tent implementation 
class Submission(TTAMethod):
    def __init__(
        self,
        model,
        steps: int = 1,
        episodic: bool = False,
        optim: str = "Adam",
        lr: float = 1e-4,
        beta: float = 0.9,
        weight_decay: float = 0.0,
        reset_stats: bool = False,
        no_stats: bool = True,
        clip_norm: float = 1.0,
        debug: bool = True,
    ):
        # configure model for tent-style adaptation (train mode, BN settings)
        model = configure_model(model, reset_stats=reset_stats, no_stats=no_stats)
        super().__init__(model)

        # collect BN affine params
        params, names = collect_params(self.model)
        print("optimizable parameters for Tent adaptation:")
        for n, p in zip(names, params):
            print(f"  {n} | shape={tuple(p.shape)} | requires_grad={p.requires_grad} | device={p.device}")

        # build optimizer using provided hyperparameters
        if optim.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                params=params, lr=lr, betas=(beta, 0.999), weight_decay=weight_decay
            )
        else:
            self.optimizer = torch.optim.SGD(
                params=params, lr=lr, momentum=0.9, weight_decay=weight_decay
            )

        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.clip_norm = clip_norm
        self.debug = debug

        # debug: print optimizer/param summary
        if self.debug:
            try:
                print(f"[Submission] optimizer: {self.optimizer.__class__.__name__}")
                print(f"[Submission] num params in optimizer: {sum(len(g.get('params', [])) for g in self.optimizer.param_groups)}")
                print("[Submission] BN params:")
                for n, p in zip(names, params):
                    print(f"  {n} | shape={tuple(p.shape)} | requires_grad={p.requires_grad} | device={p.device}")
            except Exception:
                pass

        # snapshot model/optimizer state for episodic resets
        self.model_state, self.optimizer_state = copy_model_and_optimizer(
            self.model, self.optimizer
        )

    def forward(self, x):
        if self.episodic:
            self.reset()

        # optional debug: snapshot BN params before/after adaptation to verify updates
        if self.debug:
            params_before = [p.detach().cpu().clone() for p in collect_params(self.model)[0]]

        for _ in range(self.steps):
            outputs = forward_and_adapt(
                x, self.model, self.optimizer, clip_norm=self.clip_norm, debug=self.debug
            )

        if self.debug:
            params_after = [p.detach().cpu().clone() for p in collect_params(self.model)[0]]
            try:
                total_change = 0.0
                for a, b in zip(params_before, params_after):
                    total_change += (a - b).abs().sum().item()
                print(f"[Submission] total BN param change this forward: {total_change:.6e}")
            except Exception:
                pass

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer, clip_norm: float = 0.0, debug: bool = False):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model(x)
    # adapt
    loss = softmax_entropy(outputs).mean(0)
    if debug:
        try:
            print(f"[tent] entropy loss={loss.item():.6f}")
        except Exception:
            pass
    loss.backward()
    if clip_norm and clip_norm > 0.0:
        # collect params from optimizer and clip gradients
        params = []
        for g in optimizer.param_groups:
            params.extend(g.get("params", []))
        torch.nn.utils.clip_grad_norm_(params, clip_norm)
    optimizer.step()
    optimizer.zero_grad()
    return outputs


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model, reset_stats: bool = False, no_stats: bool = True):
    """Configure model for use with tent.

    Args:
        model: model to configure
        reset_stats: if True, call reset_running_stats() on each BN
        no_stats: if True, disable tracking running stats and clear them
    """
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statistics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # optionally reset running stats
            if reset_stats:
                try:
                    m.reset_running_stats()
                except Exception:
                    pass
            if no_stats:
                # force use of batch stats and disable running buffers
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
