"""
Base Tent (Test-Time Entropy minimization) implementation.

Clean implementation of the original Tent method without any modifications.
Reference: "Tent: Fully Test-Time Adaptation by Entropy Minimization"
           https://arxiv.org/abs/2006.10726
"""

import torch
import torch.nn as nn
import torch.jit

from tta.base import TTAMethod

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def collect_params(model: nn.Module) -> tuple:
    """Collect all trainable parameters.
    
    Tent adapts only affine parameters (scale and shift) of batch norm layers.
    """
    params = []
    names = []
    
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
            if "layer1" in nm:  # Adapt only last block's BN layers
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # affine parameters
                        params.append(p)
                        names.append(f"{nm}.{np}")
    
    return params, names


def configure_model(model: nn.Module, reset_stats: bool = False, no_stats: bool = True):
    """Configure model for Tent adaptation.
    
    Args:
        model: The neural network model
        reset_stats: Whether to reset batch norm running statistics
        no_stats: Whether to compute batch statistics instead of using running statistics
    """
    model.train()  # Set to training mode to enable batch statistics
    model.requires_grad_(False)  # Disable gradients for all parameters
    
    # Configure batch norm layers
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) and "layer1" in nm:
            # Enable gradients for affine parameters (scale and shift)
            if m.weight is not None:
                m.weight.requires_grad_(True)
            if m.bias is not None:
                m.bias.requires_grad_(True)
            
            # Reset or use batch statistics
            if reset_stats:
                m.reset_running_stats()
            
            if no_stats:
                # Use batch statistics instead of running statistics
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            else:
                # Use running statistics (track during adaptation)
                m.track_running_stats = True
                m.momentum = 0.1  # Standard momentum for running stats


@torch.enable_grad()
def forward_and_adapt(x, model, optimizer, steps: int = 1, 
                     clip_norm: float = 0.0, debug: bool = False,
                     log_wandb: bool = False, batch_idx: int = 0):
    """Forward and adapt a batch using entropy minimization.
    
    Args:
        x: Input batch
        model: Neural network model
        optimizer: Optimizer for adaptation
        steps: Number of adaptation steps per batch
        clip_norm: Gradient clipping threshold (0 = no clipping)
        debug: Whether to print debug information
        log_wandb: Whether to log to wandb
        batch_idx: Current batch index for logging
        
    Returns:
        outputs: Model predictions
        stats: Dictionary with loss statistics
    """
    # Perform adaptation steps
    for step in range(steps):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(x)
        
        # Compute entropy loss
        entropy_per_sample = softmax_entropy(outputs)
        loss = entropy_per_sample.mean()
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if clip_norm > 0.0:
            params = []
            for g in optimizer.param_groups:
                params.extend(g.get("params", []))
            torch.nn.utils.clip_grad_norm_(params, clip_norm)
        
        # Update parameters
        optimizer.step()
        
        if debug and step > 0:
            print(f"  Step {step}/{steps}: entropy={loss.item():.4f}")
    
    # Final forward pass for predictions
    with torch.no_grad():
        outputs = model(x)
        final_entropy = softmax_entropy(outputs).mean()
    
    # Logging
    if log_wandb and WANDB_AVAILABLE:
        try:
            wandb.log({
                "tent/entropy": final_entropy.item(),
                "tent/batch_idx": batch_idx,
            })
        except Exception as e:
            if debug:
                print(f"Failed to log to wandb: {e}")
    
    return outputs, {
        'entropy': final_entropy.item(),
        'loss': final_entropy.item()
    }


class TentBase(TTAMethod):
    """Tent base class for test-time adaptation."""
    
    def __init__(self, model, steps=1, episodic=False,
                 optim="Adam", lr=0.00025, beta=0.9, weight_decay=0.0,
                 reset_stats=False, no_stats=True, clip_norm=0.0,
                 debug=False, log_wandb=False):
        """
        Args:
            model: Neural network model
            steps: Number of adaptation steps per batch
            episodic: Whether to reset the model state after each batch
            optim: Optimizer name (Adam or SGD)
            lr: Learning rate
            beta: Momentum parameter for Adam
            weight_decay: Weight decay
            reset_stats: Whether to reset batch norm running statistics
            no_stats: Whether to compute batch statistics instead of running statistics
            clip_norm: Gradient clipping threshold
            debug: Whether to print debug information
            log_wandb: Whether to log to wandb
        """
        super().__init__(model)
        self.steps = steps
        self.episodic = episodic
        self.reset_stats = reset_stats
        self.no_stats = no_stats
        self.clip_norm = clip_norm
        self.debug = debug
        self.log_wandb = log_wandb
        self.batch_idx = 0
        
        # Configure model
        configure_model(model, reset_stats=reset_stats, no_stats=no_stats)
        
        # Collect parameters for optimization
        params, names = collect_params(model)
        
        # Create optimizer
        if optim == "Adam":
            self.optimizer = torch.optim.Adam(
                params,
                lr=lr,
                betas=(beta, 0.999),
                weight_decay=weight_decay
            )
        elif optim == "SGD":
            self.optimizer = torch.optim.SGD(
                params,
                lr=lr,
                momentum=beta,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optim}")
        
        # Store initial model state for episodic adaptation
        self.model_state = {
            k: v.clone().detach()
            for k, v in model.state_dict().items()
        }
        
        if debug:
            print(f"\nTent Base initialized:")
            print(f"  Adaptable parameters: {len(params)}")
            print(f"  Optimizer: {optim}")
            print(f"  Learning rate: {lr}")
            print(f"  Steps per batch: {steps}")
            print(f"  Episodic: {episodic}")
            print(f"  Gradient clipping: {clip_norm if clip_norm > 0 else 'disabled'}")
            if debug:
                print(f"\nAdaptable layers:")
                for name in names:
                    print(f"    {name}")
    
    def forward(self, x):
        """Adapt and predict."""
        if self.episodic:
            self.reset()
        
        outputs, stats = forward_and_adapt(
            x, self.model, self.optimizer,
            steps=self.steps,
            clip_norm=self.clip_norm,
            debug=self.debug,
            log_wandb=self.log_wandb,
            batch_idx=self.batch_idx
        )
        
        self.batch_idx += 1
        return outputs
    
    def reset(self):
        """Reset model to initial state."""
        if self.model_state is not None:
            self.model.load_state_dict(self.model_state)
        self.batch_idx = 0
