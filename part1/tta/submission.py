"""
Tent with Consistency-Finetuned Model + Discriminator Guidance.

Uses a ResNet that was finetuned to produce consistent early layer
representations across different corruptions, combined with a discriminator
that guides activations toward "clean-like" distributions.

Components:
1. Consistency-finetuned ResNet: Early layers trained for corruption invariance
2. Entropy minimization (Tent): Adapts BN parameters at test time
3. Discriminator guidance: Pushes first conv activations toward clean distribution
"""

import os
import torch
import torch.nn as nn
import torch.jit

from tta.base import TTAMethod


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def collect_bn_params(model: nn.Module, adapt_layers: str = "layer1") -> tuple:
    """Collect batch norm parameters from specified layers."""
    params = []
    names = []
    
    for nm, m in model.named_modules():
        should_adapt = False
        
        if adapt_layers == "layer1":
            should_adapt = nm.startswith('layer1') or nm == 'bn1'
        elif adapt_layers == "layer2":
            should_adapt = nm.startswith('layer1') or nm.startswith('layer2') or nm == 'bn1'
        elif adapt_layers == "layer4":
            should_adapt = nm.startswith('layer4')
        elif adapt_layers == "all":
            should_adapt = True
        
        if should_adapt and isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    
        # add the first conv layer parameters
        elif "layer1" in nm and isinstance(m, nn.Conv2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    
    return params, names


def configure_model(model: nn.Module, adapt_layers: str = "layer1"):
    """Configure model for adaptation."""
    model.eval()
    model.requires_grad_(False)
    
    for nm, m in model.named_modules():
        should_adapt = False
        
        if adapt_layers == "layer1":
            should_adapt = nm.startswith('layer1') or nm == 'bn1'
        elif adapt_layers == "layer2":
            should_adapt = nm.startswith('layer1') or nm.startswith('layer2') or nm == 'bn1'
        elif adapt_layers == "layer4":
            should_adapt = nm.startswith('layer4')
        elif adapt_layers == "all":
            should_adapt = True
        
        if should_adapt and isinstance(m, nn.BatchNorm2d):
            m.train()
            if m.weight is not None:
                m.weight.requires_grad_(True)
            if m.bias is not None:
                m.bias.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None


class ActivationDiscriminator(nn.Module):
    """MLP that scores how 'clean' an activation is (0=corrupted, 1=clean)."""
    
    def __init__(self, input_dim, hidden_dims=[512, 256, 128]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ActivationHook:
    """Hook to capture activations from a layer."""
    def __init__(self):
        self.activation = None
    
    def __call__(self, module, input, output):
        self.activation = output.view(output.size(0), -1)


class Submission(TTAMethod):
    """Tent with consistency-finetuned early layers + discriminator guidance."""
    
    def __init__(self, model,
                 finetuned_path: str = "ckpts/finetuned_resnet50.pt",
                 discriminator_path: str = "ckpts/activation_discriminator.pt",
                 adapt_layers: str = "layer1",
                 steps: int = 1,
                 episodic: bool = False,
                 optim: str = "Adam",
                 lr: float = 0.00025,
                 beta: float = 0.9,
                 weight_decay: float = 0.0,
                 clip_norm: float = 0.0,
                 disc_weight: float = 0.1,
                 use_discriminator: bool = True,
                 debug: bool = False):
        """
        Args:
            model: Neural network model (will be replaced with finetuned weights if available)
            finetuned_path: Path to finetuned model checkpoint (state_dict only)
            discriminator_path: Path to discriminator checkpoint
            adapt_layers: Which BN layers to adapt
            steps: Number of adaptation steps per batch
            episodic: Whether to reset model after each batch
            optim: Optimizer type
            lr: Learning rate
            beta: Momentum parameter
            weight_decay: Weight decay
            clip_norm: Gradient clipping threshold
            disc_weight: Weight for discriminator loss (0 to disable)
            use_discriminator: Whether to use discriminator guidance
            debug: Whether to print debug info
        """
        # Try to load finetuned weights (state_dict format)
        try:
            state_dict = torch.load(finetuned_path, map_location='cpu', weights_only=False)
            # Handle both formats: direct state_dict or dict with 'model_state_dict'
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
                if debug:
                    print(f"✓ Loaded finetuned model from {finetuned_path}")
                    print(f"  Accuracy: {state_dict.get('accuracy', 'N/A')}")
            else:
                model.load_state_dict(state_dict)
                if debug:
                    print(f"✓ Loaded finetuned model from {finetuned_path}")
        except Exception as e:
            print(f"Warning: Could not load finetuned model: {e}")
            print("  Using original pretrained model")
        
        super().__init__(model)
        
        self.steps = steps
        self.episodic = episodic
        self.clip_norm = clip_norm
        self.debug = debug
        self.batch_idx = 0
        self.adapt_layers = adapt_layers
        self.disc_weight = disc_weight
        self.use_discriminator = use_discriminator
        
        # Load discriminator if enabled
        self.discriminator = None
        self.activation_hook = None
        self.hook_handle = None
        
        if use_discriminator and disc_weight > 0:
            try:
                disc_ckpt = torch.load(discriminator_path, map_location='cpu', weights_only=False)
                input_dim = disc_ckpt['input_dim']
                self.discriminator = ActivationDiscriminator(input_dim)
                self.discriminator.load_state_dict(disc_ckpt['model_state_dict'])
                self.discriminator.eval()
                for p in self.discriminator.parameters():
                    p.requires_grad_(False)
                
                # Register hook on first conv
                self.activation_hook = ActivationHook()
                for nm, m in model.named_modules():
                    if isinstance(m, nn.Conv2d):
                        self.hook_handle = m.register_forward_hook(self.activation_hook)
                        m.train()
                        if debug:
                            print(f"✓ Loaded discriminator from {discriminator_path}")
                            print(f"  Val accuracy: {disc_ckpt.get('val_acc', 'N/A'):.2f}%")
                            print(f"  Hook registered on: {nm}")
                        break
            except Exception as e:
                if debug:
                    print(f"Warning: Could not load discriminator: {e}")
                    print("  Continuing without discriminator guidance")
                self.use_discriminator = False
        
        # Configure model
        configure_model(model, adapt_layers=adapt_layers)
        
        # Collect parameters
        params, names = collect_bn_params(model, adapt_layers=adapt_layers)
        
        for name in names:
            print(f"✓ Adaptable parameter: {name}")

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
        
        # Store initial state for reset (always, not just for episodic)
        # This is needed when eval.py calls reset() between corruptions
        self.model_state = {
            k: v.clone().detach()
            for k, v in model.state_dict().items()
        }
        self.model.train()
        self.initial_optimizer_state = self.optimizer.state_dict()
        
        if debug:
            print(f"\nSubmission initialized:")
            print(f"  Adaptable layers: {adapt_layers}")
            print(f"  BN parameters: {len(params)}")
            print(f"  Learning rate: {lr}")
            print(f"  Steps per batch: {steps}")
            print(f"  Discriminator: {'enabled' if self.discriminator is not None else 'disabled'}")
            print(f"  Disc weight: {disc_weight}")
    
    def forward(self, x):
        """Adapt and predict with entropy minimization + discriminator guidance."""
        if self.episodic:
            self.reset()
        
        # Move discriminator to same device as input
        if self.discriminator is not None and next(self.discriminator.parameters()).device != x.device:
            self.discriminator = self.discriminator.to(x.device)
        
        for _ in range(self.steps):
            self.optimizer.zero_grad()
            
            outputs = self.model(x)
            
            # Entropy loss (Tent)
            entropy_per_sample = softmax_entropy(outputs)
            entropy_loss = entropy_per_sample.mean()
            
            # Discriminator loss: push activations toward clean distribution
            disc_loss = torch.tensor(0.0, device=x.device)
            if self.discriminator is not None and self.activation_hook.activation is not None:
                # Score how "clean" the activations are (higher = cleaner)
                cleanness_score = self.discriminator(self.activation_hook.activation)
                # Loss: push toward cleanness score of 1
                disc_loss = -torch.log(cleanness_score + 1e-8).mean()
            
            # Combined loss
            loss = entropy_loss + self.disc_weight * disc_loss
            
            loss.backward()
            
            if self.clip_norm > 0.0:
                params = []
                for g in self.optimizer.param_groups:
                    params.extend(g.get("params", []))
                torch.nn.utils.clip_grad_norm_(params, self.clip_norm)
            
            self.optimizer.step()
        
        # Final forward
        with torch.no_grad():
            outputs = self.model(x)
        
        self.batch_idx += 1
        return outputs
    
    def reset(self):
        """Reset model to initial state."""
        if self.model_state is not None:
            self.model.load_state_dict(self.model_state)
        if hasattr(self, 'initial_optimizer_state'):
            self.optimizer.load_state_dict(self.initial_optimizer_state)
        self.batch_idx = 0
    
    def __del__(self):
        """Cleanup hook on deletion."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
