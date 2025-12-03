"""
Train a discriminator to recognize clean vs corrupted activations.

This script trains an MLP discriminator that scores how "clean" the first conv
activations are. Used during TTA to guide the model toward clean-like activations.

Usage:
    python train_discriminator.py

Output:
    ckpts/activation_discriminator.pt - Discriminator checkpoint
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from pathlib import Path

from utils import set_seed, load_cfg, build_model
from dataset import TestTimeAdaptationDataset

# ============================================================================
# Configuration
# ============================================================================

SEED = 42
BATCH_SIZE = 256
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 10
HIDDEN_DIMS = [512, 256, 128]

# Corruptions to use for "corrupted" class (test set corruptions)
CORRUPTED_CORRUPTIONS = ['shot_noise', 'fog', 'pixelate', 'frost', 'gaussian_blur', 'contrast']

# Use a mild corruption as proxy for "clean" (or use exploratory set)
# Since we don't have access to clean CIFAR-10, we use exploratory corruptions
# that are NOT in the test set as "clean-like"
CLEAN_CORRUPTIONS = ['brightness', 'defocus_blur', 'elastic_transform']

# ============================================================================
# Discriminator Architecture
# ============================================================================

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


# ============================================================================
# Main Training Script
# ============================================================================

if __name__ == "__main__":
    set_seed(SEED)
    
    # Create output directory
    os.makedirs("ckpts", exist_ok=True)
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load pretrained model
    print("\n" + "="*60)
    print("Loading Pretrained Model")
    print("="*60)
    
    cfg = load_cfg('configs/base_config.yaml')
    model = build_model(cfg)
    model.to(device)
    model.eval()
    
    # Find first conv layer
    first_conv = None
    first_conv_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            first_conv = module
            first_conv_name = name
            break
    
    if first_conv is None:
        raise ValueError("Could not find first conv layer!")
    
    print(f"First conv layer: {first_conv_name}")
    print(f"Output channels: {first_conv.out_channels}")
    
    # Hook to capture activations
    activations_list = []
    
    def activation_hook(module, input, output):
        batch_size = output.size(0)
        flat = output.view(batch_size, -1)
        activations_list.append(flat.detach().cpu())
    
    # ========================================================================
    # Extract Activations
    # ========================================================================
    
    print("\n" + "="*60)
    print("Extracting Activations from Exploratory Set")
    print("="*60)
    
    dataset_path = cfg.dataset.args.dataset_path
    
    # Load exploratory set for "clean-like" samples
    exploratory_path = Path(dataset_path) / "exploratory"
    if not exploratory_path.exists():
        exploratory_path = Path("../shared/CS461/cs461_assignment2_data/part1/exploratory")
    
    clean_activations = []
    corrupted_activations = []
    
    # Extract "clean-like" activations from exploratory corruptions
    print("\nExtracting 'clean-like' activations from exploratory corruptions...")
    for corruption in CLEAN_CORRUPTIONS:
        corr_file = exploratory_path / f"{corruption}.npy"
        if corr_file.exists():
            data = np.load(corr_file)
            print(f"  {corruption}: {len(data)} samples")
            
            # Register hook
            handle = first_conv.register_forward_hook(activation_hook)
            activations_list = []
            
            # Process in batches
            transform_mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
            transform_std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
            
            with torch.no_grad():
                for i in range(0, len(data), BATCH_SIZE):
                    batch = data[i:i+BATCH_SIZE]
                    # Convert to tensor and normalize
                    batch = torch.from_numpy(batch).float().permute(0, 3, 1, 2) / 255.0
                    batch = (batch - transform_mean) / transform_std
                    batch = batch.to(device)
                    _ = model(batch)
            
            handle.remove()
            clean_activations.append(torch.cat(activations_list, dim=0))
    
    if clean_activations:
        clean_activations = torch.cat(clean_activations, dim=0)
        print(f"\n✓ Total 'clean-like' activations: {len(clean_activations)}")
    else:
        raise ValueError("No clean activations extracted!")
    
    # Extract corrupted activations from test corruptions
    print("\nExtracting corrupted activations from test corruptions...")
    
    test_dataset = TestTimeAdaptationDataset(
        dataset_path=dataset_path,
        kind='public_test_bench'
    )
    
    for corruption in CORRUPTED_CORRUPTIONS:
        print(f"  Processing {corruption}...")
        corr_ds = test_dataset.filter_by_corruption(corruption)
        loader = DataLoader(corr_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        handle = first_conv.register_forward_hook(activation_hook)
        activations_list = []
        
        with torch.no_grad():
            for images, _, _ in loader:
                images = images.to(device)
                _ = model(images)
        
        handle.remove()
        corrupted_activations.append(torch.cat(activations_list, dim=0))
    
    corrupted_activations = torch.cat(corrupted_activations, dim=0)
    print(f"\n✓ Total corrupted activations: {len(corrupted_activations)}")
    
    # Balance dataset
    num_samples = min(len(clean_activations), len(corrupted_activations))
    
    # Shuffle and subsample
    clean_indices = torch.randperm(len(clean_activations))[:num_samples]
    corrupted_indices = torch.randperm(len(corrupted_activations))[:num_samples]
    
    clean_activations = clean_activations[clean_indices]
    corrupted_activations = corrupted_activations[corrupted_indices]
    
    print(f"\n✓ Using {num_samples} samples per class")
    
    # ========================================================================
    # Prepare Training Data
    # ========================================================================
    
    print("\n" + "="*60)
    print("Preparing Training Data")
    print("="*60)
    
    # Labels: 1 = clean, 0 = corrupted
    clean_labels = torch.ones(len(clean_activations), 1)
    corrupted_labels = torch.zeros(len(corrupted_activations), 1)
    
    all_activations = torch.cat([clean_activations, corrupted_activations], dim=0)
    all_labels = torch.cat([clean_labels, corrupted_labels], dim=0)
    
    # Shuffle
    indices = torch.randperm(len(all_activations))
    all_activations = all_activations[indices]
    all_labels = all_labels[indices]
    
    # Train/val split
    split = int(0.8 * len(all_activations))
    train_acts, val_acts = all_activations[:split], all_activations[split:]
    train_labels, val_labels = all_labels[:split], all_labels[split:]
    
    train_dataset = TensorDataset(train_acts, train_labels)
    val_dataset = TensorDataset(val_acts, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # ========================================================================
    # Train Discriminator
    # ========================================================================
    
    print("\n" + "="*60)
    print("Training Discriminator")
    print("="*60)
    
    input_dim = clean_activations.shape[1]
    discriminator = ActivationDiscriminator(input_dim, hidden_dims=HIDDEN_DIMS)
    discriminator.to(device)
    
    print(f"\nDiscriminator architecture:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dims: {HIDDEN_DIMS}")
    print(f"  Parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    
    print("\nTraining...")
    for epoch in range(NUM_EPOCHS):
        # Training
        discriminator.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for acts, labels in train_loader:
            acts, labels = acts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = discriminator(acts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = (outputs > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation
        discriminator.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for acts, labels in val_loader:
                acts, labels = acts.to(device), labels.to(device)
                outputs = discriminator(acts)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                preds = (outputs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            
            torch.save({
                'model_state_dict': discriminator.state_dict(),
                'input_dim': input_dim,
                'hidden_dims': HIDDEN_DIMS,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'epoch': epoch,
            }, 'ckpts/activation_discriminator.pt')
            print(f"  -> Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: ckpts/activation_discriminator.pt")
    print("\nTo use in TTA:")
    print("  The discriminator will be loaded automatically by submission.py")
    print("  when use_discriminator=true in configs/submission.yaml")
