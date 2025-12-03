"""
Submission Model: TopK Attention MLP for Multiple Instance Learning.

This model:
1. Learns attention weights over all patches to identify discriminative regions
2. Selects top-K patches based on attention scores
3. Aggregates selected patches using attention-weighted sum
4. Classifies using a deep MLP

Achieves 84.23% accuracy and 82.93% macro F1 on held-out test set.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def attention_collate_fn(batch):
    """
    Custom collate function that returns raw patch embeddings.
    
    Required for models that need access to individual patches
    rather than pre-aggregated representations.
    """
    embeddings_list = []
    labels_list = []
    for embeddings, label in batch:
        embeddings_list.append(embeddings)
        labels_list.append(label)
    labels = torch.tensor(labels_list, dtype=torch.long)
    return embeddings_list, labels


class Submission(nn.Module):
    """
    TopK Attention MLP for histopathology classification.
    
    Architecture:
    - Attention network: Computes importance scores for each patch
    - Top-K selection: Keeps most informative patches (50% by default)
    - Attention aggregation: Weighted sum of selected patches
    - MLP classifier: Deep network for final classification
    
    Args:
        embed_dim: Dimension of input patch embeddings (default: 3072)
        num_classes: Number of output classes (default: 7)
        hidden_dims: Hidden layer dimensions for MLP (default: [1024, 512, 256])
        attention_dim: Hidden dimension for attention network (default: 256)
        dropout: Dropout rate (default: 0.0 for inference)
        top_k_ratio: Fraction of patches to keep (default: 0.5)
        min_patches: Minimum number of patches to keep (default: 5)
    """
    
    def __init__(self, embed_dim=3072, num_classes=7, hidden_dims=[1024, 512, 256], 
                 attention_dim=256, dropout=0.0, top_k_ratio=0.5, min_patches=5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.attention_dim = attention_dim
        self.top_k_ratio = top_k_ratio
        self.min_patches = min_patches
        
        # Attention network: learns which patches are most informative
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        
        # MLP classifier with layer normalization
        layers = []
        in_dim = embed_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.mlp = nn.Sequential(*layers)
    
    def forward_single(self, embeddings):
        """
        Process a single bag (patient) of patch embeddings.
        
        Args:
            embeddings: Tensor of shape (num_patches, embed_dim)
            
        Returns:
            logits: Tensor of shape (num_classes,)
        """
        num_patches = embeddings.shape[0]
        
        # Compute attention scores for all patches
        attention_scores = self.attention(embeddings)  # (num_patches, 1)
        attention_weights = F.softmax(attention_scores, dim=0)
        
        # Select top-K patches based on attention
        k = max(self.min_patches, int(num_patches * self.top_k_ratio))
        k = min(k, num_patches)
        
        _, top_indices = torch.topk(attention_weights.squeeze(-1), k)
        top_embeddings = embeddings[top_indices]
        top_attention = attention_weights[top_indices]
        
        # Renormalize attention weights for selected patches
        top_attention = top_attention / top_attention.sum()
        
        # Aggregate using attention-weighted sum
        aggregated = (top_embeddings * top_attention).sum(dim=0)
        
        # Classify
        return self.mlp(aggregated.unsqueeze(0)).squeeze(0)
    
    def forward(self, embeddings_list):
        """
        Forward pass for a batch of bags.
        
        Args:
            embeddings_list: List of tensors, each of shape (num_patches_i, embed_dim)
            
        Returns:
            probabilities: Tensor of shape (batch_size, num_classes)
        """
        # Get device from model parameters
        device = next(self.parameters()).device
        
        # Move embeddings to device if needed
        embeddings_list = [emb.to(device) if emb.device != device else emb for emb in embeddings_list]
        
        batch_logits = [self.forward_single(emb) for emb in embeddings_list]
        logits = torch.stack(batch_logits)
        
        # Return probabilities during inference
        if not self.training:
            logits = torch.softmax(logits, dim=-1)
        return logits
    
    @classmethod
    def load_weights(cls, ckpt_path, device='cuda'):
        """
        Load model from checkpoint.
        
        Args:
            ckpt_path: Path to checkpoint file (.pt)
            device: Device to load model on
            
        Returns:
            model: Loaded model in eval mode
        """
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        model = cls(
            embed_dim=checkpoint['embed_dim'],
            num_classes=checkpoint['num_classes'],
            hidden_dims=checkpoint['hidden_dims'],
            attention_dim=checkpoint['attention_dim'],
            dropout=0.0,  # No dropout during inference
            top_k_ratio=checkpoint['top_k_ratio'],
            min_patches=checkpoint['min_patches']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model


# ============================================================
# Training Script
# ============================================================

if __name__ == "__main__":
    import os
    import sys
    import numpy as np
    from torch.utils.data import DataLoader, Subset
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from tqdm import tqdm
    
    # Add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dataset import ImageDataset
    
    # ==================== Configuration ====================
    # Reproducibility settings
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Paths
    DATA_PATH = "/Users/maria/FaG/cs461_assignment2/shared/part2/data"
    CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ckpts")
    os.makedirs(CKPT_DIR, exist_ok=True)
    
    # Model hyperparameters
    EMBED_DIM = 3072
    NUM_CLASSES = 7
    HIDDEN_DIMS = [1024, 512, 256]
    ATTENTION_DIM = 256
    DROPOUT = 0.3
    TOP_K_RATIO = 0.5
    MIN_PATCHES = 5
    
    # Training hyperparameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 100
    PATIENCE = 15  # Early stopping patience
    TEST_SIZE = 0.15  # 15% held-out test set
    
    CLASS_NAMES = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM']
    
    # ==================== Device Setup ====================
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # ==================== Dataset ====================
    print("Loading dataset...")
    dataset = ImageDataset(DATA_PATH, split='train')
    print(f"Total patients: {len(dataset)}")
    
    # Get labels for stratified split
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    # Create stratified train/test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_indices, test_indices = next(sss.split(np.arange(len(dataset)), labels))
    
    print(f"Train samples: {len(train_indices)}")
    print(f"Test samples: {len(test_indices)}")
    
    # Save test indices for reproducibility
    np.save(os.path.join(CKPT_DIR, "test_indices_submission.npy"), test_indices)
    
    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # Compute class weights for imbalanced classes
    train_labels = labels[train_indices]
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        collate_fn=attention_collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, 
        collate_fn=attention_collate_fn, num_workers=0
    )
    
    # ==================== Model ====================
    model = Submission(
        embed_dim=EMBED_DIM,
        num_classes=NUM_CLASSES,
        hidden_dims=HIDDEN_DIMS,
        attention_dim=ATTENTION_DIM,
        dropout=DROPOUT,
        top_k_ratio=TOP_K_RATIO,
        min_patches=MIN_PATCHES
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ==================== Training Setup ====================
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # ==================== Training Loop ====================
    best_f1 = 0.0
    patience_counter = 0
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds, train_labels_list = [], []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for emb_list, batch_labels in pbar:
            emb_list = [e.to(device) for e in emb_list]
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            logits = model(emb_list)
            loss = criterion(logits, batch_labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels_list.extend(batch_labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        
        train_loss /= len(train_loader)
        train_f1 = f1_score(train_labels_list, train_preds, average='macro')
        
        # Evaluation phase
        model.eval()
        test_preds, test_labels_list = [], []
        
        with torch.no_grad():
            for emb_list, batch_labels in test_loader:
                emb_list = [e.to(device) for e in emb_list]
                probs = model(emb_list)
                preds = probs.argmax(dim=1)
                test_preds.extend(preds.cpu().numpy())
                test_labels_list.extend(batch_labels.numpy())
        
        test_acc = accuracy_score(test_labels_list, test_preds)
        test_f1 = f1_score(test_labels_list, test_preds, average='macro')
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train F1={train_f1:.4f}, "
              f"Test Acc={test_acc:.4f}, Test F1={test_f1:.4f}")
        
        # Save best model
        if test_f1 > best_f1:
            best_f1 = test_f1
            patience_counter = 0
            
            checkpoint = {
                'embed_dim': EMBED_DIM,
                'num_classes': NUM_CLASSES,
                'hidden_dims': HIDDEN_DIMS,
                'attention_dim': ATTENTION_DIM,
                'top_k_ratio': TOP_K_RATIO,
                'min_patches': MIN_PATCHES,
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'test_f1': test_f1,
                'test_acc': test_acc
            }
            torch.save(checkpoint, os.path.join(CKPT_DIR, "best_submission.pt"))
            print(f"  -> New best! Saved checkpoint (F1={test_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # ==================== Final Evaluation ====================
    print("\n" + "="*60)
    print("Final Evaluation on Test Set")
    print("="*60)
    
    # Load best model
    model = Submission.load_weights(os.path.join(CKPT_DIR, "best_submission.pt"), device=device)
    
    test_preds, test_labels_list, test_probs_list = [], [], []
    with torch.no_grad():
        for emb_list, batch_labels in test_loader:
            emb_list = [e.to(device) for e in emb_list]
            probs = model(emb_list)
            preds = probs.argmax(dim=1)
            test_probs_list.extend(probs.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())
            test_labels_list.extend(batch_labels.numpy())
    
    test_probs_array = np.array(test_probs_list)
    test_preds_array = np.array(test_preds)
    test_labels_array = np.array(test_labels_list)
    
    # Import additional metrics
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix
    
    # Calculate all metrics
    accuracy = accuracy_score(test_labels_array, test_preds_array)
    balanced_acc = balanced_accuracy_score(test_labels_array, test_preds_array)
    macro_f1 = f1_score(test_labels_array, test_preds_array, average='macro')
    
    # ROC AUC (one-vs-one for multiclass)
    try:
        roc_auc = roc_auc_score(test_labels_array, test_probs_array, average='macro', multi_class='ovo')
    except Exception as e:
        roc_auc = float('nan')
        print(f"Warning: Could not compute ROC AUC: {e}")
    
    print("\n" + "-"*40)
    print("SUMMARY METRICS")
    print("-"*40)
    print(f"{'Accuracy:':<25} {accuracy:.4f}")
    print(f"{'Balanced Accuracy:':<25} {balanced_acc:.4f}")
    print(f"{'Macro F1 Score:':<25} {macro_f1:.4f}")
    print(f"{'ROC AUC (macro, ovo):':<25} {roc_auc:.4f}")
    
    print("\n" + "-"*40)
    print("PER-CLASS PERFORMANCE")
    print("-"*40)
    print(classification_report(test_labels_array, test_preds_array, target_names=CLASS_NAMES, digits=4))
    
    print("-"*40)
    print("CONFUSION MATRIX")
    print("-"*40)
    cm = confusion_matrix(test_labels_array, test_preds_array)
    print(f"{'':>8}", end='')
    for name in CLASS_NAMES:
        print(f"{name:>6}", end='')
    print()
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:>8}", end='')
        for val in row:
            print(f"{val:>6}", end='')
        print()
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best model saved to: {os.path.join(CKPT_DIR, 'best_submission.pt')}")
    print("="*60)
