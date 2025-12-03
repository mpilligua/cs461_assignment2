import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import KFold, StratifiedShuffleSplit
import os
import wandb


# Class names for better visualization (update if different)
CLASS_NAMES = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']


def create_stratified_split(dataset, test_size=0.15, random_state=42):
    """
    Create a stratified train/test split at the patient level.
    
    Args:
        dataset: ImageDataset with patient-level data
        test_size: Fraction of data to use for test set (default 0.15)
        random_state: Random seed for reproducibility
    
    Returns:
        train_indices: Indices for training set
        test_indices: Indices for test set
    """
    # Get labels for each patient (dataset returns patient-level data)
    labels = []
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        labels.append(label)
    labels = np.array(labels)
    
    # Create stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    indices = np.arange(len(dataset))
    
    train_indices, test_indices = next(sss.split(indices, labels))
    
    # Print split statistics
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]
    
    print(f"\n{'='*50}")
    print("STRATIFIED SPLIT STATISTICS")
    print(f"{'='*50}")
    print(f"Total patients: {len(dataset)}")
    print(f"Train patients: {len(train_indices)} ({100*len(train_indices)/len(dataset):.1f}%)")
    print(f"Test patients: {len(test_indices)} ({100*len(test_indices)/len(dataset):.1f}%)")
    
    # Class distribution
    unique_labels = np.unique(labels)
    print(f"\nClass distribution:")
    print(f"{'Class':<10} {'Train':<10} {'Test':<10} {'Train %':<10} {'Test %':<10}")
    print("-" * 50)
    for label in unique_labels:
        train_count = np.sum(train_labels == label)
        test_count = np.sum(test_labels == label)
        total_count = np.sum(labels == label)
        print(f"{label:<10} {train_count:<10} {test_count:<10} {100*train_count/total_count:.1f}%{'':<5} {100*test_count/total_count:.1f}%")
    print(f"{'='*50}\n")
    
    return train_indices, test_indices


def attention_collate_fn(batch):
    """
    Custom collate function for attention-based model.
    Does NOT average - passes all patch embeddings to the model.
    """
    embeddings_list = []
    labels_list = []
    
    for embeddings, label in batch:
        embeddings_list.append(embeddings)  # Keep all patches, don't average
        labels_list.append(label)
    
    labels = torch.tensor(labels_list, dtype=torch.long)
    # embeddings_list is a list of tensors with variable lengths
    return embeddings_list, labels


def filter_dataset_by_classes(dataset, target_classes, remap_labels=True):
    """
    Filter a dataset to only include samples from specific classes.
    
    Args:
        dataset: Dataset or Subset to filter
        target_classes: List of class indices to keep (e.g., [0, 6] for ADI and NORM)
        remap_labels: If True, remap labels to 0, 1, 2, ... for the filtered classes
    
    Returns:
        filtered_indices: Indices of samples belonging to target classes
        label_mapping: Dict mapping original labels to new labels (if remap_labels=True)
    """
    filtered_indices = []
    
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if isinstance(label, torch.Tensor):
            label = label.item()
        if label in target_classes:
            filtered_indices.append(idx)
    
    label_mapping = None
    if remap_labels:
        label_mapping = {orig: new for new, orig in enumerate(sorted(target_classes))}
    
    print(f"Filtered dataset: {len(filtered_indices)} samples from classes {target_classes}")
    return filtered_indices, label_mapping


class FilteredDataset(torch.utils.data.Dataset):
    """
    Wrapper dataset that filters samples and optionally remaps labels.
    """
    def __init__(self, dataset, indices, label_mapping=None):
        self.dataset = dataset
        self.indices = indices
        self.label_mapping = label_mapping
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        embeddings, label = self.dataset[self.indices[idx]]
        if self.label_mapping is not None:
            if isinstance(label, torch.Tensor):
                label = label.item()
            label = self.label_mapping[label]
        return embeddings, label


class GatedAttentionHead(nn.Module):
    """
    Single Gated Attention head for MIL.
    Computes attention weights over patches and aggregates them.
    """
    def __init__(self, embed_dim, hidden_dim=512):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.attention_U = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.attention_weights = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        """
        Args:
            x: [num_patches, embed_dim]
        Returns:
            aggregated: [embed_dim]
            attention: [num_patches] - attention weights for interpretability
        """
        # Gated attention mechanism
        A_V = self.attention_V(x)  # [num_patches, hidden_dim]
        A_U = self.attention_U(x)  # [num_patches, hidden_dim]
        A = self.attention_weights(A_V * A_U)  # [num_patches, 1] - element-wise product
        
        # Softmax to get attention weights
        A = torch.softmax(A, dim=0)  # [num_patches, 1]
        
        # Weighted sum of patches
        aggregated = torch.sum(A * x, dim=0)  # [embed_dim]
        
        return aggregated, A.squeeze(-1)  # squeeze only last dim to keep [num_patches]


class AveragePoolingHead(nn.Module):
    """
    Simple average pooling head that computes mean of patches and passes through MLP.
    Acts as a baseline aggregation method alongside attention heads.
    """
    def __init__(self, embed_dim, hidden_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: [num_patches, embed_dim]
        Returns:
            aggregated: [embed_dim]
            attention: [num_patches] - uniform attention weights (for consistency)
        """
        # Simple average pooling
        avg_pooled = torch.mean(x, dim=0)  # [embed_dim]
        
        # Pass through MLP
        aggregated = self.mlp(avg_pooled)  # [embed_dim]
        
        # Return uniform attention weights for interpretability (1/num_patches each)
        num_patches = x.shape[0]
        uniform_attention = torch.ones(num_patches, device=x.device) / num_patches
        
        return aggregated, uniform_attention


class MultiHeadGatedAttention(nn.Module):
    """
    Multi-Head Gated Attention mechanism for MIL.
    Creates multiple attention heads that learn different attention patterns,
    plus an average pooling head, then averages all outputs.
    """
    def __init__(self, embed_dim, hidden_dim=256, num_heads=16, include_avg_pool=True):
        super().__init__()
        self.num_heads = num_heads
        self.include_avg_pool = include_avg_pool
        
        # Create multiple independent attention heads
        self.attention_heads = nn.ModuleList([
            GatedAttentionHead(embed_dim, hidden_dim) for _ in range(num_heads)
        ])
        
        # Average pooling head with MLP
        if include_avg_pool:
            self.avg_pool_head = AveragePoolingHead(embed_dim, hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x: [num_patches, embed_dim]
        Returns:
            aggregated: [embed_dim] - averaged output from all heads (including avg pool)
            attention: [num_heads(+1), num_patches] - attention weights from each head
        """
        aggregated_list = []
        attention_list = []
        
        # Process through each attention head
        for head in self.attention_heads:
            agg, attn = head(x)
            aggregated_list.append(agg)
            attention_list.append(attn)
        
        # Process through average pooling head
        if self.include_avg_pool:
            avg_agg, avg_attn = self.avg_pool_head(x)
            aggregated_list.append(avg_agg)
            attention_list.append(avg_attn)
        
        # Stack and average the aggregated representations
        aggregated_stack = torch.stack(aggregated_list, dim=0)  # [num_heads(+1), embed_dim]
        aggregated = torch.mean(aggregated_stack, dim=0)  # [embed_dim]
        
        # Stack attention weights for interpretability
        attention = torch.stack(attention_list, dim=0)  # [num_heads(+1), num_patches]
        
        return aggregated, attention


class Submission(nn.Module):
    """
    Attention-based Multiple Instance Learning model for histopathology classification.
    Uses multi-head gated attention to learn diverse attention patterns.
    """
    def __init__(self, embed_dim, num_classes, hidden_dim=512, dropout=0.3, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head attention mechanism
        self.attention = MultiHeadGatedAttention(embed_dim, hidden_dim, num_heads)
        
        # Bigger classifier on aggregated representation
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: Either:
               - List of tensors [num_patches_i, embed_dim] for training (batch processing)
               - Single tensor [num_patches, embed_dim] for single patient
        Returns:
            logits: [batch_size, num_classes] or [num_classes]
        """
        if isinstance(x, list):
            # Batch mode: process each patient separately
            batch_logits = []
            for patient_patches in x:
                patient_patches = patient_patches.to(self.classifier[0].weight.device)
                aggregated, _ = self.attention(patient_patches)
                logits = self.classifier(aggregated)
                batch_logits.append(logits)
            output = torch.stack(batch_logits)  # [batch_size, num_classes]
        else:
            # Single patient mode
            aggregated, _ = self.attention(x)
            output = self.classifier(aggregated)
        
        # Apply softmax during evaluation
        if not self.training:
            output = torch.softmax(output, dim=-1)
        
        return output
   
    def train_with_cv(self, dataset, num_folds=1, num_epochs=50, lr=0.001, 
                      batch_size=32, device='cuda', save_best_fold=None, 
                      early_stopping_patience=5, collate_fn=attention_collate_fn, min_delta=0.0001,
                      weight_decay=1e-4, use_class_weights=True):
        """
        Train the attention-based MIL model using k-fold cross-validation.
        
        Args:
            dataset: Full dataset (torch.utils.data.Dataset)
            num_folds: Number of folds for cross-validation
            num_epochs: Number of training epochs per fold
            lr: Learning rate
            batch_size: Batch size for DataLoader
            device: Device to train on ('cuda' or 'cpu')
            save_best_fold: Path to save the best fold model. If None, model is not saved.
            early_stopping_patience: Number of epochs to wait before early stopping
            min_delta: Minimum improvement in validation F1 to reset patience counter
            weight_decay: L2 regularization strength
            use_class_weights: Whether to use class weights for imbalanced dataset
        
        Returns:
            dict: Cross-validation results with metrics for each fold
        """
        # Compute class weights from the full dataset
        all_labels = []
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            all_labels.append(label)
        all_labels = np.array(all_labels)
        
        if use_class_weights:
            # Compute inverse frequency weights
            class_counts = np.bincount(all_labels)
            total_samples = len(all_labels)
            # Use inverse frequency: weight = total / (num_classes * count_per_class)
            class_weights = total_samples / (len(class_counts) * class_counts)
            class_weights = torch.FloatTensor(class_weights).to(device)
            print(f"\nClass weights: {class_weights.cpu().numpy()}")
            print(f"Class counts: {class_counts}")
        else:
            class_weights = None
        
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        
        cv_results = {
            'fold_train_acc': [],
            'fold_val_acc': [],
            'fold_train_f1': [],
            'fold_val_f1': [],
            'fold_train_loss': [],
            'fold_val_loss': [],
            'fold_histories': []
        }
        
        best_val_f1 = 0
        best_fold = 0
        
        indices = np.arange(len(dataset))
        
        # Initialize wandb run for this CV session
        wandb.init(
            project="Foundations_A2",
            group="part2",
            name=f"multihead_attention_mil_weighted_loss_{num_folds}fold",
            config={
                "architecture": "Multi-Head Gated Attention MIL + Avg Pool",
                "training_mode": "standard_cv",
                "num_folds": num_folds,
                "num_epochs": num_epochs,
                "learning_rate": lr,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "hidden_dim": self.hidden_dim,
                "embed_dim": self.embed_dim,
                "num_classes": self.num_classes,
                "num_heads": self.num_heads,
                "use_class_weights": use_class_weights,
                "early_stopping_patience": early_stopping_patience
            }
        )
        
        for fold, (train_ids, val_ids) in enumerate(kfold.split(indices)):
            print(f"\n{'='*50}")
            print(f"FOLD {fold + 1}/{num_folds}")
            print(f"{'='*50}")
            
            # Create data subsets
            train_subset = Subset(dataset, train_ids)
            val_subset = Subset(dataset, val_ids)
            
            # Create data loaders
            train_loader = DataLoader(train_subset, batch_size=batch_size, 
                                     shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_subset, batch_size=batch_size, 
                                   shuffle=False, collate_fn=collate_fn)
            
            # Reset model for this fold
            self.__init__(self.embed_dim, self.num_classes, self.hidden_dim, num_heads=self.num_heads)
            self.to(device)
            
            # Use class weights for imbalanced dataset
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
            
            # Learning rate scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=3
            )
            
            fold_history = {
                'train_loss': [],
                'train_acc': [],
                'train_f1': [],
                'val_loss': [],
                'val_acc': [],
                'val_f1': []
            }
            
            best_fold_val_f1 = 0
            patience_counter = 0
            best_fold_state = None
            
            for epoch in range(num_epochs):
                # Training phase
                self.train()
                total_train_loss = 0
                train_preds = []
                train_labels = []
                
                for embeddings_list, labels in tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}'):
                    # Move each patient's patches to device
                    embeddings_list = [emb.to(device) for emb in embeddings_list]
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = self(embeddings_list)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_train_loss += loss.item()
                    preds = outputs.argmax(dim=1)
                    train_preds.extend(preds.cpu().numpy())
                    train_labels.extend(labels.cpu().numpy())
                
                avg_train_loss = total_train_loss / len(train_loader)
                train_accuracy = accuracy_score(train_labels, train_preds)
                train_f1 = f1_score(train_labels, train_preds, average='macro')
                
                # Validation phase
                self.eval()
                total_val_loss = 0
                val_preds = []
                val_labels = []
                
                with torch.no_grad():
                    for embeddings_list, labels in val_loader:
                        embeddings_list = [emb.to(device) for emb in embeddings_list]
                        labels = labels.to(device)
                        
                        outputs = self(embeddings_list)
                        loss = criterion(outputs, labels)
                        
                        total_val_loss += loss.item()
                        preds = outputs.argmax(dim=1)
                        val_preds.extend(preds.cpu().numpy())
                        val_labels.extend(labels.cpu().numpy())
                
                avg_val_loss = total_val_loss / len(val_loader)
                val_accuracy = accuracy_score(val_labels, val_preds)
                val_f1 = f1_score(val_labels, val_preds, average='macro')
                
                # Compute per-class metrics
                val_f1_per_class = f1_score(val_labels, val_preds, average=None)
                val_precision_per_class = precision_score(val_labels, val_preds, average=None, zero_division=0)
                val_recall_per_class = recall_score(val_labels, val_preds, average=None, zero_division=0)
                
                train_f1_per_class = f1_score(train_labels, train_preds, average=None)
                train_precision_per_class = precision_score(train_labels, train_preds, average=None, zero_division=0)
                train_recall_per_class = recall_score(train_labels, train_preds, average=None, zero_division=0)
                
                fold_history['train_loss'].append(avg_train_loss)
                fold_history['train_acc'].append(train_accuracy)
                fold_history['train_f1'].append(train_f1)
                fold_history['val_loss'].append(avg_val_loss)
                fold_history['val_acc'].append(val_accuracy)
                fold_history['val_f1'].append(val_f1)
                
                # Log metrics to wandb
                log_dict = {
                    f"fold_{fold+1}/train_loss": avg_train_loss,
                    f"fold_{fold+1}/train_acc": train_accuracy,
                    f"fold_{fold+1}/train_f1": train_f1,
                    f"fold_{fold+1}/val_loss": avg_val_loss,
                    f"fold_{fold+1}/val_acc": val_accuracy,
                    f"fold_{fold+1}/val_f1": val_f1,
                    "epoch": epoch + 1,
                    "fold": fold + 1
                }
                
                # Log per-class metrics
                num_classes = len(val_f1_per_class)
                for c in range(num_classes):
                    class_name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
                    log_dict[f"fold_{fold+1}/val_f1_{class_name}"] = val_f1_per_class[c]
                    log_dict[f"fold_{fold+1}/val_precision_{class_name}"] = val_precision_per_class[c]
                    log_dict[f"fold_{fold+1}/val_recall_{class_name}"] = val_recall_per_class[c]
                    log_dict[f"fold_{fold+1}/train_f1_{class_name}"] = train_f1_per_class[c]
                
                wandb.log(log_dict)
                
                # Update learning rate
                scheduler.step(val_f1)
                
                # Print per-class F1 scores
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Train F1: {train_f1:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
                print(f"  Per-class Val F1: ", end="")
                for c in range(num_classes):
                    class_name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"c{c}"
                    print(f"{class_name}:{val_f1_per_class[c]:.2f} ", end="")
                print()
                
                # Track best validation F1 for this fold and early stopping
                if val_f1 > best_fold_val_f1 + min_delta:
                    best_fold_val_f1 = val_f1
                    patience_counter = 0
                    # Save best state for this fold
                    best_fold_state = self.state_dict().copy()
                    print(f"  → New best validation F1: {best_fold_val_f1:.4f}")
                else:
                    patience_counter += 1
                    print(f"  → No improvement. Patience: {patience_counter}/{early_stopping_patience}")
                
                # Early stopping check
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    print(f"Best validation F1 for this fold: {best_fold_val_f1:.4f}")
                    # Restore best model state
                    if best_fold_state is not None:
                        self.load_state_dict(best_fold_state)
                    break
            
            # Store fold results
            cv_results['fold_train_acc'].append(fold_history['train_acc'][-1])
            cv_results['fold_val_acc'].append(fold_history['val_acc'][-1])
            cv_results['fold_train_f1'].append(fold_history['train_f1'][-1])
            cv_results['fold_val_f1'].append(fold_history['val_f1'][-1])
            cv_results['fold_train_loss'].append(fold_history['train_loss'][-1])
            cv_results['fold_val_loss'].append(fold_history['val_loss'][-1])
            cv_results['fold_histories'].append(fold_history)
            
            # Log confusion matrix for this fold
            cm = confusion_matrix(val_labels, val_preds)
            num_classes = cm.shape[0]
            class_labels = [CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}" for i in range(num_classes)]
            
            # Log confusion matrix as wandb table
            try:
                wandb.log({
                    f"fold_{fold+1}/confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=val_labels,
                        preds=val_preds,
                        class_names=class_labels[:num_classes]
                    )
                })
            except Exception as e:
                print(f"Could not log confusion matrix: {e}")
            
            # Print confusion matrix
            print(f"\nFold {fold + 1} Confusion Matrix:")
            print("Predicted →")
            print("Actual ↓")
            header = "     " + " ".join([f"{class_labels[i][:4]:>5}" for i in range(num_classes)])
            print(header)
            for i in range(num_classes):
                row = f"{class_labels[i][:4]:>4} " + " ".join([f"{cm[i,j]:>5}" for j in range(num_classes)])
                print(row)
            
            print(f"\nFold {fold + 1} completed - Best Val F1: {best_fold_val_f1:.4f}")
            
            # Log fold summary to wandb
            wandb.log({
                f"fold_{fold+1}/best_val_f1": best_fold_val_f1,
                f"fold_{fold+1}/final_train_acc": fold_history['train_acc'][-1],
                f"fold_{fold+1}/final_val_acc": fold_history['val_acc'][-1],
            })
            
            # Save best fold model
            if best_fold_val_f1 > best_val_f1:
                best_val_f1 = best_fold_val_f1
                best_fold = fold + 1
                if save_best_fold:
                    self.save_weights(save_best_fold)
                    print(f"Saved best fold model (Fold {best_fold}) with Val F1: {best_val_f1:.4f}")
        
        # Print summary statistics
        print(f"\n{'='*50}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'='*50}")
        print(f"Mean Train Accuracy: {np.mean(cv_results['fold_train_acc']):.4f} ± {np.std(cv_results['fold_train_acc']):.4f}")
        print(f"Mean Val Accuracy: {np.mean(cv_results['fold_val_acc']):.4f} ± {np.std(cv_results['fold_val_acc']):.4f}")
        print(f"Mean Train F1: {np.mean(cv_results['fold_train_f1']):.4f} ± {np.std(cv_results['fold_train_f1']):.4f}")
        print(f"Mean Val F1: {np.mean(cv_results['fold_val_f1']):.4f} ± {np.std(cv_results['fold_val_f1']):.4f}")
        print(f"Mean Train Loss: {np.mean(cv_results['fold_train_loss']):.4f} ± {np.std(cv_results['fold_train_loss']):.4f}")
        print(f"Mean Val Loss: {np.mean(cv_results['fold_val_loss']):.4f} ± {np.std(cv_results['fold_val_loss']):.4f}")
        print(f"Best Fold: {best_fold} with Val F1: {best_val_f1:.4f}")
        
        # Log final summary to wandb
        wandb.log({
            "cv_summary/mean_train_acc": np.mean(cv_results['fold_train_acc']),
            "cv_summary/std_train_acc": np.std(cv_results['fold_train_acc']),
            "cv_summary/mean_val_acc": np.mean(cv_results['fold_val_acc']),
            "cv_summary/std_val_acc": np.std(cv_results['fold_val_acc']),
            "cv_summary/mean_train_f1": np.mean(cv_results['fold_train_f1']),
            "cv_summary/std_train_f1": np.std(cv_results['fold_train_f1']),
            "cv_summary/mean_val_f1": np.mean(cv_results['fold_val_f1']),
            "cv_summary/std_val_f1": np.std(cv_results['fold_val_f1']),
            "cv_summary/best_fold": best_fold,
            "cv_summary/best_val_f1": best_val_f1
        })
        
        # Finish wandb run
        wandb.finish()
        
        return cv_results
    
    def save_weights(self, path):
        """
        Save model weights along with architecture info.
        
        Args:
            path: Path to save the model weights
        """
        torch.save({
            'state_dict': self.state_dict(),
            'embed_dim': self.embed_dim,
            'num_classes': self.num_classes,
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads
        }, path)
        print(f"Model weights saved to {path}")
    
    @classmethod
    def load_weights(cls, path, device='cuda'):
        """
        Load model weights from a saved checkpoint.
        
        Args:
            path: Path to the saved model weights
            device: Device to load the model on
        
        Returns:
            Submission: Model with loaded weights
        """
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            checkpoint['embed_dim'], 
            checkpoint['num_classes'],
            hidden_dim=checkpoint.get('hidden_dim', 512),
            num_heads=checkpoint.get('num_heads', 4)
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        print(f"Model weights loaded from {path}")
        return model
    
    def evaluate_on_test(self, test_dataset, batch_size=32, device='cuda', 
                         collate_fn=attention_collate_fn, log_wandb=False):
        """
        Evaluate the model on a held-out test set.
        
        Args:
            test_dataset: Test dataset (Subset or Dataset)
            batch_size: Batch size for DataLoader
            device: Device to evaluate on
            collate_fn: Collate function for DataLoader
            log_wandb: Whether to log results to wandb
        
        Returns:
            dict: Test metrics including accuracy, F1, and classification report
        """
        self.eval()
        self.to(device)
        
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                 shuffle=False, collate_fn=collate_fn)
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for embeddings_list, labels in tqdm(test_loader, desc="Evaluating on test set"):
                embeddings_list = [emb.to(device) for emb in embeddings_list]
                labels = labels.to(device)
                
                outputs = self(embeddings_list)  # Already softmaxed in eval mode
                preds = outputs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(outputs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        test_accuracy = accuracy_score(all_labels, all_preds)
        test_f1_macro = f1_score(all_labels, all_preds, average='macro')
        test_f1_weighted = f1_score(all_labels, all_preds, average='weighted')
        
        # Per-class F1
        test_f1_per_class = f1_score(all_labels, all_preds, average=None)
        
        # Classification report
        report = classification_report(all_labels, all_preds)
        
        results = {
            'accuracy': test_accuracy,
            'f1_macro': test_f1_macro,
            'f1_weighted': test_f1_weighted,
            'f1_per_class': test_f1_per_class,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'classification_report': report
        }
        
        print(f"\n{'='*50}")
        print("TEST SET EVALUATION")
        print(f"{'='*50}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test F1 (Macro): {test_f1_macro:.4f}")
        print(f"Test F1 (Weighted): {test_f1_weighted:.4f}")
        print(f"\nClassification Report:\n{report}")
        print(f"{'='*50}\n")
        
        if log_wandb:
            try:
                wandb.log({
                    "test/accuracy": test_accuracy,
                    "test/f1_macro": test_f1_macro,
                    "test/f1_weighted": test_f1_weighted,
                })
                for i, f1 in enumerate(test_f1_per_class):
                    wandb.log({f"test/f1_class_{i}": f1})
            except Exception as e:
                print(f"Failed to log to wandb: {e}")
        
        return results
    
    def train_curriculum_lwf(self, full_dataset, hard_classes=[0, 6], 
                              phase1_epochs=30, phase2_epochs=50,
                              lr=0.001, batch_size=16, device='cuda',
                              lwf_alpha=0.5, lwf_temperature=2.0,
                              early_stopping_patience=10, weight_decay=1e-4,
                              save_path="ckpts/curriculum_best.pt",
                              skip_phase1=False, phase1_ckpt_path="ckpts/phase1_best.pt"):
        """
        Curriculum learning with Learning Without Forgetting (LwF).
        
        Phase 1: Train on hard classes only (ADI=0, NORM=6) to learn to distinguish them
        Phase 2: Fine-tune on all classes using LwF to retain knowledge of hard classes
        
        Args:
            full_dataset: Full dataset with all classes
            hard_classes: List of hard class indices to focus on first (default: [0, 6] for ADI, NORM)
            phase1_epochs: Number of epochs for phase 1 (hard classes only)
            phase2_epochs: Number of epochs for phase 2 (all classes with LwF)
            lr: Learning rate
            batch_size: Batch size
            device: Device to train on
            lwf_alpha: Weight for LwF distillation loss (0 = no distillation, 1 = only distillation)
            lwf_temperature: Temperature for knowledge distillation
            early_stopping_patience: Early stopping patience
            weight_decay: L2 regularization
            save_path: Path to save the best model
            skip_phase1: If True, load phase 1 model from checkpoint instead of training
            phase1_ckpt_path: Path to phase 1 checkpoint (used when skip_phase1=True)
        """
        print("\n" + "="*60)
        print("CURRICULUM LEARNING WITH LWF")
        print("="*60)
        
        # Initialize wandb
        hard_class_names = [CLASS_NAMES[c] for c in hard_classes]
        wandb.init(
            project="Foundations_A2",
            group="part2",
            name=f"curriculum_lwf_hard_{'-'.join(hard_class_names)}_alpha{lwf_alpha}",
            config={
                "architecture": "Multi-Head Gated Attention MIL + Avg Pool",
                "training_mode": "curriculum_lwf",
                "hard_classes": hard_classes,
                "hard_class_names": hard_class_names,
                "phase1_epochs": phase1_epochs,
                "phase2_epochs": phase2_epochs,
                "learning_rate": lr,
                "batch_size": batch_size,
                "lwf_alpha": lwf_alpha,
                "lwf_temperature": lwf_temperature,
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "skip_phase1": skip_phase1,
            }
        )
        
        # Filter dataset for hard classes (needed for label_mapping in phase 2)
        hard_indices, label_mapping = filter_dataset_by_classes(
            full_dataset, hard_classes, remap_labels=True
        )
        hard_dataset = FilteredDataset(full_dataset, hard_indices, label_mapping)
        
        # Create phase 1 model (2-class model for hard classes)
        phase1_model = Submission(
            self.embed_dim, 
            num_classes=len(hard_classes),
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads
        ).to(device)
        
        if skip_phase1:
            # ============================================================
            # SKIP PHASE 1: Load from checkpoint
            # ============================================================
            print(f"\n{'='*50}")
            print(f"PHASE 1: SKIPPED - Loading from {phase1_ckpt_path}")
            print(f"{'='*50}")
            
            if not os.path.exists(phase1_ckpt_path):
                raise FileNotFoundError(f"Phase 1 checkpoint not found: {phase1_ckpt_path}")
            
            phase1_model.load_state_dict(torch.load(phase1_ckpt_path, map_location=device))
            print(f"Loaded phase 1 model from {phase1_ckpt_path}")
            
        else:
            # ============================================================
            # PHASE 1: Train on hard classes only
            # ============================================================
            print(f"\n{'='*50}")
            print(f"PHASE 1: Training on hard classes {hard_classes}")
            print(f"Classes: {[CLASS_NAMES[c] for c in hard_classes]}")
            print(f"{'='*50}")
            
            # Split hard dataset for training/validation
            hard_labels = [hard_dataset[i][1] for i in range(len(hard_dataset))]
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, val_idx = next(sss.split(range(len(hard_dataset)), hard_labels))
            
            train_subset = Subset(hard_dataset, train_idx)
            val_subset = Subset(hard_dataset, val_idx)
            
            train_loader = DataLoader(train_subset, batch_size=batch_size, 
                                      shuffle=True, collate_fn=attention_collate_fn)
            val_loader = DataLoader(val_subset, batch_size=batch_size,
                                    shuffle=False, collate_fn=attention_collate_fn)
            
            # Compute class weights for hard classes
            hard_label_counts = np.bincount([hard_dataset[i][1] for i in range(len(hard_dataset))])
            hard_weights = len(hard_dataset) / (len(hard_classes) * hard_label_counts)
            hard_weights = torch.FloatTensor(hard_weights).to(device)
            
            criterion = nn.CrossEntropyLoss(weight=hard_weights)
            optimizer = optim.Adam(phase1_model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
            
            best_val_f1 = 0
            patience_counter = 0
            
            for epoch in range(phase1_epochs):
                # Training
                phase1_model.train()
                total_loss = 0
                train_preds, train_labels = [], []
                
                for embeddings_list, labels in tqdm(train_loader, desc=f'Phase1 Epoch {epoch+1}/{phase1_epochs}'):
                    embeddings_list = [emb.to(device) for emb in embeddings_list]
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = phase1_model(embeddings_list)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                    train_labels.extend(labels.cpu().numpy())
                
                train_f1 = f1_score(train_labels, train_preds, average='macro')
                
                # Validation
                phase1_model.eval()
                val_preds, val_labels_list = [], []
                with torch.no_grad():
                    for embeddings_list, labels in val_loader:
                        embeddings_list = [emb.to(device) for emb in embeddings_list]
                        outputs = phase1_model(embeddings_list)
                        val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                        val_labels_list.extend(labels.numpy())
                
                val_f1 = f1_score(val_labels_list, val_preds, average='macro')
                val_acc = accuracy_score(val_labels_list, val_preds)
                
                scheduler.step(val_f1)
                
                print(f"Phase1 Epoch {epoch+1}: Train F1={train_f1:.4f}, Val F1={val_f1:.4f}, Val Acc={val_acc:.4f}")
                wandb.log({
                    "phase1/train_f1": train_f1,
                    "phase1/val_f1": val_f1,
                    "phase1/val_acc": val_acc,
                    "phase1/loss": total_loss / len(train_loader),
                    "phase1/epoch": epoch + 1
                })
                
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    # Save phase 1 model
                    torch.save(phase1_model.state_dict(), "ckpts/phase1_best.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Load best phase 1 model
            phase1_model.load_state_dict(torch.load("ckpts/phase1_best.pt", map_location=device))
            print(f"\nPhase 1 complete. Best Val F1: {best_val_f1:.4f}")
        
        # ============================================================
        # PHASE 2: Fine-tune on all classes with LwF
        # ============================================================
        print(f"\n{'='*50}")
        print(f"PHASE 2: Fine-tuning on all classes with LwF")
        print(f"LwF alpha={lwf_alpha}, temperature={lwf_temperature}")
        print(f"{'='*50}")
        
        # Transfer attention weights from phase 1 model to full model
        # The attention mechanism is class-agnostic, so we can transfer directly
        self.attention.load_state_dict(phase1_model.attention.state_dict())
        self.to(device)
        
        # Freeze attention initially for a few epochs (optional - helps stability)
        # for param in self.attention.parameters():
        #     param.requires_grad = False
        
        # Create data loaders for full dataset
        all_labels = [full_dataset[i][1] for i in range(len(full_dataset))]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(sss.split(range(len(full_dataset)), all_labels))
        
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size,
                                  shuffle=True, collate_fn=attention_collate_fn)
        val_loader = DataLoader(val_subset, batch_size=batch_size,
                                shuffle=False, collate_fn=attention_collate_fn)
        
        # Compute class weights for all classes
        all_label_counts = np.bincount(all_labels)
        all_weights = len(full_dataset) / (self.num_classes * all_label_counts)
        all_weights = torch.FloatTensor(all_weights).to(device)
        
        criterion_ce = nn.CrossEntropyLoss(weight=all_weights)
        optimizer = optim.Adam(self.parameters(), lr=lr * 0.5, weight_decay=weight_decay)  # Lower LR for fine-tuning
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        
        best_val_f1 = 0
        patience_counter = 0
        
        # Reverse label mapping for LwF (new label -> original label)
        reverse_mapping = {v: k for k, v in label_mapping.items()}
        
        for epoch in range(phase2_epochs):
            # Training with LwF
            self.train()
            phase1_model.eval()  # Teacher model
            
            total_loss = 0
            total_ce_loss = 0
            total_lwf_loss = 0
            train_preds, train_labels = [], []
            
            for embeddings_list, labels in tqdm(train_loader, desc=f'Phase2 Epoch {epoch+1}/{phase2_epochs}'):
                embeddings_list = [emb.to(device) for emb in embeddings_list]
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass through student (full model)
                outputs = self(embeddings_list)
                
                # Standard cross-entropy loss
                ce_loss = criterion_ce(outputs, labels)
                
                # LwF distillation loss for hard classes
                lwf_loss = torch.tensor(0.0, device=device)
                if lwf_alpha > 0:
                    with torch.no_grad():
                        # Get teacher predictions (phase 1 model)
                        teacher_outputs = phase1_model(embeddings_list)  # [batch, 2]
                    
                    # Extract student outputs for hard classes only
                    student_hard_outputs = outputs[:, hard_classes]  # [batch, 2]
                    
                    # Knowledge distillation loss
                    teacher_soft = F.softmax(teacher_outputs / lwf_temperature, dim=1)
                    student_soft = F.log_softmax(student_hard_outputs / lwf_temperature, dim=1)
                    lwf_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (lwf_temperature ** 2)
                
                # Combined loss
                loss = (1 - lwf_alpha) * ce_loss + lwf_alpha * lwf_loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_lwf_loss += lwf_loss.item() if isinstance(lwf_loss, torch.Tensor) else lwf_loss
                train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
            
            train_f1 = f1_score(train_labels, train_preds, average='macro')
            train_f1_per_class = f1_score(train_labels, train_preds, average=None)
            
            # Validation
            self.eval()
            val_preds, val_labels_list = [], []
            with torch.no_grad():
                for embeddings_list, labels in val_loader:
                    embeddings_list = [emb.to(device) for emb in embeddings_list]
                    outputs = self(embeddings_list)
                    val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                    val_labels_list.extend(labels.numpy())
            
            val_f1 = f1_score(val_labels_list, val_preds, average='macro')
            val_f1_per_class = f1_score(val_labels_list, val_preds, average=None)
            val_acc = accuracy_score(val_labels_list, val_preds)
            
            scheduler.step(val_f1)
            
            # Print per-class F1 for hard classes
            hard_class_f1s = [val_f1_per_class[c] for c in hard_classes]
            print(f"Phase2 Epoch {epoch+1}: Train F1={train_f1:.4f}, Val F1={val_f1:.4f}, "
                  f"Hard classes F1={hard_class_f1s}")
            
            # Log to wandb
            log_dict = {
                "phase2/train_f1": train_f1,
                "phase2/val_f1": val_f1,
                "phase2/val_acc": val_acc,
                "phase2/total_loss": total_loss / len(train_loader),
                "phase2/ce_loss": total_ce_loss / len(train_loader),
                "phase2/lwf_loss": total_lwf_loss / len(train_loader),
                "phase2/epoch": epoch + 1
            }
            for i, f1 in enumerate(val_f1_per_class):
                class_name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"
                log_dict[f"phase2/val_f1_{class_name}"] = f1
            wandb.log(log_dict)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                self.save_weights(save_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f"\nPhase 2 complete. Best Val F1: {best_val_f1:.4f}")
        print(f"Model saved to {save_path}")
        
        wandb.finish()
        
        return {
            'phase1_best_f1': best_val_f1,
            'phase2_best_f1': best_val_f1
        }


if __name__ == '__main__':
    import sys
    import argparse
    
    sys.path.append("/Users/maria/FaG/cs461_assignment2/part2/")  # To import from part2
    from utils import set_seed, load_cfg, load_full_dataset, build_model
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='curriculum', 
                        choices=['curriculum', 'standard'],
                        help='Training mode: curriculum (LwF) or standard (CV)')
    parser.add_argument('--lwf_alpha', type=float, default=0.3,
                        help='Weight for LwF distillation loss')
    parser.add_argument('--skip_phase1', action='store_true',
                        help='Skip phase 1 and load from checkpoint (ckpts/phase1_best.pt)')
    args = parser.parse_args()
    
    set_seed(42)
    cfg = load_cfg("configs/submission.yaml")
    dataset_cls, dataset_args = load_full_dataset(
        cfg.get("dataset"), additional_config={"split": "train"}
    )
    full_dataset = dataset_cls(**dataset_args)
    model_cls, model_args = build_model(cfg)
    model = model_cls(**model_args)
    
    # Create ckpts directory if it doesn't exist
    os.makedirs("ckpts", exist_ok=True)
    
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("cpu")  # Use CPU for MPS compatibility
    else:
        device = torch.device("cpu")
    
    print(f"Training on device: {device}")
    
    # Create stratified train/test split (15% held out for final testing)
    train_indices, test_indices = create_stratified_split(
        full_dataset, test_size=0.15, random_state=42
    )
    
    # Create subset datasets
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    if args.mode == 'curriculum':
        # ============================================================
        # CURRICULUM LEARNING WITH LWF
        # Phase 1: Train on hard classes (ADI=0, NORM=6)
        # Phase 2: Fine-tune on all classes with LwF
        # ============================================================
        print("\n" + "="*60)
        print("USING CURRICULUM LEARNING WITH LWF")
        print("Hard classes: ADI (0) and NORM (6)")
        if args.skip_phase1:
            print("SKIPPING PHASE 1 - Loading from checkpoint")
        print("="*60)
        
        train_results = model.train_curriculum_lwf(
            train_dataset,
            hard_classes=[0, 6],  # ADI and NORM
            phase1_epochs=30,
            phase2_epochs=70,
            lr=1e-3,
            batch_size=16,
            device=device,
            lwf_alpha=args.lwf_alpha,  # Balance between CE and LwF loss
            lwf_temperature=0.2,
            early_stopping_patience=10,
            weight_decay=1e-4,
            save_path="ckpts/curriculum_best.pt",
            skip_phase1=args.skip_phase1
        )
        
        best_model_path = "ckpts/curriculum_best.pt"
        
    else:
        # ============================================================
        # STANDARD TRAINING WITH CV
        # ============================================================
        train_results = model.train_with_cv(
            train_dataset,
            num_folds=5,
            num_epochs=100,
            lr=1e-3,
            batch_size=16,
            collate_fn=attention_collate_fn,
            device=device,
            save_best_fold="ckpts/best_submission.pt",
            early_stopping_patience=10,
            weight_decay=1e-4
        )
        
        best_model_path = "ckpts/best_submission.pt"
    
    # Load the best model and evaluate on held-out test set
    print("\n" + "="*50)
    print("FINAL EVALUATION ON HELD-OUT TEST SET")
    print("="*50)
    
    best_model = Submission.load_weights(best_model_path, device=device)
    test_results = best_model.evaluate_on_test(
        test_dataset,
        batch_size=16,
        device=device,
        collate_fn=attention_collate_fn,
        log_wandb=False
    )
    
    # Print per-class F1 for analysis
    print("\nPer-class F1 scores:")
    for i, f1 in enumerate(test_results['f1_per_class']):
        class_name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}"
        marker = " <-- HARD" if i in [0, 6] else ""
        print(f"  {class_name}: {f1:.4f}{marker}")
    
    # Save test indices for reproducibility
    np.save("ckpts/test_indices.npy", test_indices)
    print(f"Test indices saved to ckpts/test_indices.npy")
