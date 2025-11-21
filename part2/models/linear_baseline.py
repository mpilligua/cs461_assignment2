import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold


def linear_collate_fn(batch):
    """
    Custom collate function to handle variable number of crops per patient.
    """
    embeddings_list = []
    labels_list = []
    
    for embeddings, label in batch:
        embeddings_list.append(embeddings.mean(dim=0)) # Average over crops
        labels_list.append(label)
    
    labels = torch.tensor(labels_list, dtype=torch.long)
    embeddings = torch.stack(embeddings_list)
    return embeddings, labels.long()


class LinearProbe(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_classes)
        self.embed_dim = embed_dim
        self.num_classes = num_classes
    
    def forward(self, x):
        x = self.linear(x)
        if self.eval:
            x = torch.softmax(x, dim=1)
        return x
   
    def train_with_cv(self, dataset, num_folds=5, num_epochs=50, lr=0.001, 
                      batch_size=32, device='cuda', save_best_fold=None, 
                      early_stopping_patience=5, collate_fn=linear_collate_fn, min_delta=0.0001):
        """
        Train the linear probe model using k-fold cross-validation.
        
        Args:
            dataset: Full dataset (torch.utils.data.Dataset)
            num_folds: Number of folds for cross-validation
            num_epochs: Number of training epochs per fold
            lr: Learning rate
            batch_size: Batch size for DataLoader
            device: Device to train on ('cuda' or 'cpu')
            save_best_fold: Path to save the best fold model. If None, model is not saved.
            early_stopping_patience: Number of epochs to wait before early stopping
            min_delta: Minimum improvement in validation accuracy to reset patience counter
        
        Returns:
            dict: Cross-validation results with metrics for each fold
        """
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        
        cv_results = {
            'fold_train_acc': [],
            'fold_val_acc': [],
            'fold_train_loss': [],
            'fold_val_loss': [],
            'fold_histories': []
        }
        
        best_val_acc = 0
        best_fold = 0
        
        indices = np.arange(len(dataset))
        
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
            self.__init__(self.embed_dim, self.num_classes)
            self.to(device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.parameters(), lr=lr)
            
            fold_history = {
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': []
            }
            
            best_fold_val_acc = 0
            patience_counter = 0
            best_fold_state = None
            
            for epoch in range(num_epochs):
                # Training phase
                self.train()
                total_train_loss = 0
                train_preds = []
                train_labels = []
                
                for embeddings, labels in tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}'):
                    embeddings = embeddings.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = self(embeddings)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_train_loss += loss.item()
                    preds = outputs.argmax(dim=1)
                    train_preds.extend(preds.cpu().numpy())
                    train_labels.extend(labels.cpu().numpy())
                
                avg_train_loss = total_train_loss / len(train_loader)
                train_accuracy = accuracy_score(train_labels, train_preds)
                
                # Validation phase
                self.eval()
                total_val_loss = 0
                val_preds = []
                val_labels = []
                
                with torch.no_grad():
                    for embeddings, labels in val_loader:
                        embeddings = embeddings.to(device)
                        labels = labels.to(device)
                        
                        outputs = self(embeddings)
                        loss = criterion(outputs, labels)
                        
                        total_val_loss += loss.item()
                        preds = outputs.argmax(dim=1)
                        val_preds.extend(preds.cpu().numpy())
                        val_labels.extend(labels.cpu().numpy())
                
                avg_val_loss = total_val_loss / len(val_loader)
                val_accuracy = accuracy_score(val_labels, val_preds)
                
                fold_history['train_loss'].append(avg_train_loss)
                fold_history['train_acc'].append(train_accuracy)
                fold_history['val_loss'].append(avg_val_loss)
                fold_history['val_acc'].append(val_accuracy)
                
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
                
                # Track best validation accuracy for this fold and early stopping
                if val_accuracy > best_fold_val_acc + min_delta:
                    best_fold_val_acc = val_accuracy
                    patience_counter = 0
                    # Save best state for this fold
                    best_fold_state = self.state_dict().copy()
                    print(f"  → New best validation accuracy: {best_fold_val_acc:.4f}")
                else:
                    patience_counter += 1
                    print(f"  → No improvement. Patience: {patience_counter}/{early_stopping_patience}")
                
                # Early stopping check
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    print(f"Best validation accuracy for this fold: {best_fold_val_acc:.4f}")
                    # Restore best model state
                    if best_fold_state is not None:
                        self.load_state_dict(best_fold_state)
                    break
            
            # Store fold results
            cv_results['fold_train_acc'].append(fold_history['train_acc'][-1])
            cv_results['fold_val_acc'].append(fold_history['val_acc'][-1])
            cv_results['fold_train_loss'].append(fold_history['train_loss'][-1])
            cv_results['fold_val_loss'].append(fold_history['val_loss'][-1])
            cv_results['fold_histories'].append(fold_history)
            
            print(f"\nFold {fold + 1} completed - Best Val Acc: {best_fold_val_acc:.4f}")
            
            # Save best fold model
            if best_fold_val_acc > best_val_acc:
                best_val_acc = best_fold_val_acc
                best_fold = fold + 1
                if save_best_fold:
                    self.save_weights(save_best_fold)
                    print(f"Saved best fold model (Fold {best_fold}) with Val Acc: {best_val_acc:.4f}")
        
        # Print summary statistics
        print(f"\n{'='*50}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'='*50}")
        print(f"Mean Train Accuracy: {np.mean(cv_results['fold_train_acc']):.4f} ± {np.std(cv_results['fold_train_acc']):.4f}")
        print(f"Mean Val Accuracy: {np.mean(cv_results['fold_val_acc']):.4f} ± {np.std(cv_results['fold_val_acc']):.4f}")
        print(f"Mean Train Loss: {np.mean(cv_results['fold_train_loss']):.4f} ± {np.std(cv_results['fold_train_loss']):.4f}")
        print(f"Mean Val Loss: {np.mean(cv_results['fold_val_loss']):.4f} ± {np.std(cv_results['fold_val_loss']):.4f}")
        print(f"Best Fold: {best_fold} with Val Acc: {best_val_acc:.4f}")
        
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
            'num_classes': self.num_classes
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
            LinearProbe: Model with loaded weights
        """
        checkpoint = torch.load(path, map_location=device)
        model = cls(checkpoint['embed_dim'], checkpoint['num_classes'])
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        print(f"Model weights loaded from {path}")
        return model
    


if __name__ == '__main__':
    from utils import set_seed, load_cfg, load_full_dataset, build_model
    set_seed(42)
    cfg = load_cfg("configs/linear_baseline.yaml")
    dataset_cls, dataset_args = load_full_dataset(
        cfg.get("dataset"), additional_config={"split": "train"}
    )
    dataset = dataset_cls(**dataset_args)
    model_cls, model_args = build_model(cfg)
    model = model_cls(**model_args)

    train_results = model.train_with_cv(
        dataset,
        num_folds=5,
        num_epochs=50,
        lr=1e-2,
        collate_fn=linear_collate_fn,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        save_best_fold="ckpts/best_linear_baseline.pt"
    )