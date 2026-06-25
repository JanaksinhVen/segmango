"""
Training and evaluation logic for SegFormer regression model.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, r2_score
from utils import MetricsTracker


class Trainer:
    """
    Trainer class for managing training, validation, and testing loops.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        criterion: Loss function
        device: Device to train on
        early_stop_patience: Patience for early stopping
    """
    
    def __init__(self, model, train_loader, val_loader, test_loader,
                 optimizer, scheduler, criterion, device, early_stop_patience=30):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.early_stop_patience = early_stop_patience
        
        self.metrics = MetricsTracker()
        self.best_val_loss = float('inf')
        self.early_stop_count = 0
    
    def train_epoch(self):
        """Run one training epoch."""
        self.model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for images, features, targets in tqdm(self.train_loader):
            images = images.to(self.device)
            features = features.to(self.device).float()
            targets = targets.to(self.device).unsqueeze(1)
            
            if len(features.shape) == 1:
                features = features.unsqueeze(1)
            
            # Forward pass
            preds = self.model(images, features)
            loss = self.criterion(preds, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_preds.append(preds.detach().cpu().numpy())
            train_targets.append(targets.detach().cpu().numpy())
        
        train_loss /= len(self.train_loader.dataset)
        
        # Concatenate predictions and targets
        import numpy as np
        train_preds = np.concatenate(train_preds)
        train_targets = np.concatenate(train_targets)
        
        self.metrics.update_train(train_loss, train_preds, train_targets)
        return train_loss
    
    def val_epoch(self):
        """Run one validation epoch."""
        self.model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for images, features, targets in tqdm(self.val_loader):
                images = images.to(self.device)
                features = features.to(self.device).float()
                targets = targets.to(self.device).unsqueeze(1)
                
                if len(features.shape) == 1:
                    features = features.unsqueeze(1)
                
                preds = self.model(images, features)
                loss = self.criterion(preds, targets)
                
                val_loss += loss.item() * images.size(0)
                val_preds.append(preds.cpu().numpy())
                val_targets.append(targets.cpu().numpy())
        
        val_loss /= len(self.val_loader.dataset)
        
        # Concatenate predictions and targets
        import numpy as np
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        
        self.metrics.update_val(val_loss, val_preds, val_targets)
        return val_loss
    
    def test_epoch(self):
        """Run test evaluation."""
        self.model.eval()
        test_loss = 0.0
        test_preds = []
        test_targets = []
        
        with torch.no_grad():
            for images, features, targets in tqdm(self.test_loader):
                images = images.to(self.device)
                features = features.to(self.device).float()
                targets = targets.to(self.device).unsqueeze(1)
                
                if len(features.shape) == 1:
                    features = features.unsqueeze(1)
                
                preds = self.model(images, features)
                loss = self.criterion(preds, targets)
                
                test_loss += loss.item() * images.size(0)
                test_preds.append(preds.cpu().numpy())
                test_targets.append(targets.cpu().numpy())
        
        test_loss /= len(self.test_loader.dataset)
        
        # Concatenate predictions and targets
        import numpy as np
        test_preds = np.concatenate(test_preds)
        test_targets = np.concatenate(test_targets)
        
        return test_loss, test_preds, test_targets
    
    def train(self, num_epochs, unfreeze_epoch, model_save_path):
        """
        Complete training loop with early stopping.
        
        Args:
            num_epochs: Number of training epochs
            unfreeze_epoch: Epoch at which to unfreeze encoder
            model_save_path: Path to save best model
        """
        for epoch in range(num_epochs):
            # Unfreeze encoder if needed
            if epoch == unfreeze_epoch:
                print("[INFO] Unfreezing encoder for fine-tuning")
                if hasattr(self.model, 'module'):
                    self.model.module.unfreeze_encoder()
                else:
                    self.model.unfreeze_encoder()
            
            # Train and validate
            train_loss = self.train_epoch()
            val_loss = self.val_epoch()
            
            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # Early stopping logic
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), model_save_path)
                print(f"[SAVE] Best model saved with val loss: {val_loss:.4f}")
                self.early_stop_count = 0
            else:
                self.early_stop_count += 1
                print(f"[INFO] Early stop counter: {self.early_stop_count}")
            
            # Print epoch summary
            summary = self.metrics.get_epoch_summary(epoch + 1)
            print(summary)
            print(f"[LR] Current learning rate: {self.optimizer.param_groups[0]['lr']}")
            
            if self.early_stop_count >= self.early_stop_patience:
                print(f"[STOP] Early stopping triggered after {epoch + 1} epochs")
                break
        
        return self.metrics
