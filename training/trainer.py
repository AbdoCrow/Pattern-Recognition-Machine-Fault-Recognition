"""
===============================================================================
training/trainer.py — Training Loop with Optimizer, Scheduler, Early Stopping
===============================================================================

OWNER: Osama
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
from config import (
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    LR_SCHEDULER_T_MAX,
    CHECKPOINT_DIR,
    BEST_MODEL_PATH,
    DEVICE,
)

class Trainer:
    """
    Handles the complete training pipeline for the CNN.
    """
    def __init__(self, model, train_loader, val_loader, device=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or DEVICE

        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=LR_SCHEDULER_T_MAX,
        )

        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [], "lr": []
        }

        self.best_val_loss = float("inf")
        self.patience_counter = 0

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    def train(self, max_epochs=None):
        if max_epochs is None:
            max_epochs = MAX_EPOCHS

        print(f"Training on {self.device}")
        print(f"Max epochs: {max_epochs} | Patience: {EARLY_STOPPING_PATIENCE}")
        print("=" * 60)

        for epoch in range(1, max_epochs + 1):
            # --- 1. TRAINING PHASE ---
            self.model.train() # Turn ON Dropout and BatchNorm updates
            running_loss = 0.0
            correct = 0
            total = 0

            # Progress bar for training
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{max_epochs} [Train]")
            
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Backward pass & Optimize
                loss.backward()
                self.optimizer.step()

                # Metrics calculation
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Update progress bar
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.2f}%"})

            epoch_train_loss = running_loss / len(self.train_loader.dataset)
            epoch_train_acc = 100. * correct / total

            # --- 2. VALIDATION PHASE ---
            epoch_val_loss, epoch_val_acc = self._validate()

            # --- 3. SCHEDULER STEP ---
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()

            # --- 4. RECORD HISTORY ---
            self.history["train_loss"].append(epoch_train_loss)
            self.history["val_loss"].append(epoch_val_loss)
            self.history["train_acc"].append(epoch_train_acc)
            self.history["val_acc"].append(epoch_val_acc)
            self.history["lr"].append(current_lr)

            print(f"   --> Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}% | LR: {current_lr:.6f}")

            # --- 5. EARLY STOPPING & CHECKPOINTING ---
            if epoch_val_loss < self.best_val_loss:
                self.best_val_loss = epoch_val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, epoch_val_loss, epoch_val_acc)
                print(f"   [!] New best model saved! (Loss: {epoch_val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"\n[!] Early stopping triggered at epoch {epoch}. No improvement for {EARLY_STOPPING_PATIENCE} epochs.")
                    break

        return self.history

    def _validate(self):
        self.model.eval() # Turn OFF Dropout, freeze BatchNorm
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad(): # Disable gradient tracking to save memory
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / len(self.val_loader.dataset)
        val_acc = 100. * correct / total
        return val_loss, val_acc

    def _save_checkpoint(self, epoch, val_loss, val_acc):
        # Save only the weights to BEST_MODEL_PATH for infer.py to use later
        torch.save(self.model.state_dict(), BEST_MODEL_PATH)