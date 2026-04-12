"""
===============================================================================
training/trainer.py — Training Loop with Optimizer, Scheduler, Early Stopping
===============================================================================
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
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

        # Move model to device (GPU if available)
        self.model = self.model.to(self.device)

        # --- Loss Function ---
        # CrossEntropyLoss for 6-class classification.
        # It applies LogSoftmax internally, so the model outputs raw logits.
        self.criterion = nn.CrossEntropyLoss()

        # --- Optimizer ---
        # AdamW: Adam with corrected weight decay.
        # lr: initial learning rate (will be annealed)
        # weight_decay: L2 regularization strength
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        # --- LR Scheduler ---
        # Cosine annealing: smoothly reduces LR from initial to near-zero
        # over T_max epochs, then optionally restarts.
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=LR_SCHEDULER_T_MAX,
        )

        # --- Training History ---
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
        }

        # --- Early Stopping State ---
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # Ensure checkpoint directory exists
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    def train(self, max_epochs=None):
        """
        Run the full training loop.

        For each epoch:
        1. Train on all batches (model.train() mode)
        2. Evaluate on validation set (model.eval() mode)
        3. Update LR scheduler
        4. Check early stopping
        5. Save best model checkpoint

        """
        if max_epochs is None:
            max_epochs = MAX_EPOCHS

        print(f"Training on {self.device}")
        print(f"Max epochs: {max_epochs}")
        print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Weight decay: {WEIGHT_DECAY}")
        print("=" * 60)

        # TODO (Osama): Implement the training loop.

        print("WARNING: Training loop not yet implemented")
        return self.history

    def _validate(self):
        """
        Evaluate the model on the validation set.

        CRITICAL: model.eval() disables dropout and batch norm uses
        running statistics instead of batch statistics. torch.no_grad()
        disables gradient computation (saves memory and time).
        Augmentation is also disabled because val_ds has augment=False.
        """
        # TODO (Osama): Implement validation.

        return 0.0, 0.0  # PLACEHOLDER

    def _save_checkpoint(self, epoch, val_loss, val_acc):
        """
        Save model checkpoint (weights + optimizer state + metadata).

        Saves two files:
        1. best_model.pth — just the model weights (for inference)
        2. checkpoint_epoch_N.pth — full state (for resuming training)
        """
        # TODO (Osama): Implement checkpoint saving.
    
        pass

    def plot_training_curves(self, save_path=None):
        """
        Plot training and validation loss/accuracy curves.

        Useful for the final report — shows if the model is overfitting
        (training acc goes up but val acc plateaus or drops).
        """
        # TODO (Osama): Implement plotting with matplotlib.
       
        pass
