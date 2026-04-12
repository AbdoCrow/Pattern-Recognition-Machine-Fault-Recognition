"""
===============================================================================
train.py — Main Training Entry Point
===============================================================================

"""

import os
import sys
import torch
import numpy as np
from config import RANDOM_SEED, DEVICE

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


def main():
    """
    Complete training pipeline from data loading to model evaluation.
    """
    print("=" * 60)
    print("Machine Sound Classification — Training Pipeline")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Random seed: {RANDOM_SEED}")
    print()

    # =========================================================================
    # Step 1: Create data splits (JSON)
    # =========================================================================
    print("[Step 1/5] Creating stratified data splits...")
    from data_pipeline import create_stratified_splits
    splits = create_stratified_splits()
    print()

    # =========================================================================
    # Step 2: Create DataLoaders (JSON)
    # =========================================================================
    print("[Step 2/5] Creating DataLoaders...")
    from data_pipeline import create_data_loaders
    loaders = create_data_loaders(splits)
    print()

    # =========================================================================
    # Step 3: Initialize model (EL sir)
    # =========================================================================
    print("[Step 3/5] Initializing CNN model...")
    from model import MachineSoundCNN
    model = MachineSoundCNN()

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: MachineSoundCNN")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Device: {DEVICE}")
    print()

    # =========================================================================
    # Step 4: Train the model (Osama)
    # =========================================================================
    print("[Step 4/5] Training model...")
    from training import Trainer
    trainer = Trainer(model, loaders["train"], loaders["val"])
    history = trainer.train()

    # Save training curves
    # trainer.plot_training_curves(save_path="checkpoints/training_curves.png")
    print()

    # =========================================================================
    # Step 5: Final evaluation on TEST set (ONCE only!)
    # =========================================================================
    print("[Step 5/5] Final evaluation on test set...")
    from training import evaluate_model, generate_confusion_matrix

    # Step 1: Load the optimized model weights from the saved checkpoint into the CNN architecture.
    # Ensure 'map_location' is set to the current device to handle CPU/GPU compatibility.
    # Step 2: Pass the sealed test set through the model for a single-pass evaluation.
    # This must be the first and only time the model sees this data to ensure an unbiased final metric.
    # Step 3: Print the global test accuracy percentage as the primary performance benchmark.
    # Step 4: Break down accuracy by class (0-5) to identify specific machine types where the model succeeds or fails.
    # Step 5: Construct and save the Confusion Matrix plot to 'checkpoints/confusion_matrix.png'.
    # This serves as the visual proof of inter-class confusion for the final report.

    print("\nTraining pipeline complete.")
    print("Best model saved to: checkpoints/best_model.pth")


if __name__ == "__main__":
    main()
