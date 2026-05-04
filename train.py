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
    trainer.plot_training_curves(save_path="checkpoints/training_curves.png")
    print()

    # =========================================================================
    # Step 5: Final evaluation on TEST set (ONCE only!)
    # =========================================================================
    print("[Step 5/5] Final evaluation on test set...")
    from training import evaluate_model, generate_confusion_matrix

    # 1. Load the absolute best weights from the early stopping checkpoint
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    model.eval() # Freeze the model for testing

    # 2. Run the actual evaluation
    print("Evaluating on the sealed Test Set. Please wait...")
    
    # Catching Osama's dictionary output correctly:
    results = evaluate_model(model, loaders["test"], DEVICE)
    
    test_acc = results["accuracy"]
    all_preds = results["all_preds"]
    all_labels = results["all_labels"]
    per_class_acc = results["per_class_acc"]

    print("\n" + "="*50)
    print(f"FINAL GLOBAL TEST ACCURACY: {test_acc:.2f}%")
    print(f"BALANCED ACCURACY:      {results['balanced_accuracy']:.2f}%")
    print(f"MACRO-F1 SCORE:         {results['macro_f1']:.4f}")
    print(f"AGGREGATE FAULT RECALL:  {results['aggregate_fault_recall']:.2f}%")
    print("="*50)
    
    print("Per-Class Detailed Breakdown:")
    print(f"{'Class Name':<25} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6}")
    print("-" * 55)
    for i, class_name in enumerate(per_class_acc.keys()):
        acc = per_class_acc[class_name]
        prec = results['per_class_precision'][i] * 100
        rec = results['per_class_recall'][i] * 100
        print(f"{class_name:<25} | {acc:>5.1f}% | {prec:>5.1f}% | {rec:>5.1f}%")
    print("="*50 + "\n")

    # 3. Generate the actual image
    generate_confusion_matrix(all_labels, all_preds, save_path="checkpoints/confusion_matrix.png")

    print("\nTraining pipeline complete.")

if __name__ == "__main__":
    main()