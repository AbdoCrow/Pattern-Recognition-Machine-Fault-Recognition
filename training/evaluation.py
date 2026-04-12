"""
===============================================================================
training/evaluation.py — Performance Analysis and Error Diagnostics
===============================================================================
"""

import torch
import numpy as np
from config import NUM_CLASSES, DEVICE


# Class name mapping for readable reports
CLASS_NAMES = [
    "Machine 1 Normal",
    "Machine 1 Abnormal",
    "Machine 2 Normal",
    "Machine 2 Abnormal",
    "Machine 3 Normal",
    "Machine 3 Abnormal",
]


def evaluate_model(model, data_loader, device=None):
    """
    Evaluate a trained model on a dataset (val or test).
    """
    if device is None:
        device = DEVICE

    # TODO (Osama): Implement model evaluation.

    print("WARNING: evaluate_model() not yet implemented")
    return {"accuracy": 0.0, "all_preds": [], "all_labels": [], "per_class_acc": {}}


def generate_confusion_matrix(all_labels, all_preds, save_path=None):
    """
    Generate and plot a confusion matrix.

    Notes
    -----
    - Use this after evaluate_model() to visualize misclassification patterns.
    - Look for off-diagonal clusters to understand systematic errors.
    - Include this plot in the final report.

    """
    # TODO (Osama): Implement confusion matrix generation.

    print("WARNING: generate_confusion_matrix() not yet implemented")
