"""
===============================================================================
training/evaluation.py — Performance Analysis and Error Diagnostics
===============================================================================

OWNER: Osama
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    f1_score, 
    balanced_accuracy_score, 
    precision_recall_fscore_support
)
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
    Returns overall accuracy, raw predictions/labels, and per-class accuracy.
    """
    if device is None:
        device = DEVICE

    model.eval()  # CRITICAL: Turn off Dropout and freeze BatchNorm
    
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    # No gradients needed for evaluation — saves massive amounts of RAM and time
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Store for confusion matrix later
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100. * correct / total

    # --- Calculate Per-Class Accuracy ---
    per_class_acc = {}
    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)

    for i in range(NUM_CLASSES):
        class_mask = (all_labels_np == i)
        class_total = np.sum(class_mask)
        
        if class_total > 0:
            class_correct = np.sum((all_preds_np == i) & class_mask)
            per_class_acc[CLASS_NAMES[i]] = 100. * class_correct / class_total
        else:
            per_class_acc[CLASS_NAMES[i]] = 0.0

    # --- Calculate New Metrics requested by user ---
    # 1. Macro-F1
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    # 2. Balanced Accuracy
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)

    # 3. Per-class Precision/Recall
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels=range(NUM_CLASSES))

    # 4. Normal vs Abnormal Aggregate Recall
    # Indices 0, 2, 4 = Normal; 1, 3, 5 = Abnormal
    abnormal_indices = [1, 3, 5]
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    
    abnormal_mask = np.isin(all_labels_np, abnormal_indices)
    if np.sum(abnormal_mask) > 0:
        # True positives for the 'Abnormal' super-class: 
        # Predicted ANY abnormal class when label was ANY abnormal class
        abnormal_preds = np.isin(all_preds_np[abnormal_mask], abnormal_indices)
        aggregate_fault_recall = np.mean(abnormal_preds) * 100.
    else:
        aggregate_fault_recall = 0.0

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc * 100.,
        "macro_f1": macro_f1,
        "per_class_precision": precision,
        "per_class_recall": recall,
        "aggregate_fault_recall": aggregate_fault_recall,
        "all_preds": all_preds,
        "all_labels": all_labels,
        "per_class_acc": per_class_acc
    }


def generate_confusion_matrix(all_labels, all_preds, save_path=None):
    """
    Generate and plot a stylized confusion matrix heatmap.
    """
    # Calculate the mathematical matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(NUM_CLASSES))

    # Plotting it beautifully with Seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True,          # Put the raw numbers inside the boxes
        fmt='d',             # Format as integers
        cmap='Blues',        # Blue color scale looks highly professional
        xticklabels=CLASS_NAMES, 
        yticklabels=CLASS_NAMES
    )
    
    plt.title('Test Set Confusion Matrix', fontsize=14, pad=15)
    plt.ylabel('Actual True Label', fontsize=12)
    plt.xlabel('CNN Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')  # Rotate bottom labels so they don't overlap
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[!] Confusion matrix saved to: {save_path}")
    else:
        plt.show()
        
    plt.close()