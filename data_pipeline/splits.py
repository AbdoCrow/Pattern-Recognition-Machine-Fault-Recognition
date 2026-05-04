"""
===============================================================================
data_pipeline/splits.py — Stratified Train / Val / Test Split
===============================================================================

"""

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from config import (
    TRAIN_DATA_DIR,
    SPLITS_DIR,
    TRAIN_RATIO, # 0.70
    VAL_RATIO,   # 0.15
    TEST_RATIO,  # 0.15
    RANDOM_SEED,
)

CLASS_MAPPING = {
    "Machine1_normal": 0,
    "Machine1_abnormal": 1,
    "Machine2_normal": 2,
    "Machine2_abnormal": 3,
    "Machine3_normal": 4,
    "Machine3_abnormal": 5
}

def create_stratified_splits(data_dir=None):
    """
    Create stratified train/val/test splits from the training data directory.
    """
    if data_dir is None:
        data_dir = TRAIN_DATA_DIR

    file_paths = []
    labels = []

    # Read folders from the dataset
    for folder_name, class_idx in CLASS_MAPPING.items():
        class_dir = os.path.join(data_dir, folder_name)
        if not os.path.exists(class_dir):
            raise FileNotFoundError(
                f"Class directory not found: {class_dir}\n"
                f"Expected folder structure: {data_dir}/{folder_name}/"
            )
        
        for fname in sorted(os.listdir(class_dir)):
            if fname.endswith(".wav"):
                file_paths.append(os.path.join(class_dir, fname))
                labels.append(class_idx)

    file_paths = np.array(file_paths)
    labels = np.array(labels)

    # First split separate the Test set (15%) from everything else
    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        file_paths, labels,
        test_size=TEST_RATIO,
        stratify=labels,
        random_state=RANDOM_SEED,
    )

    # Second split separate Train (70%) and Val (15%) from the remaining 85%
    # We must calculate the relative ratio (15 / 85 = ~0.1764) for the second split
    relative_val_ratio = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_val_files, train_val_labels,
        test_size=relative_val_ratio,
        stratify=train_val_labels,
        random_state=RANDOM_SEED,
    )

    # Save to disk for reproducibility
    os.makedirs(SPLITS_DIR, exist_ok=True)
    splits_data = [
        ("train", train_files, train_labels),
        ("val", val_files, val_labels),
        ("test", test_files, test_labels)
    ]

    output = {}
    for name, files, lbls in splits_data:
        split_dict = {
            "files": files.tolist(),
            "labels": lbls.tolist(),
        }
        
        # Save JSON -No bun intended- files so the Dataset class can load them later
        with open(os.path.join(SPLITS_DIR, f"{name}_files.json"), "w") as f:
            json.dump(split_dict, f, indent=2)
        
        # Print the exact class distribution for Osama to verify
        dist = dict(zip(*np.unique(lbls, return_counts=True)))
        print(f"{name.upper():5s}: {len(files)} samples | Class distribution: {dist}")
        
        output[name] = split_dict

    return output