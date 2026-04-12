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
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
)


def create_stratified_splits(data_dir=None):
    """
    Create stratified train/val/test splits from the training data directory.

    Scans the data directory for class folders, collects all file paths and
    labels, then splits them using stratified sampling to ensure each class
    is proportionally represented in every split.

    Side Effects
    ------------
    Saves split metadata to SPLITS_DIR for reproducibility:
        splits/train_files.json
        splits/val_files.json
        splits/test_files.json

    """
    if data_dir is None:
        data_dir = TRAIN_DATA_DIR

    # TODO (JSON): Implement stratified split logic.
   
    print("WARNING: create_stratified_splits() not yet implemented")
    return {
        "train": {"files": [], "labels": []},
        "val":   {"files": [], "labels": []},
        "test":  {"files": [], "labels": []},
    }
