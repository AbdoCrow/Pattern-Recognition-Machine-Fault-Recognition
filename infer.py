"""
===============================================================================
infer.py — Inference Script for Submission
===============================================================================

"""

import os
import sys
import time
import torch
import numpy as np

# Add project root to path so imports work
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import (
    DATA_DIR,
    BEST_MODEL_PATH,
    RESULTS_FILE,
    TIME_FILE,
    DEVICE,
    NUM_CLASSES,
)
from model import MachineSoundCNN
from data_pipeline.dataset import InferenceDataset


def main():
    """
    Main inference pipeline.

    Loads the trained model, processes all test files in integer order,
    and writes predictions + timing to output files.
    """
    # =========================================================================
    # Step 1: Load the trained model
    # =========================================================================
    model = MachineSoundCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(
        torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    model = model.to(DEVICE)
    model.eval()  # CRITICAL: Disable dropout and use running BN statistics

    # =========================================================================
    # Step 2: Initialize Inference Dataset
    # =========================================================================
    # InferenceDataset handles integer sorting and live preprocessing
    dataset = InferenceDataset(DATA_DIR)
    
    if len(dataset) == 0:
        print(f"ERROR: No .wav files found in {DATA_DIR}")
        sys.exit(1)

    # =========================================================================
    # Step 3: Run inference on each file
    # =========================================================================
    predictions = []
    total_time = 0.0

    with torch.no_grad():  # No gradients needed during inference
        for i in range(len(dataset)):
            # --- Time this iteration ---
            start_time = time.time()

            # dataset[i] performs preprocessing and feature extraction
            tensor = dataset[i]

            # Add batch dimension: (1, 128, 281) → (1, 1, 128, 281)
            tensor = tensor.unsqueeze(0).to(DEVICE)

            # Forward pass through CNN
            output = model(tensor)

            # Get predicted class (index of highest logit)
            predicted_class = output.argmax(dim=1).item()
            predictions.append(predicted_class)

            # --- Record time ---
            elapsed = time.time() - start_time
            total_time += elapsed

    # =========================================================================
    # Step 4: Write results.txt
    # =========================================================================
    # Format: One predicted class (0-5) per line, matching file order
    with open(RESULTS_FILE, "w") as f:
        for pred in predictions:
            f.write(f"{pred}\n")

    # =========================================================================
    # Step 5: Write time.txt
    # =========================================================================
    # Format: Average time per file, rounded to 3 decimal places
    avg_time_per_file = total_time / len(dataset) if len(dataset) > 0 else 0.0
    with open(TIME_FILE, "w") as f:
        f.write(f"{avg_time_per_file:.3f}\n")


if __name__ == "__main__":
    main()
