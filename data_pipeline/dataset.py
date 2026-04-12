"""
===============================================================================
data_pipeline/dataset.py — PyTorch Dataset and DataLoader
===============================================================================

"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from config import BATCH_SIZE, NUM_WORKERS, RANDOM_SEED


class MachineDataset(Dataset):
    """
    PyTorch Dataset for machine sound classification.

    Loads audio files, applies preprocessing (denoise + trim + normalize),
    extracts mel spectrograms, and returns fixed-size tensors with labels.

    Example
    -------
    >>> train_ds = MachineDataset(train_files, train_labels, augment=True)
    >>> val_ds = MachineDataset(val_files, val_labels, augment=False)
    >>> tensor, label = train_ds[0]
    >>> print(f"Shape: {tensor.shape}, Label: {label}")
    """

    def __init__(self, file_paths, labels, augment=False):
        assert len(file_paths) == len(labels), \
            f"Mismatch: {len(file_paths)} files but {len(labels)} labels"

        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Load and process a single audio file.

        This method chains the full pipeline:
        1. EL sir's preprocessing: load → resample → normalize → denoise → trim
        2. sala7's feature extraction: mel spectrogram → pad/trim → augment
        3. Return tensor + label

        """
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # TODO (JSON): Implement the full processing pipeline.
      
        from preprocessing import preprocess_audio
        from features import audio_to_tensor

        audio, sr = preprocess_audio(file_path)
        tensor = audio_to_tensor(audio, sr, augment=self.augment)

        return tensor, label


class InferenceDataset(Dataset):
    """
    PyTorch Dataset for inference (test time).

    Similar to MachineDataset but:
    - NO labels (we don't know them — that's what we're predicting)
    - NO augmentation (never augment during inference)
    - Files are loaded in INTEGER ORDER (1.wav, 2.wav, ..., not string order)

    CRITICAL: Files MUST be processed in integer order (1, 2, 3, ... 100),
    NOT string order (1, 10, 100, 2, 20...). Your results.txt must match
    input file order exactly or the grader maps every prediction to the
    wrong file.
    -------
    >>> test_ds = InferenceDataset("data/")
    >>> tensor = test_ds[0]  # First file (1.wav)
    >>> print(f"Shape: {tensor.shape}")
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir

        # --- CRITICAL: Sort files in INTEGER order, not string order ---
        # String sort: 1.wav, 10.wav, 100.wav, 2.wav, 20.wav  ← WRONG
        # Integer sort: 1.wav, 2.wav, 3.wav, ..., 10.wav, ..., 100.wav  ← CORRECT
        all_files = [f for f in os.listdir(data_dir) if f.endswith(".wav")]
        self.file_paths = sorted(
            all_files,
            key=lambda f: int(os.path.splitext(f)[0])  # Sort by integer filename
        )
        self.file_paths = [os.path.join(data_dir, f) for f in self.file_paths]

        print(f"InferenceDataset: Found {len(self.file_paths)} test files")
        if len(self.file_paths) > 0:
            print(f"  First file: {self.file_paths[0]}")
            print(f"  Last file:  {self.file_paths[-1]}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Load and process a single test audio file (no label, no augmentation).
        """
        file_path = self.file_paths[idx]

        # Full pipeline: preprocess → feature extract (no augmentation)
        from preprocessing import preprocess_audio
        from features import audio_to_tensor

        audio, sr = preprocess_audio(file_path)
        tensor = audio_to_tensor(audio, sr, augment=False)  # NEVER augment test data

        return tensor


def create_data_loaders(splits):
    """
    Create PyTorch DataLoaders for train, val, and test splits.

    This is the function Osama imports to get ready-to-use data loaders.

    """
    # TODO (JSON): Create DataLoaders for each split.
    
    print("WARNING: create_data_loaders() not yet implemented")
    return {"train": None, "val": None, "test": None}
