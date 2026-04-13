"""
===============================================================================
data_pipeline/dataset.py — PyTorch Dataset and DataLoader
===============================================================================

"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
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
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            # Preprocessing
            # This should handle loading, resampling, volume normalization, and silence trimming
            from preprocessing import preprocess_audio
            audio, sr = preprocess_audio(file_path)

            # Feature extraction 
            # This should handle the mel spectrogram, padding to fixed length, and augmentation
            from features import audio_to_tensor
            tensor = audio_to_tensor(audio, sr, augment=self.augment)

            # Ensure the output is a PyTorch float32 tensor
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.tensor(tensor, dtype=torch.float32)

            return tensor, label

        except Exception as e:
            # Error Handling: If a file is corrupted, warn the user and load the next file
            warnings.warn(f"Error loading {file_path}: {e}. Skipping to next file.")
            
            # Recursively try the next index (wrap around if at the end of the dataset)
            next_idx = (idx + 1) % len(self)
            return self.__getitem__(next_idx)


class InferenceDataset(Dataset):
    """
    PyTorch Dataset for inference (test time).
    Strictly orders files numerically and disables all augmentation.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir

        # Sort files in INTEGER order so the final submission matches perfectly
        all_files = [f for f in os.listdir(data_dir) if f.endswith(".wav")]
        self.file_paths = sorted(
            all_files,
            key=lambda f: int(os.path.splitext(f)[0])  
        )
        self.file_paths = [os.path.join(data_dir, f) for f in self.file_paths]

        print(f"InferenceDataset: Found {len(self.file_paths)} test files")
        if len(self.file_paths) > 0:
            print(f"  First file: {self.file_paths[0]}")
            print(f"  Last file:  {self.file_paths[-1]}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        try:
            from preprocessing import preprocess_audio
            from features import audio_to_tensor

            audio, sr = preprocess_audio(file_path)
            tensor = audio_to_tensor(audio, sr, augment=False) # NEVER augment inference data

            if not isinstance(tensor, torch.Tensor):
                tensor = torch.tensor(tensor, dtype=torch.float32)

            return tensor

        except Exception as e:
            raise RuntimeError(f"FATAL ERROR: Failed to process inference file {file_path}. Error: {e}")


def create_data_loaders(splits):
    """
    Create PyTorch DataLoaders for train, val, and test splits.
    """
    train_ds = MachineDataset(
        splits["train"]["files"],
        splits["train"]["labels"],
        augment=True,          # Augmentation ON for training
    )
    val_ds = MachineDataset(
        splits["val"]["files"],
        splits["val"]["labels"],
        augment=False,         # Augmentation OFF for validation
    )
    test_ds = MachineDataset(
        splits["test"]["files"],
        splits["test"]["labels"],
        augment=False,         # Augmentation OFF for testing
    )

    # Set random seed for reproducibility across multiple CPU workers
    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,               # Shuffle training data to prevent cyclical learning
        num_workers=NUM_WORKERS,
        pin_memory=True,            # Faster CPU to GPU data transfer
        generator=generator,
        drop_last=True,             # Critical for Osama's BatchNorm layers if the batch number wasn't divided by the data fed
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,              # Never shuffle validation
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,              # Never shuffle test
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }