"""
===============================================================================
features/__init__.py — Feature Extraction Pipeline
===============================================================================

This package converts preprocessed audio (clean numpy arrays) into fixed-size
tensors suitable for CNN input.

Pipeline: Clean audio → Mel spectrogram → Log scale → Pad/Trim → Tensor

OWNER: sala7
===============================================================================
"""

from features.mel_spectrogram import extract_mel_spectrogram
from features.padding import pad_or_trim_spectrogram
from features.augmentation import apply_augmentation


def audio_to_tensor(audio, sr, augment=False):
    """
    Convert a preprocessed audio waveform into a fixed-size tensor for CNN input.

    This is the main entry point called by JSON's Dataset class. It chains
    mel spectrogram extraction, padding/trimming, and optional augmentation.

    Returns
    -------
    torch.Tensor
        Shape: (1, N_MELS, FIXED_TIME_FRAMES) = (1, 128, 256)
        - 1 channel (grayscale spectrogram)
        - 128 mel frequency bins
        - 256 time frames

    Pipeline
    --------
    1. extract_mel_spectrogram: audio → log-mel spectrogram (128 × T)
    2. pad_or_trim_spectrogram: (128 × T) → (128 × 256) fixed size or 344 idk man
    3. apply_augmentation: (optional) SpecAugment + noise injection
    4. Add channel dimension: (128 × 256) → (1, 128, 256)
    """
    import torch

    # Step 1: Extract log-mel spectrogram
    mel_spec = extract_mel_spectrogram(audio, sr)

    # Step 2: Pad or trim to fixed time dimension
    mel_spec = pad_or_trim_spectrogram(mel_spec)

    # Step 3: Optional augmentation (training only)
    if augment:
        mel_spec = apply_augmentation(mel_spec)

    # Step 4: Convert to tensor and add channel dimension
    # Shape: (n_mels, time_frames) → (1, n_mels, time_frames)
    tensor = torch.FloatTensor(mel_spec).unsqueeze(0)

    return tensor
