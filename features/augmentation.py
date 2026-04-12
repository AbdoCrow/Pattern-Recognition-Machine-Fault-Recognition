"""
===============================================================================
features/augmentation.py — Data Augmentation for Training
===============================================================================

"""

import numpy as np
from config import FREQ_MASK_PARAM, TIME_MASK_PARAM, NOISE_STD, AUGMENT_PROB

AUGMENT_PROB = 0.5         # 50% chance to apply each augmentation independently
FREQ_MASK_PARAM = 24       # Max number of consecutive frequency bins to mask (out of 128)
TIME_MASK_PARAM = 40       # Max number of consecutive time frames to mask 
NOISE_STD = 0.05           # Standard deviation of the Gaussian noise

# note regarding the noise the random noise might push data over the limit we made 
# in the mel spectro gram to be between 0 and 1 so will need to clip

def apply_augmentation(mel_spec):
    """
    Apply data augmentation to a mel spectrogram (training only).

    Notes
    -----
    - This function creates a COPY of the input — the original is not modified.
    - Each call produces a DIFFERENT augmentation due to randomness.
      So each epoch, the same sample looks slightly different → diversity.
    - The augmentation is seeded by numpy's global RNG. For reproducibility
      across runs, set np.random.seed() in the training script.

    """
    augmented = mel_spec.copy()
    n_mels, time_frames = augmented.shape

    # Frequency Masking (SpecAugment) 
    if np.random.random() < AUGMENT_PROB:
        f = np.random.randint(0, FREQ_MASK_PARAM)  # Width of freq mask
        f0 = np.random.randint(0, n_mels - f)      # Starting freq bin
        augmented[f0:f0 + f, :] = 0.0              # Zero out the horizontal band

    # Time Masking (SpecAugment) 
    if np.random.random() < AUGMENT_PROB:
        t = np.random.randint(0, TIME_MASK_PARAM)    # Width of time mask
        t0 = np.random.randint(0, time_frames - t)   # Starting time frame
        augmented[:, t0:t0 + t] = 0.0                # Zero out the vertical block

    # Gaussian Noise Injection 
    if np.random.random() < AUGMENT_PROB:
        noise = np.random.normal(0, NOISE_STD, augmented.shape)
        augmented = augmented + noise
        
        # Safety clip to ensure noise doesn't break the [0, 1] normalization bounds
        augmented = np.clip(augmented, 0.0, 1.0)

    return augmented