"""
===============================================================================
features/augmentation.py — Data Augmentation for Training
===============================================================================

"""

import numpy as np
from config import FREQ_MASK_PARAM, TIME_MASK_PARAM, NOISE_STD, AUGMENT_PROB


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

    # TODO (sala7): Implement the augmentation techniques below.
   
    return augmented  # PLACEHOLDER — replace with actual implementation
