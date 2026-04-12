"""
===============================================================================
features/padding.py — Fixed-Size Spectrogram Padding / Trimming
===============================================================================

"""

import numpy as np
from config import FIXED_TIME_FRAMES


def pad_or_trim_spectrogram(mel_spec, target_length=None):
    """
    Pad or trim a mel spectrogram to a fixed number of time frames.

    Notes
    -----
    - Zero-padding doesn't add energy — padded regions are "silence" in the
      spectrogram and the CNN will learn to ignore them.
    - Trimming from the right assumes the important machine sound is at the
      beginning. After silence removal, this is guaranteed.
   
    """
    if target_length is None:
        target_length = FIXED_TIME_FRAMES

    n_mels, current_length = mel_spec.shape

    # TODO (sala7): Implement padding/trimming logic.
   
    if current_length < target_length:
        pad_width = target_length - current_length
        mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode="constant")
    elif current_length > target_length:
        mel_spec = mel_spec[:, :target_length]

    return mel_spec
