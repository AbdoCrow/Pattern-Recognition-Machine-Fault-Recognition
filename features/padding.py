"""
===============================================================================
features/padding.py — Fixed-Size Spectrogram Padding / Trimming
===============================================================================

"""

import numpy as np

# we will use (top_db=40), trimmed clips average ~8.5 to 9.0 seconds.
# Calculation: 9.0 seconds * 16000 Hz / 512 hop_length = ~281 frames.
DEFAULT_FIXED_FRAMES = 281 

# Padding strategy. Options: "constant" (fill with a number) or "edge" (repeat last frame)
PAD_MODE = "constant" 

# Log-mel spectrograms use negative numbers for silence (e.g., -80.0). 
# Padding with absolute 0.0 might accidentally represent maximum volume.
# Setting this to True forces the pad to use the quietest part of the current file.
USE_MIN_VALUE = True  
CONSTANT_PAD_VALUE = 0.0


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
        target_length = DEFAULT_FIXED_FRAMES

    n_mels, current_length = mel_spec.shape

    if current_length < target_length:
        # --- PAD: Add values on the right ---
        pad_width = target_length - current_length
        
        if PAD_MODE == "constant":
            # Decide what "silence" means mathematically
            # so here I try to use the lowest possible sound in the current spectrogram as the padding value, which is more realistic than using a fixed constant like 0.0 that might not represent silence in log-mel space.
            pad_val = mel_spec.min() if USE_MIN_VALUE else CONSTANT_PAD_VALUE
            
            
            mel_spec = np.pad(
                mel_spec,
                pad_width=((0, 0), (0, pad_width)),  # Only pad the time axis, not the mel axis
                mode="constant",
                constant_values=pad_val
            )
        else:
            # Fallback for other numpy pad modes "edge"
            # so in this method we just repeat the last frame of the spectrogram, which is a common padding strategy that doesn't introduce new values but extends the existing pattern. 
            # This can be useful if the end of the spectrogram contains relevant information that we want to preserve.
            # also depending on the CNN we need to try and error for this as in the CNN it might see the sudden 
            # silience as a FEATURE 
            mel_spec = np.pad(
                mel_spec,
                pad_width=((0, 0), (0, pad_width)),
                mode=PAD_MODE
            )
            
    elif current_length > target_length:
        # --- TRIM: Cut from the right ---
        mel_spec = mel_spec[:, :target_length]

    return mel_spec
