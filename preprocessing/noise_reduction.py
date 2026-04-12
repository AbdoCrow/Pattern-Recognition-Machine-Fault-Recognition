"""
===============================================================================
preprocessing/noise_reduction.py — Spectral Noise Reduction
===============================================================================

"""

import numpy as np
import noisereduce as nr
from config import NOISE_PROFILE_DURATION, TARGET_SR


def reduce_noise(audio, sr):
    """
    Apply spectral noise reduction to the audio signal.

    Estimates the noise profile from the first NOISE_PROFILE_DURATION seconds
    of the recording, then uses spectral subtraction to remove stationary noise.

    Parameters
    ----------
    audio : np.ndarray
        1D array of audio samples (already resampled to TARGET_SR).
    sr : int
        Sampling rate of the audio.

    Returns
    -------
    np.ndarray
        Denoised audio signal, same length as input.

    Notes
    -----
    - `prop_decrease`: How much to reduce the noise (0.0 = no reduction,
      1.0 = full removal). Start with 0.8 and tune based on results.
    - If the noise profile segment is too short (< 0.1s), fall back to
      using the entire clip for noise estimation with a lower prop_decrease.

    """
    # Calculate how many samples correspond to the noise profile duration
    noise_samples = int(sr * NOISE_PROFILE_DURATION)

    # --- Estimate noise profile ---
    if len(audio) > noise_samples and noise_samples > int(sr * 0.1):
        # Use the first N seconds as the noise reference
        noise_clip = audio[:noise_samples]
        prop_decrease = 0.8  # Moderate reduction
    else:
        # Fallback: use the whole clip (less accurate, so be conservative)
        noise_clip = audio
        prop_decrease = 0.5  # Conservative reduction

    # --- Apply spectral noise reduction ---
    # TODO (EL sir): Implement the actual noise reduction call.
    cleaned = nr.reduce_noise(  y = audio,
        sr = sr,
        y_noise = noise_clip,
        prop_decrease = prop_decrease
    )  # PLACEHOLDER — replace with actual implementation

    return cleaned
