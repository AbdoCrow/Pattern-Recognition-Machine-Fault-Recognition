"""
===============================================================================
preprocessing/normalization.py — Volume / Amplitude Normalization
===============================================================================

"""

import numpy as np


def normalize_volume(audio, method="rms", target_rms=0.1):
    """
    Normalize the volume of an audio signal.

    Parameters
    ----------
    audio : np.ndarray
        1D array of audio samples.
    method : str, optional
        Normalization method: "peak" or "rms". Default is "rms".
    target_rms : float, optional
        Target RMS value when method="rms". Default is 0.1.

    Returns
    -------
    np.ndarray
        Volume-normalized audio signal.

    Notes
    -----
    - Always check for silence (all zeros) before dividing — division by zero
      will produce NaN/Inf values that corrupt the entire pipeline.
    - The normalized audio should be clipped to [-1.0, 1.0] to avoid clipping
      artifacts when writing back to WAV (if needed for debugging).

    """
    if method == "peak":
        peak = np.max(np.abs(audio))
        if peak < 1e-6:  # Silence check to prevent division by zero
            return audio
        normalized = audio / peak

    # --- RMS Normalization ---
    # RMS will be better because it won't consider sudden pop in loudness
    # as machine 3 is like 100 times louder machine 2
    elif method == "rms":
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-6:  # Silence check to prevent division by zero safety if the silience was too aggressive but it shouldn't be as we will use dp 40 like we said 
            return audio
        normalized = audio * (target_rms / rms)
        
    else:
        raise ValueError(f"Unknown normalization method: '{method}'. Please use 'peak' or 'rms'.")

    # Clip to [-1.0, 1.0] to prevent audio distortion/clipping artifacts
    normalized = np.clip(normalized, -1.0, 1.0)

    return normalized
