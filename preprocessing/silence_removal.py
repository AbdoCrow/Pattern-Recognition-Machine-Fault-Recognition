"""
===============================================================================
preprocessing/silence_removal.py — Energy-Based Silence Trimming
===============================================================================

"""

import numpy as np
import librosa
from config import SILENCE_TOP_DB


def remove_silence(audio, sr):
    """
    Remove leading and trailing silence from the audio signal.

    Uses energy-based detection to find where the "real" sound starts and ends,
    then trims everything outside that window.

    Parameters
    ----------
    audio : np.ndarray
        1D array of audio samples.
    sr : int
        Sampling rate of the audio.

    Returns
    -------
    np.ndarray
        Trimmed audio signal. If the entire signal is below the threshold,
        returns the original audio unchanged (safety fallback).

    Notes
    -----
    - top_db is imported from config.py (default: 20 dB).
    - A small margin (frame_length=2048, hop_length=512) is used for
      energy estimation. These are independent of the mel spectrogram
      parameters — they only control how finely we detect silence boundaries.

    """
    # TODO (EL sir): Implement silence removal.
    
    trimmed_audio = audio  # PLACEHOLDER — replace with actual implementation

    return trimmed_audio
