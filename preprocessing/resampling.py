"""
===============================================================================
preprocessing/resampling.py — Audio Loading and Resampling
===============================================================================

"""

import librosa
import numpy as np
from config import TARGET_SR


def resample_audio(file_path):
    """
    Load a WAV file and resample it to the target sampling rate.

    Parameters
    ----------
    file_path : str
        Path to the .wav audio file.

    Returns
    -------
    audio : np.ndarray
        1D numpy array of audio samples, resampled to TARGET_SR.
    sr : int
        The target sampling rate (always TARGET_SR).

    Notes
    -----
    - librosa.load() with sr=TARGET_SR automatically resamples on load.
    - mono=True ensures we get a 1D array (no stereo channels).
    - If the file is already at TARGET_SR, no resampling is performed (fast path).

    """
    # TODO (JSON): Implement audio loading and resampling.

    audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
    return audio, sr
