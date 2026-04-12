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

   try:
        # librosa.load automatically handles resampling if sr != native_sr
        #  I used kaiser_best as it's very strict mathematical filter prevents aliasing
        audio, sr = librosa.load(
            file_path,
            sr=TARGET_SR,
            mono=True,             # Ensure 1D array for CNN
            res_type="kaiser_best" 
        )
        
        # Safety check: if audio is completely empty
        if audio.size == 0:
            warnings.warn(f"Warning: Loaded audio from {file_path} is completely empty.")
            
        return audio, sr

    except Exception as e:
        raise RuntimeError(f"Failed to load or resample audio file at {file_path}. Error: {e}")
