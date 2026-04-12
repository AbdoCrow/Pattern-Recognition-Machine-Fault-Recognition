"""
===============================================================================
preprocessing/__init__.py — Audio Preprocessing Pipeline
===============================================================================

This package contains all audio-level preprocessing steps that happen BEFORE
feature extraction (mel spectrograms). The pipeline order is:

    Raw WAV → Resampling (JSON) → Normalization (JSON)
            → Noise Reduction (EL sir) → Silence Removal (EL sir)
            → Clean numpy array ready for feature extraction (sala7)

OWNERS: EL sir (noise_reduction, silence_removal)
        JSON   (normalization, resampling)
===============================================================================
"""

from preprocessing.resampling import resample_audio
from preprocessing.normalization import normalize_volume
from preprocessing.noise_reduction import reduce_noise
from preprocessing.silence_removal import remove_silence


def preprocess_audio(file_path):
    """
    Full preprocessing pipeline: load → resample → normalize → denoise → trim.

    This is the main entry point that JSON's Dataset class and infer.py will call.
    It chains all four preprocessing steps in the correct order and returns
    a clean numpy array ready for mel spectrogram extraction.

    Parameters
    ----------
    file_path : str
        Absolute path to the .wav file.

    Returns
    -------
    audio : np.ndarray
        1D numpy array of the preprocessed audio waveform.
    sr : int
        Sampling rate (always TARGET_SR after resampling).

    Pipeline Order & Rationale
    --------------------------
    1. Resample FIRST — all downstream steps assume a fixed sample rate.
       If we denoise before resampling, the noise profile frequency bins
       won't match after resampling.
    2. Normalize SECOND — bring all recordings to the same loudness level
       before noise estimation, so the noise threshold is consistent.
    3. Denoise THIRD — remove background hum/motor noise.
    4. Trim LAST — now that noise is gone, energy-based silence detection
       can accurately find where the real machine sound starts/ends.
    """
    # Step 1: Load and resample to TARGET_SR
    audio, sr = resample_audio(file_path)

    # Step 2: Volume normalization (peak or RMS)
    audio = normalize_volume(audio)

    # Step 3: Noise reduction
    audio = reduce_noise(audio, sr)

    # Step 4: Silence removal (trim leading/trailing silence)
    audio = remove_silence(audio, sr)

    return audio, sr
