"""
===============================================================================
features/mel_spectrogram.py — Mel Spectrogram Extraction
===============================================================================

"""

import numpy as np
import librosa
from config import N_MELS, N_FFT, HOP_LENGTH, FMAX, MEL_POWER


def extract_mel_spectrogram(audio, sr):
    """
    Extract a log-mel spectrogram from an audio waveform.

    Notes
    -----
    - The output is in log (dB) scale, normalized per-sample.
    - ref=np.max in power_to_db means 0 dB = the loudest point in this sample.
      All other values are negative dB below the peak.
    - This per-sample normalization is important because sala7's augmentation
      thresholds and EL sir's CNN batch normalization depend on consistent
      value ranges.

    """
    # TODO (sala7): Implement mel spectrogram extraction.

    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmax=FMAX, power=MEL_POWER,
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec
