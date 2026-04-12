"""
===============================================================================
features/mel_spectrogram.py — Mel Spectrogram Extraction
===============================================================================

"""

import numpy as np
import librosa
from config import N_MELS, N_FFT, HOP_LENGTH, FMAX, MEL_POWER

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
FMAX = 8000
MEL_POWER = 2.0  # 2.0 = power spectrogram (standard), 1.0 = energy spectrogram

# Toggle whether to scale the final output dB array to exactly [0.0, 1.0]
# HIGHLY RECOMMENDED for CNN inputs to keep gradients stable.
# why did I choose to use min-max scaling here? 
# because the log-mel spectrogram can have a wide range of values depending on the loudness of the audio, 
# and scaling it to [0.0, 1.0] ensures that the CNN receives inputs in a consistent range
# which will make the weights explode in different directions when The data is fed to the CNN
# all weights will be standradized 
APPLY_MIN_MAX_SCALING = True

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
    mel_spec = librosa.feature.melspectrogram(
          y=audio,
          sr=sr,
          n_fft=N_FFT,
          hop_length=HOP_LENGTH,
          n_mels=N_MELS,
          fmax=FMAX,
          power=MEL_POWER
      )

      # Convert to log scale (dB)
      # Using ref=np.max sets the absolute loudest point in this file to 0 dB, 
      # making all other values negative relative to the peak.
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max) # so here I get the loudest point in the array and make it my refrence to be the zero decibel

      # Per-sample normalization to [0.0, 1.0]
    if APPLY_MIN_MAX_SCALING:
        # Prevent division by zero if the file is completely silent
        ptp = log_mel_spec.max() - log_mel_spec.min()
        if ptp > 1e-6:
            log_mel_spec = (log_mel_spec - log_mel_spec.min()) / ptp
        else:
            # If the file is just silent static, zero it out entirely
            log_mel_spec = np.zeros_like(log_mel_spec)

    return log_mel_spec
