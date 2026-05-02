"""
===============================================================================
config.py — Central Configuration for the Machine Sound Classification Pipeline
===============================================================================

PURPOSE:
    This file is the SINGLE SOURCE OF TRUTH for every hyperparameter,
    file path, and constant used across the project. Every team member
    should import values from here instead of hard-coding numbers.

    If you need to change a value (e.g. sample rate, n_mels, batch size),
    change it HERE and it propagates everywhere automatically.

OWNER: Shared (everyone imports from here)
===============================================================================
"""

import os

# =============================================================================
# 1. PATH CONFIGURATION
# =============================================================================
# Root directory of the project (where this file lives)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Directory where the examiner places test .wav files
# CRITICAL: infer.py reads from this exact folder
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Directory for the training dataset (organized by class)
TRAIN_DATA_DIR = os.path.join(PROJECT_ROOT, "train_data")

PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_features")

# Directory where trained model checkpoints are saved
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

# Output files required by the submission
RESULTS_FILE = os.path.join(PROJECT_ROOT, "results.txt")
TIME_FILE = os.path.join(PROJECT_ROOT, "time.txt")

# Directory for split metadata (train/val/test file lists)
SPLITS_DIR = os.path.join(PROJECT_ROOT, "splits")


# =============================================================================
# 2. AUDIO PREPROCESSING CONSTANTS
# =============================================================================
# Target sampling rate — all audio is resampled to this before processing.
# 16 kHz is standard for machine sound analysis; Nyquist limit = 8 kHz.
# Owner: JSON (resampling.py)
TARGET_SR = 16000

# Silence removal threshold in dB.
# Frames quieter than this (relative to peak) are considered silence.
# 20 dB is a safe starting point for factory recordings.
# Owner: EL sir (silence_removal.py)
SILENCE_TOP_DB = 40

# Noise reduction — duration (in seconds) of the noise profile sample.
# We estimate the noise floor from the first N seconds of each clip.
# Owner: EL sir (noise_reduction.py)
NOISE_PROFILE_DURATION = 0.5  # seconds


# =============================================================================
# 3. MEL SPECTROGRAM PARAMETERS
# =============================================================================
# These MUST be agreed upon by sala7 (feature extraction) and EL sir (CNN input).
# Changing n_mels here changes the CNN input height automatically.
# Owner: sala7 (mel_spectrogram.py) — coordinated with EL sir

# Number of mel filter banks (height of the spectrogram "image")
N_MELS = 128

# FFT window size — 1024 samples @ 16 kHz = 64 ms analysis window.
# For machine fault detection, 64 ms captures one full rotation of many motors.
N_FFT = 1024

# Hop length — 512 samples = 50% overlap between consecutive frames.
# Good time resolution without quadrupling computation.
HOP_LENGTH = 512

# Maximum frequency for the mel filterbank.
# At 16 kHz sampling rate, Nyquist = 8 kHz, so fmax = 8000.
FMAX = 8000

# Power for the mel spectrogram (2.0 = power spectrogram)
MEL_POWER = 2.0


# =============================================================================
# 4. FIXED-SIZE TENSOR PARAMETERS
# =============================================================================
# After silence removal, audio clips vary in length. We must pad/trim
# spectrograms to a uniform time dimension for batching.
#
# CALCULATION:
#   Original audio ≈ 11 seconds → 11 * 16000 = 176,000 samples
#   Time frames = ceil(176000 / 512) = 344 frames (full 11s)
#   After silence trimming, clips are shorter → 256 frames ≈ 8.2 seconds
#   is a reasonable target that captures the machine sound while
#   discarding silence.
#

# Final input shape to the CNN: (batch, channels, n_mels, time_frames)
# channels = 1 (grayscale spectrogram)
CNN_INPUT_CHANNELS = 1


# =============================================================================
# 5. DATA SPLIT RATIOS
# =============================================================================
# Stratified split ratios — every class appears in every split at the same proportion.
# Owner: JSON (splits.py)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility across all random operations
RANDOM_SEED = 42


# =============================================================================
# 6. MODEL ARCHITECTURE PARAMETERS
# =============================================================================
# Number of output classes:
#   0 = Machine 1 Normal, 1 = Machine 1 Abnormal,
#   2 = Machine 2 Normal, 3 = Machine 2 Abnormal,
#   4 = Machine 3 Normal, 5 = Machine 3 Abnormal
# Owner: EL sir (cnn.py)
NUM_CLASSES = 6

# Convolutional layer filter counts (depth progression)
# Each successive layer doubles the filters to capture more complex patterns.
CNN_FILTERS = [32, 64, 128, 256]

# Kernel size for all Conv2D layers
CNN_KERNEL_SIZE = 3

# Padding for Conv2D layers (1 = 'same' padding with kernel_size=3)
CNN_PADDING = 1

# Pool size for MaxPool2d layers
CNN_POOL_SIZE = 2

# Output size of AdaptiveAvgPool2d before the classifier head
# This makes the model accept any time-length input gracefully.
ADAPTIVE_POOL_OUTPUT = (4, 4)


# =============================================================================
# 7. TRAINING HYPERPARAMETERS
# =============================================================================
# Owner: Osama (trainer.py)

# Optimizer: AdamW (corrects weight decay application vs vanilla Adam)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Batch size for training DataLoader
BATCH_SIZE = 64

# Maximum number of training epochs
MAX_EPOCHS = 100

# Early stopping — stop if val loss doesn't improve for this many epochs
EARLY_STOPPING_PATIENCE = 10

# Learning rate scheduler — cosine annealing
LR_SCHEDULER_T_MAX = MAX_EPOCHS  # period of the cosine cycle

# Number of DataLoader workers for parallel data loading
NUM_WORKERS = 8


# =============================================================================
# 8. AUGMENTATION PARAMETERS
# =============================================================================
# Owner: sala7 (augmentation.py)
# These are applied ONLY during training (not val/test).

# SpecAugment: number of frequency bands to mask
FREQ_MASK_PARAM = 20

# SpecAugment: number of time steps to mask
TIME_MASK_PARAM = 30

# Gaussian noise injection — standard deviation
NOISE_STD = 0.005

# Probability of applying each augmentation
AUGMENT_PROB = 0.5


# =============================================================================
# 9. INFERENCE PARAMETERS
# =============================================================================
# Path to the best saved model checkpoint (used by infer.py)
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

# Device selection for inference (auto-detect GPU)
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
