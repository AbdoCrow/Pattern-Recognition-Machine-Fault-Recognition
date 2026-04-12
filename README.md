# Machine Sound Classification — CNN Deep Learning Pipeline

## Project Overview

6-class machine sound classification using a CNN trained on mel spectrograms.
Given a WAV recording of a machine, the model predicts whether it's operating
normally or abnormally, across 3 different machines.

| Class | Description |
|-------|------------------------|
| 0 | Machine 1 — Normal |
| 1 | Machine 1 — Abnormal |
| 2 | Machine 2 — Normal |
| 3 | Machine 2 — Abnormal |
| 4 | Machine 3 — Normal |
| 5 | Machine 3 — Abnormal |

---

## Project Structure

```
project/
│
├── config.py                    # Central config — ALL constants live here
├── train.py                     # Training entry point
├── infer.py                     # Inference entry point (submission)
├── requirements.txt             # Pinned dependencies
├── Dockerfile                   # Container for submission
├── README.md                    # This file
│
├── data/                        # Test WAVs (examiner places files here)
│   ├── 1.wav
│   ├── 2.wav
│   └── ...
│
├── train_data/                  # Training data (organized by class)
│   ├── class_0/                 #   Machine 1 Normal
│   ├── class_1/                 #   Machine 1 Abnormal
│   ├── class_2/                 #   Machine 2 Normal
│   ├── class_3/                 #   Machine 2 Abnormal
│   ├── class_4/                 #   Machine 3 Normal
│   └── class_5/                 #   Machine 3 Abnormal
│
├── checkpoints/                 # Saved model weights
│   └── best_model.pth           #   Best model (used by infer.py)
│
├── splits/                      # Train/val/test split metadata
│   ├── train_files.json
│   ├── val_files.json
│   └── test_files.json
│
├── preprocessing/               # Audio preprocessing (Before features)
│   ├── __init__.py              #   Full pipeline: resample→normalize→denoise→trim
│   ├── resampling.py            #   [JSON]   Resample all audio to 16 kHz
│   ├── normalization.py         #   [JSON]   Volume normalization (peak/RMS)
│   ├── noise_reduction.py       #   [EL sir] Spectral noise subtraction
│   └── silence_removal.py       #   [EL sir] Energy-based silence trimming
│
├── features/                    # Feature extraction (Audio → Tensor)
│   ├── __init__.py              #   Full pipeline: mel spec→pad→augment→tensor
│   ├── mel_spectrogram.py       #   [sala7]  Log-mel spectrogram extraction
│   ├── padding.py               #   [sala7]  Fixed-size padding/trimming
│   └── augmentation.py          #   [sala7]  SpecAugment + noise injection
│
├── model/                       # CNN Architecture
│   ├── __init__.py
│   └── cnn.py                   #   [EL sir] Conv2D blocks + classifier head
│
├── data_pipeline/               # Data Loading Pipeline
│   ├── __init__.py
│   ├── dataset.py               #   [JSON]   PyTorch Dataset + DataLoader
│   └── splits.py                #   [JSON]   Stratified train/val/test split
│
└── training/                    # Training & Evaluation
    ├── __init__.py
    ├── trainer.py               #   [Osama]  Training loop + optimizer + scheduler
    └── evaluation.py            #   [Osama]  Confusion matrix + error analysis
```

---

## Data Pipeline Flow

```
                         PREPROCESSING                        FEATURES                    MODEL
                    ┌─────────────────────┐            ┌──────────────────┐        ┌──────────────┐
    Raw WAV ──────► │ 1. Resample (16kHz) │            │                  │        │              │
                    │ 2. Normalize volume │ ──────────►│ Mel Spectrogram  │───────►│   CNN        │
                    │ 3. Denoise          │            │ Pad/Trim (256)   │        │   6-class    │
                    │ 4. Trim silence     │            │ Augment (train)  │        │   output     │
                    └─────────────────────┘            └──────────────────┘        └──────────────┘
                        JSON + EL sir                        sala7                     EL sir
```

---

## Team Responsibilities

### EL sir — Preprocessing + Model Architecture
| File | Status | Description |
|------|--------|-------------|
| `preprocessing/noise_reduction.py` | TODO | Spectral noise subtraction using noisereduce |
| `preprocessing/silence_removal.py` | TODO | Energy-based VAD trimming (top_db=20) |
| `model/cnn.py` | Skeleton | CNN architecture (Conv blocks + AdaptiveAvgPool) |

### JSON — Data Pipeline
| File | Status | Description |
|------|--------|-------------|
| `preprocessing/normalization.py` | TODO | Peak/RMS volume normalization |
| `preprocessing/resampling.py` | Basic | Resample to 16 kHz via librosa |
| `data_pipeline/splits.py` | TODO | Stratified 70/15/15 split |
| `data_pipeline/dataset.py` | Skeleton | PyTorch Dataset + DataLoader |

### sala7 — Feature Extraction + Augmentation
| File | Status | Description |
|------|--------|-------------|
| `features/mel_spectrogram.py` | Basic | Log-mel spectrogram (n_mels=128) |
| `features/padding.py` | Basic | Pad/trim to 256 time frames |
| `features/augmentation.py` | TODO | SpecAugment + Gaussian noise |

### Osama — Training + Evaluation + Submission
| File | Status | Description |
|------|--------|-------------|
| `training/trainer.py` | TODO | Training loop (AdamW + cosine annealing) |
| `training/evaluation.py` | TODO | Confusion matrix + per-class accuracy |
| `train.py` | Skeleton | Main training entry point |
| `infer.py` | Complete | Inference script (integer file ordering!) |
| `Dockerfile` | Complete | Container for submission |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Organize your training data
```bash
train_data/
├── class_0/    # Machine 1 Normal .wav files
├── class_1/    # Machine 1 Abnormal .wav files
├── class_2/    # Machine 2 Normal .wav files
├── class_3/    # Machine 2 Abnormal .wav files
├── class_4/    # Machine 3 Normal .wav files
└── class_5/    # Machine 3 Abnormal .wav files
```

### 3. Verify CNN architecture shapes
```bash
python -m model.cnn
```

### 4. Train the model
```bash
python train.py
```

### 5. Run inference
```bash
python infer.py
```

### 6. Docker submission
```bash
docker build -t machine-classifier .
docker run -v /path/to/test/data:/app/data machine-classifier
```

---

## Critical Rules

1. **File ordering**: Test files MUST be processed in **integer order** (1, 2, 3, ..., 100), NOT string order (1, 10, 100, 2, ...)
2. **results.txt**: One predicted class (0-5) per line, matching file order
3. **time.txt**: Average time per file, rounded to 3 decimal places
4. **No console output**: Remove all debug prints before submission
5. **Test set sealed**: NEVER touch the test split until final evaluation
6. **Augmentation training-only**: Never augment validation or test data

---

## Key Configuration (config.py)

| Parameter | Value | Owner |
|-----------|-------|-------|
| `TARGET_SR` | 16000 Hz | JSON |
| `N_MELS` | 128 | sala7 + EL sir |
| `N_FFT` | 1024 | sala7 |
| `HOP_LENGTH` | 512 | sala7 |
| `FIXED_TIME_FRAMES` | 256 | sala7 |
| `BATCH_SIZE` | 32 | Osama |
| `LEARNING_RATE` | 1e-3 | Osama |
| `NUM_CLASSES` | 6 | EL sir |
| `CNN_FILTERS` | [32, 64, 128, 256] | EL sir |