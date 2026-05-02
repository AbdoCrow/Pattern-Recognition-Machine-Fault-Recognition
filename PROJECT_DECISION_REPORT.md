# Machine Sound Classification Project Decision Report

Generated: 2026-04-29

Reference book: `Deep Learning Foundations and Concepts (Christopher Bishop  Hugh Bishop).pdf`, referred to below as Bishop and Bishop.

## Executive Summary

This project uses a practical and defensible pipeline:

```text
Raw WAV -> audio preprocessing -> log-mel spectrogram -> fixed-size tensor -> CNN -> 6-class prediction
```

The strongest design decisions already in the project are:

- Using log-mel spectrograms instead of raw waveform input.
- Using a 2D CNN because spectrograms have local time-frequency structure.
- Using stratified train/validation/test splits.
- Keeping augmentation training-only.
- Caching deterministic preprocessed features for faster training.
- Using normalization, BatchNorm, Dropout, AdamW, early stopping, and confusion-matrix evaluation.

The most important improvement to do first is to make the training data path fully correct and class-aware:

- Actually apply spectrogram augmentation in `MachineDataset` when `augment=True`.
- Add class-balanced training through weighted loss or a weighted sampler.
- Add macro-F1 or balanced accuracy to evaluation.
- Re-enable or separate the real training step in `train.py`.

Why this should come first: the data is highly imbalanced and the current cached-feature dataset says augmentation is enabled but does not apply it. Bigger models, more aggressive preprocessing, and higher sample rates may help later, but they are less urgent than making the current training setup match the intended design.

## Evidence From The Project And Dataset

### Code structure used for this report

- Audio preprocessing: `preprocessing/`
- Feature extraction: `features/`
- Dataset and splits: `data_pipeline/`
- CNN model: `model/cnn.py`
- Training and evaluation: `training/`
- Inference: `infer.py`
- Offline feature cache script: `preprocess.py`
- Prior analysis tools: `Analyze before code/eda_audio.py` and `Analyze before code/extract_and_categorize.py`

The prior EDA script checks class counts, sample rate, duration, peak amplitude, RMS amplitude, noise floor from the first segment, and trimming sensitivity for different `top_db` values. I reused that analysis intent and also measured a deterministic random sample of 160 WAV files per class.

### Dataset shape

| Class | Files | Duration in sample | Sample rate in sample | Channels | Median RMS |
|---|---:|---:|---:|---:|---:|
| Machine1_normal | 16,200 | 11.0 s | 48 kHz | mono | 0.004280 |
| Machine1_abnormal | 3,176 | 11.0 s | 48 kHz | mono | 0.006003 |
| Machine2_normal | 16,200 | 11.0 s | 48 kHz | mono | 0.002181 |
| Machine2_abnormal | 3,240 | 11.0 s | 48 kHz | mono | 0.002465 |
| Machine3_normal | 14,400 | 10.0 s | 48 kHz | mono | 0.007747 |
| Machine3_abnormal | 3,020 | 10.0 s | 48 kHz | mono | 0.005363 |

Total files: 56,236.

The normal classes are much larger than the abnormal classes:

- Normal total: 46,800 files.
- Abnormal total: 9,436 files.
- Normal share: about 83.2 percent.
- Abnormal share: about 16.8 percent.

This is why class-aware training is a high-priority improvement.

### Cached feature shape

Sampled cached features in `processed_features/` all had this shape:

```text
(1, 128, 281)
```

Their values are already normalized to approximately `[0, 1]`, which matches the intended CNN input.

## Book Reference Map

The project choices are mostly supported by these Bishop and Bishop sections:

| Topic | Book reference |
|---|---|
| Feature extraction / fixed preprocessing | Section 4.1.1, p. 113 |
| Fourier and wavelet basis functions for signal-like data | Section 4.1.1, p. 114 |
| Train/validation/test separation | Section 1.2.6, p. 14 |
| Multi-class softmax and cross-entropy | Chapter 5, especially p. 162 |
| ReLU activation | Section 6.2, p. 185 |
| Mini-batch training | Section 7.2.4, p. 216 |
| Adam optimizer | Section 7.3.3, p. 223 |
| Input normalization | Sections 7.4 and 7.4.1, pp. 224 and 226 |
| Batch normalization | Section 7.4.2, p. 227 |
| Regularization, augmentation, dropout, early stopping | Chapter 9, especially pp. 254, 257, 266, and 279 |
| Weight decay | Section 9.2, p. 260 |
| Parameter sharing | Section 9.4, p. 270 |
| CNN inductive bias, locality, hierarchy | Chapter 10, especially pp. 287 and 290 |
| Multi-dimensional convolutional tensors | Section 10.2.5, p. 295 |
| Pooling | Section 10.2.6, p. 296 |
| Denoising, masking, and Gaussian corruption as learning ideas | Section 19.1.4, p. 567 |

Important note: Bishop and Bishop is not an audio DSP textbook. It does not prescribe mel spectrograms or the exact `librosa` and `noisereduce` functions. The book supports the machine-learning reasoning: feature extraction, normalization, regularization, CNN inductive bias, and evaluation discipline.

## Overall Design Choice

### Chosen approach

Use handcrafted audio preprocessing plus log-mel feature extraction, then train a 2D CNN classifier.

### Main alternatives

| Option | Why not chosen as the main design |
|---|---|
| Feed raw waveform directly to a 1D CNN | Raw audio is much longer and harder to learn from. At 16 kHz, a 10-11 second clip still has 160,000-176,000 samples. A raw model needs more data and compute to discover frequency structure that a spectrogram gives explicitly. |
| Use MFCC features | MFCCs are compact and good for speech-like pipelines, but they discard detail that may matter for machine faults. Log-mel spectrograms preserve richer local time-frequency patterns for a CNN. |
| Use STFT spectrogram directly | STFT keeps linear frequency spacing, but mel compression gives a more compact representation and reduces input size. |
| Use a transformer | Transformers can model long-range patterns, but are heavier and need careful regularization. The current problem has a natural 2D time-frequency grid, so a CNN is the simpler first choice. |
| Use a pretrained audio model | Could help, but adds dependency, input-format constraints, and possible mismatch to machine sounds. Start with a transparent baseline first. |

### Book support

Bishop and Bishop explain that preprocessing can transform inputs into a feature space where learning is easier (Section 4.1.1, p. 113). They also explain that CNNs exploit locality, hierarchy, parameter sharing, and structured grid data (Chapter 10, pp. 287-290). A log-mel spectrogram turns audio into exactly the kind of structured grid where a CNN is appropriate.

## Preprocessing Decisions

### 1. Resampling

Code: `preprocessing/resampling.py`

Chosen:

- Load with `librosa.load`.
- Convert to mono.
- Resample to `TARGET_SR = 16000`.
- Use `res_type="kaiser_best"` for higher-quality resampling.

Why this is good:

- The dataset sample is consistently 48 kHz mono, but inference data may not always be identical.
- A fixed sample rate makes time and frequency coordinates comparable across all files.
- 16 kHz gives an 8 kHz Nyquist limit, which is usually enough for many mechanical sound patterns while reducing compute.

Options considered:

| Option | Decision |
|---|---|
| Keep native 48 kHz | Not chosen because it triples compute compared with 16 kHz and produces larger spectrograms. Revisit if faults are proven to live above 8 kHz. |
| Downsample to 8 kHz | Not chosen because it would remove everything above 4 kHz, which may be too aggressive for machine sound. |
| Use 24 kHz | Reasonable future ablation. It keeps frequencies up to 12 kHz with moderate compute. |
| Use stereo | Not needed because sampled data is mono. |

Book reference: input variables should be made comparable before learning; Bishop and Bishop discuss input normalization and preprocessing in Sections 4.1.1 and 7.4.1.

### 2. Volume normalization

Code: `preprocessing/normalization.py`

Chosen:

- Default method: RMS normalization.
- Target RMS: `0.1`.
- Safety checks prevent division by zero.
- Output clipped to `[-1.0, 1.0]`.

Why this is good:

- The sampled data shows different loudness ranges between machines.
- RMS normalization uses average energy, so one short spike does not dominate the scaling.
- Peak normalization would be more sensitive to pops or clicks.

Options considered:

| Option | Decision |
|---|---|
| No normalization | Not chosen because loudness differences can become an unintended shortcut or can make optimization harder. |
| Peak normalization | Available in the function, but not default because peak values are sensitive to outliers. |
| Dataset-level mean/std waveform normalization | Useful for some pipelines, but less direct for audio loudness than RMS. Global feature normalization can still be added later. |
| Per-machine normalization | Risky because inference may not provide machine identity separately. It could also remove useful machine-specific differences. |

Book reference: Bishop and Bishop discuss normalization as important for avoiding very large or small values and for improving gradient descent behavior (Sections 7.4 and 7.4.1, pp. 224-226).

### 3. Noise reduction

Code: `preprocessing/noise_reduction.py`

Chosen:

- Estimate a noise profile from the first `0.5` seconds.
- Use stronger reduction (`prop_decrease = 0.8`) when enough noise profile exists.
- Use conservative reduction (`prop_decrease = 0.5`) for short clips.
- Clip output to avoid artifacts outside normal waveform range.

Why this is good:

- Machine recordings may include stationary hum or environmental noise.
- A fixed noise profile is simple, fast, and explainable.
- The existing EDA script was designed to measure first-segment noise RMS, which supports this design.

Options considered:

| Option | Decision |
|---|---|
| No denoising | Safer if noise itself carries label information, but may reduce generalization to new recording environments. Should be tested as an ablation. |
| High-pass / band-pass filtering | Could be useful, but requires stronger domain knowledge about fault frequencies. |
| More aggressive spectral subtraction | Not chosen because it can create artifacts and erase weak fault signatures. |
| Learned denoising model | Too complex for the first version and requires clean/noisy training pairs or a self-supervised setup. |

Book reference: Bishop and Bishop support the broader idea that preprocessing can build invariance into the inputs (Section 9.1.3, p. 257). Their denoising autoencoder discussion also explains why masking or adding/removing noise can push models toward useful structure (Section 19.1.4, p. 567).

### 4. Silence removal

Code: `preprocessing/silence_removal.py`

Chosen:

- Use `librosa.effects.trim`.
- Use `SILENCE_TOP_DB = 40`.
- Use `frame_length = 2048` and `hop_length = 512`.
- If the trimmed signal becomes too short, return the original audio.

Why this is good:

- It removes leading/trailing low-energy parts that do not describe the machine.
- It makes important machine sound occupy more of the spectrogram.
- The fallback prevents over-trimming from breaking the pipeline.

Options considered:

| Option | Decision |
|---|---|
| No silence trimming | Simpler, but wastes spectrogram frames and may teach the CNN about silence position instead of machine state. |
| Trim with lower `top_db` such as 20 | More aggressive; may remove quiet but useful machine signals. |
| Trim with higher `top_db` such as 60 | Less aggressive; may leave too much silence/noise. |
| Voice-activity detection | Not appropriate because this is machine audio, not speech. |
| Fixed center crop | Simple, but assumes the useful part is centered. |

Book reference: this is a preprocessing-based invariance: the label should not depend on how much silence occurs before or after the useful sound. Bishop and Bishop discuss preprocessing as one way to encode invariance (Section 9.1.3, p. 257).

### 5. Preprocessing order

Code: `preprocessing/__init__.py`

Chosen order:

```text
resample -> normalize -> reduce noise -> remove silence
```

Why this order is good:

- Resampling first makes all later parameters operate at the same sample rate.
- Normalization before denoising makes noise estimation more consistent across files.
- Denoising before trimming helps silence detection avoid treating noise as real signal.
- Trimming last avoids removing useful low-level content before the signal is cleaned.

Alternative order:

| Option | Why not preferred |
|---|---|
| Trim before denoise | Background noise may prevent correct trimming. |
| Denoise before normalize | Noise thresholds become less consistent across loud and quiet recordings. |
| Feature extraction before audio cleanup | The CNN would receive more nuisance variation. |

## Feature Extraction Decisions

### 1. Log-mel spectrogram

Code: `features/mel_spectrogram.py`

Chosen:

- `N_MELS = 128`
- `N_FFT = 1024`
- `HOP_LENGTH = 512`
- `FMAX = 8000`
- `MEL_POWER = 2.0`
- Convert to decibels with `librosa.power_to_db(..., ref=np.max)`.
- Min-max scale each sample to `[0, 1]`.

Why this is good:

- It converts a long 1D waveform into a compact 2D representation.
- Local patterns in frequency and time become visible to a CNN.
- Log scaling compresses large energy differences.
- Min-max scaling makes the CNN input range stable.

Options considered:

| Option | Decision |
|---|---|
| Raw waveform | Not chosen because it is long, high-dimensional, and requires the model to learn frequency decomposition from scratch. |
| MFCC | More compact, but can remove detail that fault detection may need. |
| Linear STFT spectrogram | Good alternative, but larger and less compact than mel. |
| Constant-Q transform | Useful for musical pitch-like signals, less obviously suited to these machine recordings. |
| Learnable frontend | Future option, but more complex and less explainable. |

Book reference: Bishop and Bishop discuss fixed feature extraction in Section 4.1.1 (p. 113) and Fourier/wavelet bases for signal-like data on p. 114. They also explain CNNs for structured 2D grids in Chapter 10.

### 2. Fixed-size padding and trimming

Code: `features/padding.py`

Chosen:

- Target length: `281` frames.
- Pad on the right if shorter.
- Trim from the right if longer.
- Pad using the minimum value of the current spectrogram when constant padding is enabled.

Why this is good:

- Batching requires equal tensor shapes.
- The current cached feature sample confirms all outputs are `(1, 128, 281)`.
- Padding with the quietest value is more meaningful than padding with a value that might represent high energy after min-max scaling.

Options considered:

| Option | Decision |
|---|---|
| Full 11-second length | Would preserve everything, but increases input size and compute. |
| Variable-length model | Possible with adaptive pooling or recurrent/attention models, but harder to batch and debug. |
| Random crop during training | Useful future augmentation, but not necessary for deterministic cached features. |
| Center crop | Not chosen because after silence removal the useful sound is expected near the start. |
| Edge padding | Available, but could repeat the final pattern and create artificial evidence. |

Book reference: Bishop and Bishop describe convolutional inputs as structured tensors with spatial dimensions and channels (Section 10.2.5, p. 295). Fixed-size tensors make this structure consistent for mini-batch training.

### 3. Per-sample min-max scaling

Chosen:

- Scale each log-mel spectrogram to `[0, 1]`.

Why this is good:

- The CNN receives stable input values.
- The cached feature sample confirms values are bounded.
- This helps prevent loudness differences from dominating training.

Options considered:

| Option | Decision |
|---|---|
| No feature scaling | Not chosen because gradients and learned filters can become sensitive to raw scale. |
| Dataset-level mean/std scaling | Good future improvement, especially if computed on training set only and reused for val/test/inference. |
| Per-machine scaling | Not chosen because it can leak machine assumptions and complicates inference. |

Book reference: Bishop and Bishop emphasize input normalization for gradient descent stability (Section 7.4.1, p. 226).

## Augmentation Decisions

Code: `features/augmentation.py`

Current intended augmentations:

- Frequency masking.
- Time masking.
- Gaussian noise injection.

### Why use augmentation instead of training directly on the data?

Training directly on the data teaches the model only the exact observed examples. This is risky here because:

- Abnormal classes are much smaller than normal classes.
- Recordings may contain random noise, timing differences, and missing/weak bands.
- A CNN can overfit repeated patterns in the training set.

Augmentation creates label-preserving variations so the model learns more robust features. Bishop and Bishop present data augmentation as a regularization method and as a way to encourage invariance (Chapter 9, especially pp. 254 and 257).

### Why these augmentation types?

| Augmentation | Why it fits this project |
|---|---|
| Frequency masking | Forces the CNN not to depend on one narrow frequency band only. Useful if microphones or machines vary slightly. |
| Time masking | Forces the CNN not to depend on one exact moment in the clip. Useful because fault evidence may occur at slightly different times. |
| Gaussian noise | Simulates measurement noise and makes the model less brittle. Bishop and Bishop discuss Gaussian corruption in denoising contexts (Section 19.1.4, p. 567). |

### Why not more augmentation immediately?

| Possible augmentation | Why not chosen first |
|---|---|
| Pitch shift | Machine fault frequencies can be physically meaningful. Shifting them may change the label semantics. |
| Time stretch / speed perturbation | Could simulate RPM variation, but can also move fault frequencies. Use only small ranges and validate. |
| Gain jitter | Could help, but amplitude may carry useful machine-state information even after normalization. Use carefully. |
| Mixup | Can improve regularization, but mixed machine states may be hard to interpret physically. |
| CutMix on spectrograms | Similar concern: pasted fault regions may create unrealistic examples. |
| Random time crop | Promising, but should be paired with labels and duration checks so the crop does not remove the fault evidence. |
| Background-noise mixing | Good future option if noise clips are collected from the same environment. |

### Important current issue

`create_data_loaders()` passes `augment=True` for training, but `MachineDataset.__getitem__()` currently does this:

```python
if self.augment:
    pass
```

So training on cached `.npy` features does not currently apply the augmentation function. This is one of the top project improvements.

## Data Pipeline Decisions

### Stratified splitting

Code: `data_pipeline/splits.py`

Chosen:

- Train/val/test ratio: `70/15/15`.
- Stratified by class.
- Random seed: `42`.
- Split metadata saved to JSON.

Current split counts:

| Split | Total | Class distribution |
|---|---:|---|
| train | 39,364 | `{0: 11340, 1: 2222, 2: 11340, 3: 2268, 4: 10080, 5: 2114}` |
| val | 8,436 | `{0: 2430, 1: 477, 2: 2430, 3: 486, 4: 2160, 5: 453}` |
| test | 8,436 | `{0: 2430, 1: 477, 2: 2430, 3: 486, 4: 2160, 5: 453}` |

Why this is good:

- Every split preserves the class imbalance pattern.
- Validation and test sets include every class.
- Saved JSON files improve reproducibility.

Options considered:

| Option | Decision |
|---|---|
| Random non-stratified split | Not chosen because abnormal classes could be underrepresented in validation/test. |
| K-fold cross-validation | More reliable but much more expensive. Good for final reporting if time allows. |
| Group-aware split | Important future improvement if adjacent filenames come from the same recording session. It avoids near-duplicate leakage. |
| Use only train/validation and no test | Not chosen because final performance needs a sealed test set. |

Book reference: Bishop and Bishop explain train/validation/test separation for model selection and final evaluation in Section 1.2.6, p. 14.

### Cached features

Code: `preprocess.py` and `data_pipeline/dataset.py`

Chosen:

- Precompute deterministic features into `processed_features/`.
- Load `.npy` files during training instead of processing WAV live.
- Process raw WAV live only for inference.

Why this is good:

- Noise reduction and mel extraction are slow.
- Caching gives faster epochs and easier debugging.
- Training can focus on model learning, not CPU-heavy audio processing.

Options considered:

| Option | Decision |
|---|---|
| Process WAV live every epoch | Too slow and repeats deterministic work. |
| Cache cleaned waveform only | Saves some work but still repeats mel extraction. |
| Cache augmented features | Not chosen because augmentation should be random each epoch. Cache clean base features, then augment on load. |

Improvement:

- Use `pathlib` and relative paths instead of string replacement from `train_data` to `processed_features`.
- Add a preprocessing version file so old caches are invalidated when parameters change.

## CNN Architecture Decisions

Code: `model/cnn.py`

Current architecture:

```text
Input: (batch, 1, 128, 281)

ConvBlock x 4:
  Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d

Filters:
  32 -> 64 -> 128 -> 256

AdaptiveAvgPool2d:
  output size (4, 4)

Classifier:
  Flatten -> Dropout(0.5) -> Linear(4096, 512) -> ReLU -> Dropout(0.5) -> Linear(512, 6)
```

Parameter count from the project venv: 2,489,062 trainable parameters.

### Why a CNN?

The log-mel spectrogram is a 2D grid. Nearby time-frequency bins are related, and useful patterns may appear as local bands, bursts, or textures. A CNN is designed for this kind of structured input.

Bishop and Bishop explain that CNNs use locality, hierarchy, equivariance, invariance, and parameter sharing to reduce data requirements and improve generalization on grid-like inputs (Chapter 10, pp. 287-290).

### Why Conv2d instead of fully connected layers?

| Option | Decision |
|---|---|
| Fully connected MLP over flattened spectrogram | Not chosen because it ignores local structure and uses many more parameters. |
| 2D CNN | Chosen because it shares filters across the spectrogram and learns local patterns. |
| 1D CNN over waveform | Possible, but needs the model to learn spectral decomposition from scratch. |
| RNN/LSTM over time frames | Could model temporal order, but is slower and less natural for local 2D frequency patterns. |
| Transformer | Future option, but heavier and less necessary for a strong first baseline. |

Book reference: Bishop and Bishop discuss parameter sharing in Section 9.4 (p. 270) and CNN filters in Section 10.2 (p. 290).

### Why 3x3 kernels?

Chosen:

- `CNN_KERNEL_SIZE = 3`
- `CNN_PADDING = 1`

Why this is good:

- 3x3 filters are a standard local pattern detector.
- Padding preserves spatial size before pooling.
- Stacking multiple 3x3 layers increases the effective receptive field while keeping parameter count manageable.

Options considered:

| Option | Decision |
|---|---|
| 5x5 or 7x7 kernels | Larger context, but more parameters and more risk of overfitting. |
| 1x1 kernels | Useful for channel mixing, but not enough alone to capture local time-frequency texture. |
| Dilated convolutions | Future option for wider context without much extra compute. |

### Why BatchNorm?

Chosen:

- `BatchNorm2d` after each convolution.
- Convolution bias disabled because BatchNorm has its own shift/scale behavior.

Why this is good:

- Stabilizes intermediate activation distributions.
- Helps train deeper CNN stacks.
- Works well with batch sizes like `64`.

Options considered:

| Option | Decision |
|---|---|
| No normalization | Less stable training. |
| LayerNorm | Useful for transformers and small batches, but BatchNorm is standard for CNNs. |
| GroupNorm | Good if batch size becomes small. |

Book reference: Bishop and Bishop discuss BatchNorm in Section 7.4.2, p. 227.

### Why ReLU?

Chosen:

- `ReLU(inplace=True)` in blocks and classifier.

Why this is good:

- Simple, fast, and widely used.
- Helps avoid saturation problems common with sigmoid/tanh.

Options considered:

| Option | Decision |
|---|---|
| Sigmoid/tanh | Not chosen because they saturate more easily. |
| Leaky ReLU | Good future alternative if dead ReLU behavior appears. |
| GELU/SiLU | Often strong, but not necessary for a simple CNN baseline. |

Book reference: Bishop and Bishop discuss ReLU in Section 6.2, p. 185.

### Why MaxPool and AdaptiveAvgPool?

Chosen:

- `MaxPool2d(2)` inside each block.
- `AdaptiveAvgPool2d((4, 4))` before classifier.

Why this is good:

- MaxPool reduces spatial size and keeps strong local activations.
- Adaptive pooling gives the classifier a fixed input size even if earlier dimensions change.
- It reduces the number of classifier parameters.

Options considered:

| Option | Decision |
|---|---|
| No pooling | Higher compute and more overfitting risk. |
| Strided convolution | Good alternative; learnable downsampling, but slightly more complex. |
| Global average pooling to `(1, 1)` | Very compact, but may discard useful spatial layout. |
| Larger adaptive output | More detail but more classifier parameters. |

Book reference: Bishop and Bishop discuss pooling in CNNs in Section 10.2.6, p. 296.

### Why Dropout?

Chosen:

- `Dropout(p=0.5)` in the classifier.

Why this is good:

- The model has about 2.49 million parameters.
- The abnormal classes are smaller.
- Dropout reduces co-adaptation and overfitting in the dense classifier.

Options considered:

| Option | Decision |
|---|---|
| No Dropout | More overfitting risk. |
| Lower Dropout such as 0.2 | Could improve if current dropout underfits. Tune with validation. |
| Dropout in convolution blocks | Possible, but spatial dropout is usually better than ordinary dropout there. |

Book reference: Bishop and Bishop discuss Dropout as cheap and effective regularization in Section 9.6.1, p. 279.

## Training Decisions

Code: `training/trainer.py`

Chosen:

- Loss: `CrossEntropyLoss`.
- Optimizer: `AdamW`.
- Learning rate: `1e-3`.
- Weight decay: `1e-4`.
- Scheduler: `CosineAnnealingLR`.
- Early stopping patience: `10`.
- Batch size: `64`.

### Why CrossEntropyLoss?

The model outputs six raw logits. `CrossEntropyLoss` combines softmax behavior with the multi-class negative log-likelihood objective.

Options:

| Option | Decision |
|---|---|
| MSE on one-hot labels | Not chosen because it is less appropriate for multi-class classification. |
| Binary losses per class | Not chosen because each sample belongs to exactly one of six classes. |
| Focal loss | Future option for class imbalance if weighted cross-entropy is insufficient. |

Book reference: Bishop and Bishop discuss softmax and multi-class cross-entropy in Chapter 5, especially p. 162.

### Why AdamW and weight decay?

AdamW is a strong default optimizer for neural networks, and weight decay discourages unnecessarily large weights.

Options:

| Option | Decision |
|---|---|
| SGD with momentum | Good but may need more learning-rate tuning. |
| Adam | Good, but AdamW handles weight decay more cleanly. |
| RMSProp | Good alternative, but AdamW is the stronger modern default. |

Book reference: Bishop and Bishop discuss Adam in Section 7.3.3 (p. 223) and weight decay in Section 9.2 (p. 260).

### Why early stopping?

Chosen:

- Save the best validation-loss checkpoint.
- Stop after validation loss fails to improve for `10` epochs.

Why this is good:

- It controls overfitting without needing to know the best epoch in advance.
- It protects the final checkpoint from later over-training.

Book reference: Bishop and Bishop discuss learning curves and early stopping in Section 9.3.1, p. 266.

### Current training issue

In `train.py`, the real training block is commented out. The script currently creates splits/loaders/model and then jumps to final test evaluation by loading `checkpoints/best_model.pth`.

This is acceptable only if the checkpoint already exists and the goal is final evaluation. It is not ideal as the main training entry point.

Improvement:

- Either uncomment Step 4 for normal training, or split the script into explicit modes:
  - `python train.py --mode train`
  - `python train.py --mode evaluate`
  - `python train.py --mode train-and-evaluate`

## Evaluation Decisions

Code: `training/evaluation.py`

Chosen:

- Overall accuracy.
- Per-class accuracy.
- Confusion matrix.

Why this is good:

- Overall accuracy is easy to communicate.
- Per-class accuracy reveals whether abnormal classes are being missed.
- Confusion matrix shows which classes are confused.

What should be added:

| Metric | Why |
|---|---|
| Macro-F1 | Treats each class equally, important under imbalance. |
| Balanced accuracy | Average recall across classes; useful when normal classes dominate. |
| Precision/recall per class | Important if abnormal detection is the real priority. |
| Normal-vs-abnormal aggregate metrics | Useful because the practical question may be fault detection, not only machine identity. |

Book reference: Bishop and Bishop emphasize evaluation on held-out data and generalization rather than only training error (Section 1.2.6, p. 14).

## Inference Decisions

Code: `infer.py`

Chosen:

- Load `best_model.pth`.
- Set `model.eval()`.
- Use `torch.no_grad()`.
- Sort test files by integer filename.
- Disable augmentation.
- Write `results.txt` and `time.txt`.

Why this is good:

- Integer sorting avoids the classic `1, 10, 100, 2` filename bug.
- `eval()` disables Dropout and freezes BatchNorm behavior.
- No augmentation at inference keeps predictions deterministic.

Options considered:

| Option | Decision |
|---|---|
| String filename sorting | Not chosen because it produces wrong submission order. |
| Augment inference data | Not chosen because it changes the sample and makes outputs random. Test-time augmentation could be considered later but is slower. |
| Process files one by one | Chosen for simplicity and timing clarity. Batch inference can improve speed later. |

## Existing Good Decisions

### Central configuration

`config.py` is a good idea because shared constants should live in one place.

Improvement:

- Remove duplicate constants in `features/mel_spectrogram.py` and `features/augmentation.py`, because they currently override values imported from `config.py`.
- Update stale comments and README values. For example, README mentions some old values such as fixed frames `256` and batch size `32`, while the current code uses `281` and `64`.

### Offline deterministic cache

This is good because expensive deterministic work is done once. The right pattern is:

```text
cache clean base feature -> apply random augmentation during training load
```

The current cache is correct. The missing piece is random augmentation at load time.

### Stratified split

Good because the dataset is imbalanced. A non-stratified split could accidentally distort the already-small abnormal classes.

### CNN as a first model

Good because it matches the spectrogram structure and the book's CNN inductive-bias argument.

### Integer inference sorting

Good because it protects the submission format.

## Main Improvement Options

### Option A: Fix the training data path and imbalance handling

Tasks:

- Apply `features.apply_augmentation()` inside `MachineDataset` when `augment=True`.
- Add `WeightedRandomSampler` or class-weighted `CrossEntropyLoss`.
- Add macro-F1 and balanced accuracy.
- Restore an explicit training path in `train.py`.

Decision: choose this first.

Why:

- It addresses a confirmed code gap.
- It addresses the strongest dataset problem: imbalance.
- It uses existing code instead of adding new architecture complexity.
- It is low-risk and directly tied to generalization.

### Option B: Try a deeper CNN or ResNet-style model

Decision: not first.

Why:

- Bigger models can overfit abnormal classes if the data pipeline is not fixed first.
- The current CNN is already reasonable and has about 2.49 million parameters.
- Bishop and Bishop emphasize regularization and inductive bias; model size alone is not the first lever.

### Option C: Keep 48 kHz or increase target sample rate

Decision: not first.

Why:

- It increases compute and feature size.
- We need evidence that useful fault information above 8 kHz is being lost.
- A fair ablation can be run later: 16 kHz vs 24 kHz vs 48 kHz.

### Option D: Add aggressive augmentations

Decision: not first.

Why:

- Pitch/time changes can alter physical fault frequencies.
- Existing simple augmentations are safer and already aligned with spectrogram robustness.
- First make existing augmentation actually run.

### Option E: Remove denoising or change trimming settings

Decision: run as an ablation, not as the first code change.

Why:

- Noise reduction can help generalization but can also erase weak fault evidence.
- The correct answer should come from validation comparisons:
  - no denoise vs denoise
  - `top_db=20` vs `40` vs `60`

### Option F: Use transformer or pretrained audio model

Decision: not first.

Why:

- Higher complexity and more dependency risk.
- Harder to explain and tune.
- CNN baseline should be made correct and strong before replacing it.

## Recommended First Implementation Plan

1. Fix `MachineDataset` augmentation for cached features.
   - Load `.npy`.
   - Convert `(1, 128, 281)` to `(128, 281)` temporarily.
   - Apply `apply_augmentation` if `augment=True`.
   - Restore channel dimension.

2. Add class imbalance handling.
   - Start with class-weighted `CrossEntropyLoss`.
   - Alternative: `WeightedRandomSampler`.
   - Compare both on validation macro-F1.

3. Add metrics.
   - Macro-F1.
   - Balanced accuracy.
   - Precision/recall per class.

4. Clean training modes.
   - Make `train.py` actually train by default, or add explicit CLI modes.

5. Run ablations after the first fix.
   - Baseline current cached features.
   - Augmentation on/off.
   - Weighted loss on/off.
   - Noise reduction on/off.
   - Target sample rate 16 kHz vs 24 kHz if compute allows.

## Risk Register

| Risk | Impact | Fix |
|---|---|---|
| Training augmentation is currently a no-op for cached features | Lower generalization than intended | Apply `apply_augmentation` in `MachineDataset` |
| Heavy class imbalance | High accuracy can hide poor abnormal recall | Weighted loss/sampler and macro-F1 |
| `train.py` skips training | Reproducibility problem | Restore training mode |
| Duplicate constants outside `config.py` | Config edits may not apply | Remove local overrides |
| String path replacement for cache paths | Brittle on path changes | Use `Path.relative_to()` and `with_suffix()` |
| Hard-coded paths in test scripts | Tests fail outside one machine | Use `config.PROJECT_ROOT` |
| Docker copies full project, likely including train data/cache | Large image and possible submission risk | Add `.dockerignore` |
| Denoising may erase weak fault signals | Possible accuracy loss | Validate with ablation |
| Random split may leak similar sequential samples | Over-optimistic scores | Consider group-aware split by recording/session if metadata exists |

## Final Position

The current project direction is sound: clean audio, convert to log-mel spectrograms, train a CNN, evaluate on held-out data, and keep inference deterministic. That matches Bishop and Bishop's main deep-learning principles: useful representations, normalized inputs, inductive bias, regularization, and disciplined validation.

The first priority is not to replace the whole model. The first priority is to make the existing intended design truly active: training augmentation should run, imbalance should be handled, metrics should expose abnormal-class performance, and `train.py` should clearly train or evaluate depending on the requested mode. After that, the project can fairly compare stronger options such as 24 kHz audio, deeper CNNs, stronger augmentation, or pretrained audio models.
