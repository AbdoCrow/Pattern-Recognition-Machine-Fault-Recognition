import os
import numpy as np
import torch
from tqdm import tqdm
import config

# The actual functions used in your dataset.py
from preprocessing import preprocess_audio
from features import audio_to_tensor

def preprocess_dataset():
    # Save to a new folder so we don't mess up original data
    PROCESSED_DIR = os.path.join(config.PROJECT_ROOT, "processed_features")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # We use the raw training data directory
    for root, dirs, files in os.walk(config.TRAIN_DATA_DIR):
        for file in tqdm(files, desc="Cleaning & Converting Audio"):
            if file.endswith('.wav'):
                input_path = os.path.join(root, file)
                
                # Setup output path (.wav -> .npy)
                rel_path = os.path.relpath(input_path, config.TRAIN_DATA_DIR)
                output_path = os.path.join(PROCESSED_DIR, rel_path).replace('.wav', '.npy')
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                if os.path.exists(output_path): continue

                try:
                    # STEP 1: Denoise, Trim, Resample (The slow part!)
                    audio, sr = preprocess_audio(input_path)

                    # STEP 2: Mel Spectrogram & Padding (The math part!)
                    # IMPORTANT: We save WITHOUT augmentation so the base feature is clean
                    tensor = audio_to_tensor(audio, sr, augment=False)

                    # Convert to numpy and save
                    if torch.is_tensor(tensor):
                        data = tensor.cpu().numpy()
                    else:
                        data = tensor
                    
                    np.save(output_path, data)
                except Exception as e:
                    print(f"Skipping {file} due to error: {e}")

if __name__ == "__main__":
    preprocess_dataset()