import gradio as gr
import torch
import librosa
import numpy as np
import noisereduce as nr  # <-- CRITICAL ADDITION
from model import MachineSoundCNN  # Make sure this matches your model class name!

# 1. Define your exact classes
CLASSES = [
    "Machine 1 Normal", "Machine 1 Abnormal",
    "Machine 2 Normal", "Machine 2 Abnormal",
    "Machine 3 Normal", "Machine 3 Abnormal"
]

# 2. Load the Model (Forcing CPU for the Hugging Face Free Tier)
device = torch.device('cpu')
model = MachineSoundCNN() # Initialize EL sir's architecture
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval() # Set to evaluation mode (freezes BatchNorm/Dropout)

# 3. Preprocessing & Inference Function
def predict_machine_sound(audio_path):
    if audio_path is None:
        return "Please upload an audio file."

    # Load audio at exactly 16kHz
    y, sr = librosa.load(audio_path, sr=16000)

    # --- CRITICAL FIX: NOISE REDUCTION ---
    # Replicating your specific 0.5-second noise profile logic
    profile_length = int(0.5 * sr)
    if len(y) > profile_length:
        noise_profile = y[:profile_length]
        y_clean = nr.reduce_noise(y=y, sr=sr, y_noise=noise_profile, prop_decrease=0.8)
    else:
        y_clean = nr.reduce_noise(y=y, sr=sr, y_noise=y, prop_decrease=0.5)
        
    # Clip signal to [-1.0, 1.0] just like your training pipeline
    y_clean = np.clip(y_clean, -1.0, 1.0)
    # -------------------------------------

    # Generate Mel Spectrogram (128 bins) using the CLEAN audio
    mel_spec = librosa.feature.melspectrogram(y=y_clean, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Reshape for the CNN: (Batch, Channel, Height, Time) -> (1, 1, 128, time_steps)
    input_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Forward Pass through the CNN
    with torch.no_grad():
        outputs = model(input_tensor)
        # Convert raw output numbers into percentages (0.0 to 1.0)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Format output for the Gradio UI: a dictionary of {Class Name: Probability}
    result = {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}
    return result

# 4. Build the Web Interface
interface = gr.Interface(
    fn=predict_machine_sound,
    inputs=gr.Audio(type="filepath", label="Upload Machine Audio (.wav)"),
    outputs=gr.Label(num_top_classes=6, label="CNN Prediction Confidence"),
    title="Industrial Machine Sound Anomaly Detector",
    description="Upload an audio clip of an industrial machine. The Custom CNN will analyze the Mel Spectrogram and predict if it is operating normally or failing.",
    allow_flagging="never"
)

# 5. Launch the App
if __name__ == "__main__":
    interface.launch()