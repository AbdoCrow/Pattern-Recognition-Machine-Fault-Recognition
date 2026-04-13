import time
from preprocessing import preprocess_audio

# Pick one of your known files to test
test_file = r"C:\pattern\project\train_data\Machine2_normal\1.wav"

print(f"Testing pipeline on: {test_file}")

start_time = time.time()
try:
    # This will trigger the entire __init__.py chain
    clean_audio, sr = preprocess_audio(test_file)
    
    elapsed = time.time() - start_time
    
    print("\nPIPELINE SUCCESS!")
    print(f"Final Sampling Rate: {sr} Hz")
    print(f"Final Array Shape: {clean_audio.shape}")
    print(f"Final Duration: {len(clean_audio) / sr:.2f} seconds")
    print(f"Max Amplitude (should be <= 1.0): {clean_audio.max():.4f}")
    print(f"Processing Time: {elapsed:.3f} seconds")

except Exception as e:
    print(f"\nPIPELINE FAILED: {e}")