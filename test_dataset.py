import time
from data_pipeline.dataset import MachineDataset
import matplotlib.pyplot as plt

# We will test the dataset by feeding it a single file and a mock label
test_files = [r"C:\pattern\project\train_data\Machine2_normal\1.wav"]
test_labels = [2]  # The integer label for Machine2_normal

print("Testing full PyTorch MachineDataset integration...")

try:
    # Initialize the dataset with augmentation ON to test sala7's SpecAugment
    dataset = MachineDataset(file_paths=test_files, labels=test_labels, augment=True)
    
    start_time = time.time()
    
    # This single line triggers __getitem__, which runs EL sir's, JSON's, and sala7's code
    tensor, label = dataset[0]  
    
    elapsed = time.time() - start_time
    
    print("\nDATASET SUCCESS!")
    print(f"Final Tensor Shape: {tensor.shape}  <-- (Should be [1, 128, 281])")
    print(f"Output Label: {label}")
    print(f"PyTorch Data Type: {tensor.dtype}")
    print(f"Total Pipeline Time (Preproc + Features + Augment): {elapsed:.3f} seconds")

    # Remove the fake [1] channel dimension so we just have [128, 281] for drawing
    image_data = tensor.squeeze(0).numpy()

    plt.figure(figsize=(10, 4))
    # used the 'magma' 
    plt.imshow(image_data, origin='lower', aspect='auto', cmap='magma')
    plt.title(f"Augmented Mel Spectrogram (Label: {label})")
    plt.ylabel("Mel Frequency Bins (128)")
    plt.xlabel("Time Frames (281)")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"\nDATASET FAILED: {e}")



