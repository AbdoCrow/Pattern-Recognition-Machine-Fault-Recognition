"""
===============================================================================
model/cnn.py — CNN Architecture for Machine Sound Classification
===============================================================================

OWNER: EL sir
"""

import torch
import torch.nn as nn
from config import (
    CNN_INPUT_CHANNELS,
    NUM_CLASSES,
    CNN_FILTERS,
    CNN_KERNEL_SIZE,
    CNN_PADDING,
    CNN_POOL_SIZE,
    ADAPTIVE_POOL_OUTPUT,
)

class ConvBlock(nn.Module):
    """
    A single convolutional block: Conv2d → BatchNorm → ReLU → MaxPool2d.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=CNN_KERNEL_SIZE,
            padding=CNN_PADDING,
            bias=False  # Bias is redundant when followed by BatchNorm
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(CNN_POOL_SIZE)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class MachineSoundCNN(nn.Module):
    """
    Custom CNN for 6-class machine sound classification from mel spectrograms.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super(MachineSoundCNN, self).__init__()

        # ---------------------------------------------------------------------
        # 1. Feature Extractor (Dynamic Conv Blocks)
        # ---------------------------------------------------------------------
        layers = []
        in_channels = CNN_INPUT_CHANNELS
        
        # Dynamically build blocks based on config.py (e.g., 32 -> 64 -> 128 -> 256)
        for out_channels in CNN_FILTERS:
            layers.append(ConvBlock(in_channels, out_channels))
            in_channels = out_channels  # The output of this layer is the input to the next
            
        self.features = nn.Sequential(*layers)
        
        # ---------------------------------------------------------------------
        # 2. Adaptive Pooling
        # ---------------------------------------------------------------------
        # This squashes whatever time dimension is left into a fixed (4, 4) grid
        self.adaptive_pool = nn.AdaptiveAvgPool2d(ADAPTIVE_POOL_OUTPUT)

        # ---------------------------------------------------------------------
        # 3. Classifier Head (Fully Connected)
        # ---------------------------------------------------------------------
        # Calculate flattened size: Last filter size (256) * height (4) * width (4) = 4096
        flattened_size = CNN_FILTERS[-1] * ADAPTIVE_POOL_OUTPUT[0] * ADAPTIVE_POOL_OUTPUT[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),                   # Regularization: Prevent overfitting
            nn.Linear(flattened_size, 512),      # Hidden layer to compress features
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),                   # Regularization
            nn.Linear(512, num_classes)          # Output raw logits (6 classes)
        )

    def forward(self, x):
        # Extract visual features from the spectrogram
        x = self.features(x)
        # Pool them to a fixed mathematical size
        x = self.adaptive_pool(x)
        # Make the final classification
        x = self.classifier(x)
        return x


# =============================================================================
# SHAPE SANITY CHECK
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CNN Architecture Shape Sanity Check")
    print("=" * 60)

    model = MachineSoundCNN()

    # Create a dummy batch based on JSON's exact tensor output (batch=4, ch=1, mels=128, time=281)
    dummy_input = torch.randn(4, 1, 128, 281)
    print(f"\nInput shape:  {dummy_input.shape}")

    # Trace through the architecture
    x = dummy_input
    for i, block in enumerate(model.features):
        x = block(x)
        print(f"After Conv Block {i+1} ({CNN_FILTERS[i]} filters): {x.shape}")

    x = model.adaptive_pool(x)
    print(f"After AdaptivePool:  {x.shape}")

    x = model.classifier(x)
    print(f"After Classifier:    {x.shape}")

    print(f"\n✓ Output shape is correct: {x.shape} (batch=4, classes={NUM_CLASSES})")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print("=" * 60)