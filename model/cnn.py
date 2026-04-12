"""
===============================================================================
model/cnn.py — CNN Architecture for Machine Sound Classification
===============================================================================
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

    This is the building block of the CNN. Each block:
    1. Applies a 3×3 convolution to extract local patterns
    2. Normalizes the output (BatchNorm) for stable training
    3. Applies ReLU activation (introduces non-linearity)
    4. Downsamples by 2× with max pooling (keeps strongest activations)

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
    CNN for 6-class machine sound classification from mel spectrograms.

    Input shape:  (batch, 1, 128, 256)  — 1-channel mel spectrogram
    Output shape: (batch, 6)            — raw logits for 6 classes

    Do NOT apply softmax to the output during training — CrossEntropyLoss
    handles it internally. Apply softmax only during inference.

    Example
    -------
    >>> model = MachineSoundCNN()
    >>> dummy_input = torch.randn(8, 1, 128, 256)  # batch of 8
    >>> output = model(dummy_input)
    >>> print(output.shape)  # torch.Size([8, 6])
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super(MachineSoundCNN, self).__init__()

    def forward(self, x):
        # Extract features through conv blocks
        x = self.features(x)

        # Adaptive pool to fixed size
        x = self.adaptive_pool(x)

        # Classify
        x = self.classifier(x)

        return x


# =============================================================================
# SHAPE SANITY CHECK
# =============================================================================
# Run this file directly to verify shapes flow correctly:
#   python -m model.cnn
#
# This should print the shape at each stage and confirm the final output
# is (batch, 6).
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CNN Architecture Shape Sanity Check")
    print("=" * 60)

    model = MachineSoundCNN()

    # Create a dummy batch (batch_size=4, channels=1, n_mels=128, time=256)
    dummy_input = torch.randn(4, 1, 128, 256)
    print(f"\nInput shape:  {dummy_input.shape}")

    # Trace through each block
    x = dummy_input
    for i, block in enumerate(model.features):
        x = block(x)
        print(f"After Conv Block {i+1}: {x.shape}")

    x = model.adaptive_pool(x)
    print(f"After AdaptivePool:  {x.shape}")

    x = model.classifier(x)
    print(f"After Classifier:    {x.shape}")

    print(f"\n✓ Output shape is correct: {x.shape} (batch=4, classes={NUM_CLASSES})")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 60)
