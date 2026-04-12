"""
===============================================================================
model/__init__.py — CNN Model Package
===============================================================================

OWNER: EL sir

This package contains the CNN architecture for machine sound classification.
The model takes mel spectrograms as input and outputs class probabilities.

Architecture: Conv2D blocks → AdaptiveAvgPool2d → FC → Softmax (6 classes)
===============================================================================
"""

from model.cnn import MachineSoundCNN
