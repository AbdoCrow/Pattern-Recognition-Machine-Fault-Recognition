"""
===============================================================================
data_pipeline/__init__.py — Data Loading Pipeline
===============================================================================

OWNER: JSON

This package handles:
1. Stratified train/val/test splitting of the dataset
2. PyTorch Dataset class for loading and processing audio files
3. DataLoader creation for training, validation, and testing

This is the CENTRAL INTEGRATION POINT — it calls EL sir's preprocessing
and sala7's feature extraction, wiring everyone's code together.
===============================================================================
"""

from data_pipeline.dataset import MachineDataset, InferenceDataset, create_data_loaders
from data_pipeline.splits import create_stratified_splits
