"""
===============================================================================
training/__init__.py — Training and Evaluation Pipeline
===============================================================================

OWNER: Osama

This package contains:
1. Training loop with optimizer, scheduler, early stopping, checkpointing
2. Evaluation / performance analysis (confusion matrix, per-class accuracy)
===============================================================================
"""

from training.trainer import Trainer
from training.evaluation import evaluate_model, generate_confusion_matrix
