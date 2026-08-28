"""Training and evaluation orchestrator package for Multimodal Document Understanding."""

from .train_baseline import run_training_pipeline, train_baselines_for_dataset
from .evaluate_baseline import evaluate_baselines_pipeline

__all__ = ["run_training_pipeline", "train_baselines_for_dataset", "evaluate_baselines_pipeline"]
