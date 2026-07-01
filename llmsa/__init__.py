"""LLM-SA: Large Language Model-Enhanced Survival Analysis.

Public entry points:
    llmsa.data          -- dataset loaders (structured and LLM-augmented)
    llmsa.model         -- the LLM-SA stacking ensemble
    llmsa.baselines     -- structured-covariate survival baselines
    llmsa.eval          -- multi-seed IPCW evaluation harness
"""

from .data import load_dataset, load_dataset_v4, ALL_DATASETS
from .model import LLMSAStacking, TrainConfig

__all__ = [
    "load_dataset",
    "load_dataset_v4",
    "ALL_DATASETS",
    "LLMSAStacking",
    "TrainConfig",
]
