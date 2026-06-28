"""Decoupled Mixture-of-Experts for Parametric Knowledge Injection.

Based on: "Decoupled Mixture-of-Experts for Parametric Knowledge Injection"
(https://arxiv.org/abs/2606.14243)

Experiment setup:
- Base models: Llama-3.2-1B-Instruct (paper), Gemma-3-1B-IT (generalization)
- Training: Modal cloud GPU (no local GPU required)
- Corpora: World Cup 2026, arXiv 2026 Top Papers
- Benchmarks: CWQ, HotpotQA, Quasar-T, StrategyQA
"""

__version__ = "0.1.0"

from dmoe.router import DMoERouter, TokenUncertainty
from dmoe.models import ExpertManager, Expert, create_base_model
from dmoe.inference import DMoEGenerator
from dmoe.training import ExpertTrainer
from dmoe.database import Database

__all__ = [
    "DMoERouter",
    "TokenUncertainty",
    "ExpertManager",
    "Expert",
    "DMoEGenerator",
    "ExpertTrainer",
    "create_base_model",
    "Database",
]
