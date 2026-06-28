"""Training subpackage: expert trainer and QA generator."""
from dmoe.training.expert_trainer import ExpertTrainer
from dmoe.training.qa_generator import OpenRouterTeacher, DummyTeacher, create_teacher

__all__ = ["ExpertTrainer", "OpenRouterTeacher", "DummyTeacher", "create_teacher"]
