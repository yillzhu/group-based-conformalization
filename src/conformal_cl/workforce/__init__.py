"""Cross-trained workforce-planning experiment for conformal optimization."""

from .experiment import ExperimentConfig, run_experiment
from .study import run_study, summarize_runs

__all__ = ["ExperimentConfig", "run_experiment", "run_study", "summarize_runs"]
