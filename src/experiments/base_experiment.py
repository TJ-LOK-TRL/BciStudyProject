from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from dataclasses import dataclass, field
from src.evaluation.results import EvaluationResult


@dataclass
class ExperimentConfig:
    """Human-readable description of the experiment setup."""
    name: str
    dataset: str
    model: str
    preprocessing: str
    evaluation: str
    notes: str = ''


class BaseExperiment(ABC):
    """
    A reproducible experiment = dataset + preprocessing + model + evaluation.
    Each subclass is a complete, self-contained record of how a result was obtained.
    """

    @property
    @abstractmethod
    def config(self) -> ExperimentConfig:
        """Describes this experiment in human-readable form."""
        pass

    @abstractmethod
    def prepare_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess data. Returns X, y, subject_ids."""
        pass

    @abstractmethod
    def build_model(self):
        """Instantiate and return the model."""
        pass

    @abstractmethod
    def run(self) -> EvaluationResult:
        """Run the full experiment and return results."""
        pass

    def __repr__(self) -> str:
        c = self.config
        return f'Experiment({c.name} | {c.dataset} | {c.model} | {c.evaluation})'