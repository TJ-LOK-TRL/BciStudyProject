from abc import ABC, abstractmethod
from src.models.core import IFittable, IPredictable, IHyperparametrizable

class ITrainableModel(IFittable, IPredictable, IHyperparametrizable, ABC):

    @abstractmethod
    def clone(self) -> 'ITrainableModel':
        """Return a new unfitted instance with same configuration."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'ITrainableModel':
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__