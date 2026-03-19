from abc import abstractmethod


class IHyperparametrizable:
    """
    Interface for models with serialisable hyperparameters.
    All hyperparams must be JSON-serialisable primitives (int, float, str, list, dict).
    Used by NNWrapper and future save/load mechanisms (yaml, pt, etc.).
    """

    @abstractmethod
    def get_hyperparams(self) -> dict:
        """
        Return constructor kwargs as a plain dict.
        Must contain everything needed to reconstruct the model via from_hyperparams().

        Example:
            return {'n_channels': 22, 'n_classes': 4, 'dropout': 0.5}
        """
        pass

    @classmethod
    def from_hyperparams(cls, hyperparams: dict) -> 'IHyperparametrizable':
        """
        Reconstruct instance from saved hyperparams.
        Default calls cls(**hyperparams) — override if construction is more complex.
        """
        return cls(**hyperparams)