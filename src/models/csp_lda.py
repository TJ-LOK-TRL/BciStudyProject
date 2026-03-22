from typing import Optional
from pathlib import Path
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from mne.decoding import CSP
import joblib

from src.models.core import ITrainableModel


class CSPLDAModel(ITrainableModel):
    """
    CSP + LDA classifier for Motor Imagery EEG.
    CSP extracts spatial filters, LDA classifies the log-variance features.
    """

    def __init__(
        self,
        n_components: int = 4,
        reg: Optional[str] = None,
        log: bool = True,
    ):
        super().__init__()
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self._label_encoder = LabelEncoder()
        self.model: Pipeline = Pipeline([
            ('csp', CSP(n_components=self.n_components, reg=self.reg, log=self.log)),
            ('lda', LinearDiscriminantAnalysis()),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        y_encoded = self._label_encoder.fit_transform(y)
        self.model.fit(X, y_encoded)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError('Model is not fitted yet, call fit() first')
        return self._label_encoder.inverse_transform(self.model.predict(X))

    def clone(self) -> 'CSPLDAModel':
        return CSPLDAModel(
            n_components=self.n_components,
            reg=self.reg,
            log=self.log,
        )

    def get_hyperparams(self) -> dict:
        return {
            'n_components': self.n_components,
            'reg':          self.reg,
            'log':          self.log,
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'pipeline':      self.model,
            'label_encoder': self._label_encoder,
            'hyperparams':   self.get_hyperparams(),
        }, path)
        print(f'  Model saved to {path}')

    @classmethod
    def load(cls, path: str) -> 'CSPLDAModel':
        ckpt = joblib.load(path)
        model = cls(**ckpt['hyperparams'])
        model.model = ckpt['pipeline']
        model._label_encoder = ckpt['label_encoder']
        model.is_fitted = True
        print(f'  Model loaded from {path}')
        return model

    def __repr__(self) -> str:
        return (
            f'CSPLDAModel('
            f'n_components={self.n_components}, '
            f'fitted={self.is_fitted})'
        )