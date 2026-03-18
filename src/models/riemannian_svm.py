from typing import Optional, List
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib
from pathlib import Path

from src.models.base_model import BaseModel

class RiemannianSVM(BaseModel):
    """
    Covariance + Tangent Space + SVM classifier.

    Expects X with shape (n_trials, n_bands, n_channels, n_times).
    Filter bank preprocessing must be applied upstream (FilterBankTransform).

    Responsibilities:
        - Fit/transform Covariances and TangentSpace per band
        - Concatenate tangent vectors across bands
        - GridSearchCV over SelectKBest + SVC pipeline
    """

    def __init__(
        self,
        n_jobs: int = -1,
        cv: int = 5,
        param_grid: Optional[dict] = None,
        band_mode: bool = True, # TODO: Improve in the future the overall struct to be more modular
    ):
        super().__init__()
        self.n_jobs = n_jobs
        self.cv = cv
        self.param_grid = param_grid or {
            'select__k': list(range(40, 150, 2)),
            'svc__kernel': ['rbf'],
            'svc__C': [0.01, 0.1, 0.5, 1, 2, 5, 10],
            'svc__gamma': ['scale'],
        }
        self.model: Optional[GridSearchCV] = None
        self._cov_ts_list: Optional[List] = None  # fitted (Covariances, TangentSpace) per band
        self.band_mode = band_mode

    def _extract_features(
        self,
        X: np.ndarray,
        fit: bool = False,
    ) -> np.ndarray:
        """
        Compute covariance matrices and project to tangent space per band.

        Args:
            X:   shape (n_trials, n_bands, n_channels, n_times)
            fit: if True, fit Covariances and TangentSpace objects;
                 if False, use already-fitted objects from self._cov_ts_list

        Returns:
            features: shape (n_trials, n_bands * tangent_dim)
        """
        if self.band_mode:
            n_trials, n_bands, n_channels, n_times = X.shape
            features = []
            cov_ts_list = []

            for i in range(n_bands):
                X_band = X[:, i, :, :].astype(np.float64)  # (n_trials, n_channels, n_times)

                cov = Covariances('oas')
                ts = TangentSpace()

                if fit:
                    feat = ts.fit_transform(cov.fit_transform(X_band))
                    cov_ts_list.append((cov, ts))
                else:
                    cov_fitted, ts_fitted = self._cov_ts_list[i]
                    feat = ts_fitted.transform(cov_fitted.transform(X_band))

                features.append(feat)

            if fit:
                self._cov_ts_list = cov_ts_list

            return np.concatenate(features, axis=1)  # (n_trials, n_bands * tangent_dim)
        else:
            # single band — (n_trials, n_channels, n_times)
            X_64 = X.astype(np.float64)
            cov = Covariances('oas')
            ts = TangentSpace()

            if fit:
                feat = ts.fit_transform(cov.fit_transform(X_64))
                self._cov_ts_list = [(cov, ts)]
            else:
                cov_fitted, ts_fitted = self._cov_ts_list[0]
                feat = ts_fitted.transform(cov_fitted.transform(X_64))

            return feat

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Fit the full pipeline.

        Args:
            X: shape (n_trials, n_bands, n_channels, n_times)
            y: shape (n_trials,)
        """
        X_feat = self._extract_features(X, fit=True)

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('select', SelectKBest(f_classif)),
            ('svc', SVC()),
        ])

        self.model = GridSearchCV(
            pipe,
            self.param_grid,
            cv=self.cv,
            n_jobs=self.n_jobs,
            verbose=0,
        )
        self.model.fit(X_feat, y)
        self.is_fitted = True
        print(f'    Best params: {self.model.best_params_}')

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: shape (n_trials, n_bands, n_channels, n_times)

        Returns:
            y_pred: shape (n_trials,)
        """
        if not self.is_fitted:
            raise RuntimeError('Model is not fitted yet, call fit() first')
        X_feat = self._extract_features(X, fit=False)
        return self.model.predict(X_feat)

    def clone(self) -> 'RiemannianSVM':
        return RiemannianSVM(
            n_jobs=self.n_jobs,
            cv=self.cv,
            param_grid=self.param_grid,
            band_mode=self.band_mode
        )

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'cov_ts_list': self._cov_ts_list,
            'hyperparams': {
                'n_jobs': self.n_jobs,
                'cv': self.cv,
                'param_grid': self.param_grid,
                'band_mode': self.band_mode,
            }
        }, path)
        print(f'  Model saved to {path}')

    @classmethod
    def load(cls, path: str) -> 'RiemannianSVM':
        ckpt = joblib.load(path)
        model = cls(**ckpt['hyperparams'])
        model.model = ckpt['model']
        model._cov_ts_list = ckpt['cov_ts_list']
        model.is_fitted = True
        print(f'  Model loaded from {path}')
        return model

    def __repr__(self) -> str:
        n_bands = len(self._cov_ts_list) if self._cov_ts_list else '?'
        return (
            f'RiemannianSVM('
            f'n_bands={n_bands}, '
            f'fitted={self.is_fitted})'
        )