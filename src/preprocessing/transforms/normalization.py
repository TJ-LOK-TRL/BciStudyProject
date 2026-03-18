import numpy as np
from sklearn.preprocessing import StandardScaler
from src.preprocessing.base_preprocessor import BasePreprocessor


class ZScoreNormalizer(BasePreprocessor):
    """Z-score normalization per trial."""

    def fit(self, X: np.ndarray) -> 'ZScoreNormalizer':
        return self   # stateless — no fitting needed

    def transform(self, X: np.ndarray) -> np.ndarray:
        mean = X.mean(axis=-1, keepdims=True)
        std = X.std(axis=-1, keepdims=True) + 1e-8
        return (X - mean) / std

    def __repr__(self) -> str:
        return 'ZScoreNormalizer()'
    

class ChannelScaler:
    """
    StandardScaler per channel, fitted on training data.
    Replicates the paper's preprocessing exactly:
        - fit on X_train, channel by channel
        - transform both X_train and X_test using train statistics
 
    This is NOT stateless — must be fitted before transforming.
 
    Args:
        X: shape (n_trials, n_channels, n_times)
 
    Example:
        >>> scaler = ChannelScaler()
        >>> X_train = scaler.fit_transform(X_train)
        >>> X_test  = scaler.transform(X_test)
    """
 
    def __init__(self):
        self._scalers: list[StandardScaler] = []
        self._is_fitted: bool = False
 
    def fit(self, X: np.ndarray) -> 'ChannelScaler':
        """
        Fit one StandardScaler per channel using X_train.
 
        Args:
            X: shape (n_trials, n_channels, n_times)
        """
        n_channels = X.shape[1]
        self._scalers = []
        for j in range(n_channels):
            scaler = StandardScaler()
            scaler.fit(X[:, j, :])   # fit across trials for channel j
            self._scalers.append(scaler)
        self._is_fitted = True
        return self
 
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X using the fitted scalers.
        Can be called on both train and test sets after fit().
 
        Args:
            X: shape (n_trials, n_channels, n_times)
 
        Returns:
            X_scaled: shape (n_trials, n_channels, n_times)
        """
        if not self._is_fitted:
            raise RuntimeError('ChannelScaler is not fitted yet, call fit() first.')
        X_out = X.copy()
        for j, scaler in enumerate(self._scalers):
            X_out[:, j, :] = scaler.transform(X[:, j, :])
        return X_out
 
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
 
    def __repr__(self) -> str:
        status = f'fitted={len(self._scalers)} channels' if self._is_fitted else 'not fitted'
        return f'ChannelScaler({status})'


class MinMaxNormalizer(BasePreprocessor):
    """Min-max normalization per trial."""

    def fit(self, X: np.ndarray) -> 'MinMaxNormalizer':
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        min_val = X.min(axis=-1, keepdims=True)
        max_val = X.max(axis=-1, keepdims=True)
        return (X - min_val) / (max_val - min_val + 1e-8)
    
    def __repr__(self) -> str:
        return 'MinMaxNormalizer()'