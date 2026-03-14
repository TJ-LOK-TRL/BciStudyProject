import numpy as np
from src.preprocessing.base_preprocessor import BasePreprocessor


class BandpassFilter(BasePreprocessor):
    """Bandpass filter using MNE IIR filter."""

    def __init__(self, sfreq: float, l_freq: float, h_freq: float):
        self.sfreq = sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq

    def fit(self, X: np.ndarray) -> 'BandpassFilter':
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        from mne.filter import filter_data
        return filter_data(
            X.astype(np.float64),
            sfreq=self.sfreq,
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            method='iir',
            verbose=False,
        ).astype(np.float32)

    def __repr__(self) -> str:
        return f'BandpassFilter(sfreq={self.sfreq}, l_freq={self.l_freq}, h_freq={self.h_freq})'