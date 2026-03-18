import numpy as np
from typing import Literal, List, Tuple
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
    
class FilterBankTransform:
    """
    Applies a bank of bandpass filters in parallel to an EEG signal.

    Each band produces a filtered copy of the signal. The results are
    stacked along a new axis, giving one 'view' of the signal per band.

    Args:
        sfreq:       Sampling frequency in Hz.
        order:       Filter order (higher = sharper cutoff, more ringing).
        bands:       List of (fmin, fmax) tuples in Hz.
                     Defaults to 6 bands covering 8–32 Hz (alpha + beta).
        filter_type: Type of IIR filter to use.
                     - 'butter'  : Butterworth — maximally flat, no ripple (default)
                     - 'cheby1'  : Chebyshev I — sharper cutoff, ripple in passband
                     - 'cheby2'  : Chebyshev II — sharper cutoff, ripple in stopband
                     - 'ellip'   : Elliptic — sharpest cutoff, ripple in both bands
                     - 'bessel'  : Bessel — best phase response, gentle rolloff
        rp:          Max ripple in passband (dB). Only used by cheby1 and ellip.
        rs:          Min attenuation in stopband (dB). Only used by cheby2 and ellip.

    Input shape:  (n_trials, n_channels, n_times)
    Output shape: (n_trials, n_bands, n_channels, n_times)

    This transform is stateless — fit() is a no-op and exists only for
    sklearn Pipeline compatibility.

    Example:
        >>> fb = FilterBankTransform(sfreq=250.0, filter_type='butter')
        >>> X_filtered = fb.fit_transform(X)   # X: (720, 22, 1000)
        >>> X_filtered.shape                   # (720, 6, 22, 1000)
    """

    def __init__(
        self,
        bands: List[Tuple[float, float]],
        sfreq: float,
        order: int = 5,
        filter_type: Literal['butter', 'cheby1', 'cheby2', 'ellip', 'bessel'] = 'butter',
        rp: float = 5.0,
        rs: float = 40.0,
    ):
        self.sfreq = sfreq
        self.order = order
        self.bands = bands
        self.filter_type = filter_type
        self.rp = rp
        self.rs = rs

    def _build_sos(self, fmin: float, fmax: float):
        """Build second-order sections for one band."""
        from scipy.signal import butter, cheby1, cheby2, ellip, bessel

        Wn = [fmin, fmax]
        kwargs = dict(N=self.order, Wn=Wn, btype='band', fs=self.sfreq, output='sos')

        if self.filter_type == 'butter':
            return butter(**kwargs)
        elif self.filter_type == 'cheby1':
            return cheby1(rp=self.rp, **kwargs)
        elif self.filter_type == 'cheby2':
            return cheby2(rs=self.rs, **kwargs)
        elif self.filter_type == 'ellip':
            return ellip(rp=self.rp, rs=self.rs, **kwargs)
        elif self.filter_type == 'bessel':
            return bessel(**kwargs)
        else:
            raise ValueError(
                f'Unknown filter_type \'{self.filter_type}\'. '
                f'Choose from: butter, cheby1, cheby2, ellip, bessel.'
            )

    def fit(self, X: np.ndarray, y=None) -> 'FilterBankTransform':
        """No-op. Filter bank has no learnable parameters."""
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply all bandpass filters to X.

        Args:
            X: shape (n_trials, n_channels, n_times)

        Returns:
            shape (n_trials, n_bands, n_channels, n_times)
        """
        from scipy.signal import sosfilt

        bands_out = []
        for fmin, fmax in self.bands:
            sos = self._build_sos(fmin, fmax)
            bands_out.append(sosfilt(sos, X).astype(np.float64))

        return np.stack(bands_out, axis=1)

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def __repr__(self) -> str:
        return (
            f'FilterBankTransform('
            f'sfreq={self.sfreq}, '
            f'bands={len(self.bands)}, '
            f'filter_type={self.filter_type!r}, '
            f'order={self.order})'
        )