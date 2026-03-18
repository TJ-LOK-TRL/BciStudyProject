from abc import abstractmethod
from typing import Optional, Literal, List
import numpy as np
from src.preprocessing.base_preprocessor import BasePreprocessor


# Artifact types supported by MNE's find_bads_* methods
ArtifactType = Literal['eog', 'ecg', 'muscle', 'ref_meg']


class BaseArtifactRemover(BasePreprocessor):
    """Base class for EEG artifact removal methods."""

    @abstractmethod
    def fit(self, X: np.ndarray, **kwargs) -> 'BaseArtifactRemover':
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        pass


class RegressionRemover(BaseArtifactRemover):
    """
    Type 1: Artifact removal via linear regression.
    Regresses artifact channel contributions out of EEG channels.
    Fast and simple — good baseline for artifact removal.

    Args:
        artifact_channel_indices: indices of artifact channels (EOG, ECG, etc.)
        artifact_type: type label for logging purposes only — does not affect computation.
    """

    def __init__(
        self,
        artifact_channel_indices: List[int],
        artifact_type: ArtifactType = 'eog',
    ):
        self.artifact_channel_indices = artifact_channel_indices
        self.artifact_type = artifact_type
        self._eeg_indices: Optional[List[int]] = None
        self._weights: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, **kwargs) -> 'RegressionRemover':
        """
        Fit regression weights from artifact to EEG channels.
        X: (n_trials, n_channels, n_times)
        """
        n_channels = X.shape[1]
        self._eeg_indices = [i for i in range(n_channels) if i not in self.artifact_channel_indices]

        artifact = X[:, self.artifact_channel_indices, :].transpose(0, 2, 1).reshape(-1, len(self.artifact_channel_indices))
        eeg = X[:, self._eeg_indices, :].transpose(0, 2, 1).reshape(-1, len(self._eeg_indices))

        self._weights, _, _, _ = np.linalg.lstsq(artifact, eeg, rcond=None)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Remove artifact contribution from EEG channels.
        Returns EEG-only array: (n_trials, n_eeg_channels, n_times)
        """
        if self._weights is None or self._eeg_indices is None:
            raise RuntimeError('RegressionRemover must be fitted before transform.')

        n_trials, _, n_times = X.shape
        artifact = X[:, self.artifact_channel_indices, :].transpose(0, 2, 1).reshape(-1, len(self.artifact_channel_indices))
        eeg = X[:, self._eeg_indices, :]

        contribution = (artifact @ self._weights).reshape(n_trials, n_times, len(self._eeg_indices)).transpose(0, 2, 1)
        return eeg - contribution

    def __repr__(self) -> str:
        return (
            f'RegressionRemover(type={self.artifact_type}, '
            f'channels={self.artifact_channel_indices})'
        )


class ICARemover(BaseArtifactRemover):
    """
    Type 2: Artifact removal via Independent Component Analysis.
    Automatically detects and removes artifact-correlated ICA components.
    More robust than regression but slower.

    Args:
        sfreq: sampling frequency in Hz.
        artifact_indices: indices of artifact channels in the data array.
        artifact_type: type of artifact — used by MNE's find_bads_* methods.
        n_components: number of ICA components to compute.
    """

    # maps artifact_type to the MNE find_bads_* method name
    _MNE_FIND_BADS: dict[str, str] = {
        'eog': 'find_bads_eog',
        'ecg': 'find_bads_ecg',
        'muscle': 'find_bads_muscle',
        'ref_meg': 'find_bads_ref',
    }

    def __init__(
        self,
        sfreq: float,
        artifact_indices: List[int],
        artifact_type: ArtifactType = 'eog',
        n_components: int = 20,
    ):
        self.sfreq = sfreq
        self.artifact_indices = artifact_indices
        self.artifact_type = artifact_type
        self.n_components = n_components
        self._ica = None
        self._exclude: List[int] = []
        self._eeg_indices: Optional[List[int]] = None

    def fit(self, X: np.ndarray, **kwargs) -> 'ICARemover':
        """
        Fit ICA and detect artifact-correlated components.
        X: (n_trials, n_channels, n_times)
        """
        import mne
        from mne.preprocessing import ICA

        n_trials, n_chs, n_times = X.shape
        self._eeg_indices = [i for i in range(n_chs) if i not in self.artifact_indices]

        # set correct channel types so MNE can use find_bads_*
        ch_types = ['eeg'] * n_chs
        for idx in self.artifact_indices:
            ch_types[idx] = self.artifact_type

        info = mne.create_info(
            ch_names=[f'ch{i}' for i in range(n_chs)],
            sfreq=self.sfreq,
            ch_types=ch_types,
            verbose=False,
        )
        raw = mne.io.RawArray(
            X.transpose(1, 0, 2).reshape(n_chs, -1),
            info,
            verbose=False,
        )

        self._ica = ICA(
            n_components=self.n_components,
            method='fastica',
            random_state=42,
            verbose=False,
        )
        self._ica.fit(raw, verbose=False)

        # use the appropriate MNE find_bads_* method for this artifact type
        find_bads_method = self._MNE_FIND_BADS.get(self.artifact_type)
        if find_bads_method and hasattr(self._ica, find_bads_method):
            artifact_ch_names = [f'ch{i}' for i in self.artifact_indices]
            self._exclude, _ = getattr(self._ica, find_bads_method)(
                raw, ch_name=artifact_ch_names, verbose=False
            )
        else:
            self._exclude = []

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Remove artifact components and return EEG-only array.
        X: (n_trials, n_channels, n_times)
        Returns: (n_trials, n_eeg_channels, n_times)
        """
        import mne

        n_trials, n_chs, n_times = X.shape
        info = mne.create_info(
            ch_names=[f'ch{i}' for i in range(n_chs)],
            sfreq=self.sfreq,
            ch_types=['eeg'] * n_chs,
            verbose=False,
        )
        epochs = mne.EpochsArray(X, info, verbose=False)

        self._ica.exclude = self._exclude
        self._ica.apply(epochs, verbose=False)

        return epochs.get_data()[:, self._eeg_indices, :]

    def __repr__(self) -> str:
        return (
            f'ICARemover(type={self.artifact_type}, '
            f'sfreq={self.sfreq}, '
            f'n_components={self.n_components}, '
            f'excluded={self._exclude})'
        )


class HybridRemover(BaseArtifactRemover):
    """
    Type 3: Hybrid ICA + Regression artifact removal.
    Fits ICA on EEG-only channels, then regresses artifact signal
    out of each ICA source before reconstruction.
    More precise than pure ICA exclusion — removes partial artifact
    contamination from components rather than discarding them entirely.

    Args:
        sfreq: sampling frequency in Hz.
        artifact_indices: indices of artifact channels in the data array.
        artifact_type: type label for logging purposes.
        n_components: number of ICA components to compute.
    """

    def __init__(
        self,
        sfreq: float,
        artifact_indices: List[int],
        artifact_type: ArtifactType = 'eog',
        n_components: int = 20,
    ):
        self.sfreq = sfreq
        self.artifact_indices = artifact_indices
        self.artifact_type = artifact_type
        self.n_components = n_components
        self._ica = None
        self._weights: Optional[np.ndarray] = None
        self._eeg_indices: Optional[List[int]] = None

    def fit(self, X: np.ndarray, **kwargs) -> 'HybridRemover':
        """
        Fit ICA on EEG channels, then fit regression from artifact to ICA sources.
        X: (n_trials, n_channels, n_times)
        """
        import mne
        from mne.preprocessing import ICA

        n_trials, n_chs, n_times = X.shape
        self._eeg_indices = [i for i in range(n_chs) if i not in self.artifact_indices]
        n_eeg = len(self._eeg_indices)

        info_eeg = mne.create_info(
            ch_names=[f'ch{i}' for i in self._eeg_indices],
            sfreq=self.sfreq,
            ch_types='eeg',
            verbose=False,
        )
        raw_eeg = mne.io.RawArray(
            X[:, self._eeg_indices, :].transpose(1, 0, 2).reshape(n_eeg, -1),
            info_eeg,
            verbose=False,
        )

        self._ica = ICA(
            n_components=self.n_components,
            method='fastica',
            random_state=42,
            verbose=False,
        )
        self._ica.fit(raw_eeg, verbose=False)

        # sources: (n_samples, n_components)
        sources = self._ica.get_sources(raw_eeg).get_data().T

        # artifact: (n_samples, n_artifact)
        artifact = X[:, self.artifact_indices, :].transpose(1, 0, 2).reshape(len(self.artifact_indices), -1).T

        # fit: artifact @ W ≈ sources
        self._weights, _, _, _ = np.linalg.lstsq(artifact, sources, rcond=None)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Remove artifact contribution from ICA sources then reconstruct EEG.
        Returns EEG-only array: (n_trials, n_eeg_channels, n_times)
        """
        import mne

        n_trials, n_chs, n_times = X.shape

        info_eeg = mne.create_info(
            ch_names=[f'ch{i}' for i in self._eeg_indices],
            sfreq=self.sfreq,
            ch_types='eeg',
            verbose=False,
        )
        epochs_eeg = mne.EpochsArray(
            X[:, self._eeg_indices, :], info_eeg, verbose=False
        )

        # sources: (n_trials, n_components, n_times)
        sources = self._ica.get_sources(epochs_eeg).get_data()

        # artifact: (n_trials, n_artifact, n_times)
        artifact = X[:, self.artifact_indices, :]

        # remove artifact contribution from sources
        # weights: (n_artifact, n_components)
        # einsum: (n_trials, n_times, n_artifact) @ (n_artifact, n_components) → (n_trials, n_times, n_components)
        eog_contribution = np.einsum('tij,jk->tik', artifact.transpose(0, 2, 1), self._weights)
        sources_clean = sources - eog_contribution.transpose(0, 2, 1)

        # reconstruct: mixing (n_eeg, n_components) @ sources (n_components, n_times)
        mixing = self._ica.mixing_matrix_   # (n_eeg, n_components)
        cleaned = np.einsum('ij,tjk->tik', mixing, sources_clean)

        return cleaned

    def __repr__(self) -> str:
        return (
            f'HybridRemover(type={self.artifact_type}, '
            f'sfreq={self.sfreq}, '
            f'n_components={self.n_components})'
        )