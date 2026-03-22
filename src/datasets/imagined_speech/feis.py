from typing import Optional, List, Literal
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd

from src.datasets.base_imaged_speech_dataset import BaseImagedSpeechDataset


class FEIS(BaseImagedSpeechDataset):
    """
    FEIS: Fourteen-channel EEG with Imagined Speech.
    14 channels (Emotiv EPOC+), 21 English subjects, 16 phonemes, 256Hz.
    10 trials per phoneme per subject, 5 seconds per trial (1280 samples).

    Source: https://zenodo.org/records/3369178
    Wellington, S., Clayton, J. (2019). FEIS dataset. doi:10.5281/zenodo.3369178

    Phases available: thinking (imagined), speaking (actual), stimuli (heard),
                      articulators (fixation), resting (baseline).
    """

    EEG_CHANNELS: List[str] = [
        'F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1',
        'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4',
    ]

    PHONEMES: List[str] = [
        'f', 'fleece', 'goose', 'k', 'm', 'n', 'ng',
        'p', 's', 'sh', 't', 'thought', 'trap', 'v', 'z', 'zh',
    ]

    def __init__(
        self,
        data_path: str,
        subject_ids: Optional[List[int]] = None,
        phase: Literal['thinking', 'speaking', 'stimuli', 'articulators', 'resting'] = 'thinking',
        labels: Optional[List[str]] = None,
    ):
        super().__init__(
            data_path=data_path,
            subject_ids=subject_ids or list(range(1, 22)),
        )
        self.phase = phase
        self.labels = labels or self.PHONEMES

    @property
    def n_classes(self) -> int:
        return len(self.labels)

    @property
    def class_names(self) -> List[str]:
        return self.labels

    @property
    def n_channels(self) -> int:
        return len(self.EEG_CHANNELS)

    @property
    def sfreq(self) -> float:
        return 256.0

    def _subject_dir(self, subject_id: int) -> Path:
        """Return path to subject directory — zero-padded (01, 02, ...)."""
        return Path(self.data_path) / f'{subject_id:02d}'

    def _load_subject(self, subject_id: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Load one subject's data from zip.
        Returns X: (n_trials, n_channels, n_times), y: (n_trials,)
        """
        zip_path = self._subject_dir(subject_id) / f'{self.phase}.zip'

        if not zip_path.exists():
            raise FileNotFoundError(f'Missing: {zip_path}')

        with zipfile.ZipFile(zip_path) as z:
            with z.open(f'{self.phase}.csv') as f:
                df = pd.read_csv(f)

        # filter to requested labels only
        df = df[df['Label'].isin(self.labels)]

        all_X = []
        all_y = []

        for (label, epoch), group in df.groupby(['Label', 'Epoch']):
            data = group[self.EEG_CHANNELS].values.T  # (14, n_times)
            all_X.append(data)
            all_y.append(label)

        return np.stack(all_X, axis=0).astype(np.float32), np.array(all_y)

    def load(self) -> None:
        """Load all subjects."""
        all_X, all_y, all_subjects = [], [], []

        base = Path(self.data_path)
        for subject_id in self.subject_ids:
            subject_dir = base / f'{subject_id:02d}'
            if not subject_dir.exists():
                print(f'  Warning: subject {subject_id:02d} not found, skipping')
                continue
            try:
                X, y = self._load_subject(subject_id)
                all_X.append(X)
                all_y.append(y)
                all_subjects.extend([subject_id] * len(X))
                print(f'  Subject {subject_id:02d}: {X.shape}')
            except Exception as e:
                print(f'  Warning: subject {subject_id:02d} failed — {e}')

        if not all_X:
            raise RuntimeError(f'No data loaded for subjects {self.subject_ids}')

        self.X = np.concatenate(all_X, axis=0)
        self.y = np.concatenate(all_y, axis=0)
        self.metadata['subject_ids'] = np.array(all_subjects)
        self.metadata['n_subjects'] = len(set(all_subjects))

    def preprocess(self) -> None:
        pass

if __name__ == '__main__':
    DATA_PATH = 'data/imagined_speech/scottwellington-FEIS-7e726fd/experiments'

    # test with 2 subjects, 4 phonemes
    dataset = FEIS(
        data_path=DATA_PATH,
        subject_ids=[1, 2],
        phase='thinking',
        labels=['m', 'n', 's', 'f'],
    )
    print(dataset)
    X, y = dataset.get_data()
    subject_ids = dataset.metadata['subject_ids']

    print(f'X shape:   {X.shape}')
    print(f'y unique:  {np.unique(y)}')
    print(f'Subjects:  {np.unique(subject_ids)}')
    print(f'Trials/subject: {len(X) // len(np.unique(subject_ids))}')
