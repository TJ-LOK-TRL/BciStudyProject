from typing import Optional, List, Literal, Dict
from pathlib import Path
import numpy as np
import mne

from src.datasets.base_imaged_speech_dataset import BaseImagedSpeechDataset

class KumarImagedSpeech(BaseImagedSpeechDataset):
    """
    Kumar et al. 2018 — Envisioned Speech Recognition using EEG.
    14 channels, 23 subjects, 128Hz, 10s trials.
    3 task types: Digits (0-9), Characters (A-Z), Objects (everyday items).
    Source: https://www.kaggle.com/datasets/ignazio/kumars-eeg-imagined-speech

    Kumar, P., Saini, R., Roy, P.P., Sahu, P.K. and Dogra, D.P., 2018.
    Envisioned speech recognition using EEG sensors.
    Personal and Ubiquitous Computing, 22, pp.185-199.
    """

    TASK_DIRS: Dict[str, str] = {
        'digit':     'Digit',
        'character': 'Char',
        'image':     'Image',
    }

    EEG_CHANNELS: List[str] = [
        'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
        'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'
    ]

    def __init__(
        self,
        data_path: Optional[str] = None,
        subject_ids: Optional[List[int]] = None,
        task: Literal['digit', 'character', 'image'] = 'digit',
    ):
        super().__init__(
            data_path=data_path or '',
            subject_ids=subject_ids or list(range(1, 24)),
        )
        self.task = task

    @property
    def n_classes(self) -> int:
        return {'digit': 10, 'character': 26, 'image': 10}[self.task]

    @property
    def class_names(self) -> List[str]:
        if self.task == 'digit':
            return [str(i) for i in range(10)]
        elif self.task == 'character':
            return [chr(ord('A') + i) for i in range(26)]
        else:
            return [f'object_{i}' for i in range(10)]

    @property
    def n_channels(self) -> int:
        return 14

    @property
    def sfreq(self) -> float:
        return 128.0

    def load(self) -> None:
        task_dir = Path(self.data_path) / 'Imagined_speech_EEG_edf' / self.TASK_DIRS[self.task]
        edf_files = sorted(task_dir.glob('*.edf'))

        if not edf_files:
            raise FileNotFoundError(f'No EDF files found in {task_dir}')

        # most common n_times across files
        target_samples = 1280  # 10s * 128Hz as per paper

        all_X = []
        all_y = []
        all_subjects = []

        for edf_path in edf_files:
            try:
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

                # pick only EEG channels — removes annotations/trigger channels
                raw.pick_channels(self.EEG_CHANNELS, verbose=False)

                subject_id, label = self._parse_filename(edf_path.name)
                if subject_id not in self.subject_ids:
                    continue

                data = raw.get_data()   # (n_eeg_channels, n_times)

                # truncate or pad to target_samples
                n_times = data.shape[1]
                if n_times >= target_samples:
                    data = data[:, :target_samples]   # truncate
                else:
                    # pad with zeros if shorter
                    pad = np.zeros((data.shape[0], target_samples - n_times))
                    data = np.concatenate([data, pad], axis=1)

                all_X.append(data)
                all_y.append(label)
                all_subjects.append(subject_id)

            except Exception as e:
                print(f'  Warning: skipping {edf_path.name} — {e}')
                continue

        if not all_X:
            raise RuntimeError(f'No data loaded for subjects {self.subject_ids}')

        self.X = np.stack(all_X, axis=0).astype(np.float32)
        self.y = np.array(all_y)
        self.metadata['subject_ids'] = np.array(all_subjects)
        self.metadata['n_subjects'] = len(set(all_subjects))
        print(f'  Loaded: {self.X.shape}, n_channels after pick: {self.X.shape[1]}')

    def _parse_filename(self, filename: str) -> tuple[int, str]:
        """
        Parse subject id and label from filename.
        Pattern: name<subject_id>_<label>.edf
        Examples: name0_A.edf, name1_0.edf, name0_Apple.edf
        Note: subject_ids in filenames are 0-indexed (name0 = subject 1).
        """
        stem = Path(filename).stem          # 'name0_A'
        parts = stem.split('_', 1)          # ['name0', 'A']
        subject_id = int(parts[0].replace('name', '')) + 1  # 0-indexed → 1-indexed
        label = parts[1].lower()            # 'a', '0', 'apple'
        return subject_id, label

    def preprocess(self) -> None:
        """Placeholder — add filtering as needed."""
        pass

if __name__ == '__main__':
    dataset = KumarImagedSpeech(
        data_path='data/imagined_speech',
        task='digit',
        subject_ids=list(range(1, 24)),
    )
    print(dataset)
    X, y = dataset.get_data()
    print(f'X shape: {X.shape}')
    print(f'y unique: {np.unique(y)}')
    print(f'Subject ids: {np.unique(dataset.metadata["subject_ids"])}')
    print(f'Trials per subject: {len(X) // len(np.unique(dataset.metadata["subject_ids"]))}')