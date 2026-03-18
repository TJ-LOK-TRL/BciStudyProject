from typing import Optional, List
import numpy as np
from moabb.datasets import BNCI2015_001
from src.datasets.base_moabb_dataset import BaseMoabbMiDataset


class BNCI2015001(BaseMoabbMiDataset):
    """
    BNCI 2015-001 Motor Imagery Dataset.
    13 channels, 12 subjects, classes: right hand / feet.
    Loaded via MOABB (BNCI2015_001).
    """

    CLASS_NAMES: List[str] = ['right_hand', 'feet']

    def __init__(
        self,
        subject_ids: Optional[List[int]] = None,
        tmin: float = 0.0,
        tmax: float = 5.0,
        resample: float = 250.0,
    ):
        super().__init__(
            moabb_dataset=BNCI2015_001(),
            subject_ids=subject_ids or list(range(1, 13)),
            resample=resample,
            tmin=tmin,
            tmax=tmax,
        )

    @property
    def n_classes(self) -> int:
        return len(self.CLASS_NAMES)

    @property
    def class_names(self) -> List[str]:
        return self.CLASS_NAMES

    @property
    def n_channels(self) -> int:
        return 13

    @property
    def sfreq(self) -> float:
        return 250.0


if __name__ == '__main__':
    dataset = BNCI2015001(subject_ids=[1, 2])
    print(dataset)
    X, y = dataset.get_data()
    print(f'X shape: {X.shape}')
    print(f'Classes: {np.unique(y)}')