from typing import Optional, List
import numpy as np
from moabb.datasets import Stieger2021 as MOABBStieger2021
from src.datasets.base_moabb_dataset import BaseMoabbMiDataset


class Stieger2021(BaseMoabbMiDataset):
    """
    Stieger 2021 Motor Imagery Dataset.
    64 channels, 62 subjects, 4 classes, up to 11 sessions.
    One of the largest MI datasets available.
    Loaded via MOABB (Stieger2021).
    """

    CLASS_NAMES: List[str] = ['left_hand', 'right_hand', 'feet', 'rest']

    def __init__(
        self,
        subject_ids: Optional[List[int]] = None,
        tmin: float = 0.0,
        tmax: float = 3.0,
        resample: float = 250.0,
    ):
        super().__init__(
            moabb_dataset=MOABBStieger2021(),
            subject_ids=subject_ids or list(range(1, 63)),
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
        return 64

    @property
    def sfreq(self) -> float:
        return 250.0


if __name__ == '__main__':
    dataset = Stieger2021(subject_ids=[1, 2])
    print(dataset)
    X, y = dataset.get_data()
    print(f'X shape: {X.shape}')
    print(f'Classes: {np.unique(y)}')