from typing import Optional, List
import numpy as np
from moabb.datasets import Lee2019_MI as MOABBLee2019MI
from src.datasets.base_moabb_dataset import BaseMoabbMiDataset


class Lee2019MI(BaseMoabbMiDataset):
    """
    Lee 2019 Motor Imagery Dataset.
    62 channels, 54 subjects, classes: left hand / right hand.
    Loaded via MOABB (Lee2019_MI).
    """

    CLASS_NAMES: List[str] = ['left_hand', 'right_hand']

    def __init__(
        self,
        subject_ids: Optional[List[int]] = None,
        tmin: float = 0.0,
        tmax: float = 4.0,
        resample: float = 250.0,
    ):
        super().__init__(
            moabb_dataset=MOABBLee2019MI(),
            subject_ids=subject_ids or list(range(1, 55)),
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
        return 62

    @property
    def sfreq(self) -> float:
        return 250.0

if __name__ == '__main__':
    dataset = Lee2019MI(subject_ids=[1, 2])
    print(dataset)
    X, y = dataset.get_data()
    print(f'X shape: {X.shape}')
    print(f'Classes: {np.unique(y)}')