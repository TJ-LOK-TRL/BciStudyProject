from typing import Optional, List
import numpy as np
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

from src.datasets.base_moabb_dataset import BaseMoabbMiDataset 


class BCICompIV2a(BaseMoabbMiDataset):
    """
    Dataset 1: BCI Competition IV Dataset 2a.
    22 channels, 9 subjects, classes: left hand / right hand / tongue / foot.
    Source: https://www.bbci.de/competition/iv/download/index.html
    Loaded via MOABB (BNCI2014_001).
    """

    CLASS_NAMES: List[str] = ['left_hand', 'right_hand', 'tongue', 'foot']

    def __init__(
        self,
        subject_ids: Optional[List[int]] = None,
        tmin: float = 0.5,
        tmax: float = 3.5,
        resample: float = 250 
    ):
        super().__init__(
            moabb_dataset=BNCI2014_001(),
            subject_ids=subject_ids,
            resample=resample,
            tmin=tmin,
            tmax=tmax
        )

    @property
    def n_classes(self) -> int:
        return len(self.CLASS_NAMES)

    @property
    def class_names(self) -> list[str]:
        return self.CLASS_NAMES

    @property
    def n_channels(self) -> int:
        return 22

    @property
    def sfreq(self) -> float:
        return 250.0

if __name__ == '__main__':
    dataset = BCICompIV2a()
    print(dataset)
    X, y = dataset.get_data()
    print(f'X shape: {X.shape}')
    print(f'y shape: {y.shape}')
    print(f'Classes: {np.unique(y)}')