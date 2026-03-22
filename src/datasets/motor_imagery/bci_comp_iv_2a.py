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

    CLASS_NAMES: List[str] = ['left_hand', 'right_hand', 'tongue', 'feet']

    EEG_AND_EOG_CHANNELS = [
        'Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6',
        'CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz',
        'EOG1','EOG2','EOG3'
    ]

    def __init__(
        self,
        subject_ids: Optional[List[int]] = None,
        tmin: float = 0.5,
        tmax: float = 3.5,
        resample: float = 250,
        include_eog: bool = False
    ):
        super().__init__(
            moabb_dataset=BNCI2014_001(),
            subject_ids=subject_ids or [1, 2, 3, 4, 5, 6, 7, 8, 9],
            resample=resample,
            tmin=tmin,
            tmax=tmax,
            channels=self.EEG_AND_EOG_CHANNELS if include_eog else None
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
    dataset = BCICompIV2a(include_eog=True)
    print(dataset)
    X, y = dataset.get_data()
    print(f'X shape: {X.shape}')
    print(f'y shape: {y.shape}')
    print(f'Classes: {np.unique(y)}')
    