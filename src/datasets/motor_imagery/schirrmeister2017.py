from typing import Optional, List
import numpy as np
from moabb.datasets import Schirrmeister2017 as MOABBSchirrmeister2017
from src.datasets.base_moabb_dataset import BaseMoabbMiDataset


class Schirrmeister2017(BaseMoabbMiDataset):
    """
    High Gamma Dataset (Schirrmeister et al. 2017).
    128 channels, 14 subjects, classes: left hand / right hand / feet / rest.
    Source: https://github.com/robintibor/high-gamma-dataset
    Loaded via MOABB (Schirrmeister2017).
    Note: Same authors as ShallowConvNet paper — designed for deep learning.
    """

    CLASS_NAMES: List[str] = ['left_hand', 'right_hand', 'feet', 'rest']

    def __init__(
        self,
        subject_ids: Optional[List[int]] = None,
        tmin: float = 0.0,
        tmax: float = 4.0,
        resample: float = 250.0,
    ):
        super().__init__(
            moabb_dataset=MOABBSchirrmeister2017(),
            subject_ids=subject_ids or list(range(1, 15)),
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
        return 128

    @property
    def sfreq(self) -> float:
        return 250.0


if __name__ == '__main__':
    dataset = Schirrmeister2017(subject_ids=[1, 2])
    print(dataset)
    X, y = dataset.get_data()
    print(f'X shape: {X.shape}')
    print(f'Classes: {np.unique(y)}')