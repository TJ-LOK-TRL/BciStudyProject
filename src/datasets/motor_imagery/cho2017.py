from typing import Optional, List
import numpy as np
from moabb.datasets import Cho2017 as MOABBCho2017
from src.datasets.base_moabb_dataset import BaseMoabbMiDataset


class Cho2017(BaseMoabbMiDataset):
    """
    Dataset 4 (approx): GigaDB Motor Imagery.
    64 channels, 52 subjects, classes: left hand / right hand.
    Source: http://gigadb.org/dataset/100295
    Loaded via MOABB (Cho2017).
    """

    CLASS_NAMES: List[str] = ['left_hand', 'right_hand']

    def __init__(
        self,
        subject_ids: Optional[List[int]] = None,
        tmin: float = 0.0,
        tmax: float = 3.0,
        resample: float = 250.0,
    ):
        super().__init__(
            moabb_dataset=MOABBCho2017(),
            subject_ids=subject_ids or list(range(1, 53)),
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
    dataset = Cho2017(subject_ids=[1, 2])
    print(dataset)
    X, y = dataset.get_data()
    print(f'X shape: {X.shape}')
    print(f'Classes: {np.unique(y)}')