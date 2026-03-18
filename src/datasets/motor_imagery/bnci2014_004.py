from typing import Optional, List
import numpy as np
from moabb.datasets import BNCI2014_004
from src.datasets.base_moabb_dataset import BaseMoabbMiDataset


class BNCICompIII3a(BaseMoabbMiDataset):
    """
    Dataset 5 (approx): BCI Competition III Dataset IIIa.
    3 channels (C3, Cz, C4), 9 subjects, classes: left hand / right hand.
    Source: http://www.bbci.de/competition/iii/desc_IIIa.html
    Loaded via MOABB (BNCI2014_004) — closest available equivalent.
    """

    CLASS_NAMES: List[str] = ['left_hand', 'right_hand']

    def __init__(
        self,
        subject_ids: Optional[List[int]] = None,
        tmin: float = 0.0,
        tmax: float = 4.5,
        resample: float = 250.0,
    ):
        super().__init__(
            moabb_dataset=BNCI2014_004(),
            subject_ids=subject_ids or list(range(1, 10)),
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
        return 3

    @property
    def sfreq(self) -> float:
        return 250.0


if __name__ == '__main__':
    dataset = BNCICompIII3a(subject_ids=[1, 2])
    print(dataset)
    X, y = dataset.get_data()
    print(f'X shape: {X.shape}')
    print(f'Classes: {np.unique(y)}')