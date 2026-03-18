from typing import Optional, List
import numpy as np
from moabb.datasets import BNCI2014_002
from src.datasets.base_moabb_dataset import BaseMoabbMiDataset


class BNCIHorizon2014002(BaseMoabbMiDataset):
    """
    Dataset 6: BNCI Horizon 2014-002.
    15 channels, 14 subjects, classes: right hand / feet.
    Source: http://bnci-horizon-2020.eu/database/data-sets
    Loaded via MOABB (BNCI2014_002).
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
            moabb_dataset=BNCI2014_002(),
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
        return 15

    @property
    def sfreq(self) -> float:
        return 250.0


if __name__ == '__main__':
    dataset = BNCIHorizon2014002(subject_ids=[1, 2])
    print(dataset)
    X, y = dataset.get_data()
    print(f'X shape: {X.shape}')
    print(f'Classes: {np.unique(y)}')