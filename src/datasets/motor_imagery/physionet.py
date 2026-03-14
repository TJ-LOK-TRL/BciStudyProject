from typing import Optional, List
import numpy as np
from moabb.datasets import PhysionetMI as MOABBPhysionetMI

from src.datasets.base_moabb_dataset import BaseMoabbMiDataset


class PhysionetMI(BaseMoabbMiDataset):
    """
    Dataset 2: PhysioNet Motor Imagery (EEGMMIDB).
    64 channels, 109 subjects, classes: left hand / right hand / feet / both hands.
    Source: https://physionet.org/content/eegmmidb/1.0.0
    Loaded via MOABB (PhysionetMI).
    """

    CLASS_NAMES: List[str] = ['left_hand', 'right_hand', 'feet', 'both_hands']

    def __init__(
        self,
        subject_ids: Optional[List[int]] = None,
        tmin: float = 0.0,
        tmax: float = 3.0,
        resample: float = 160.0,
    ):
        super().__init__(
            moabb_dataset=MOABBPhysionetMI(),
            subject_ids=subject_ids or list(range(1, 110)),
            resample=resample,
            tmin=tmin,
            tmax=tmax,
            events=['left_hand', 'right_hand', 'feet', 'hands'],
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
        return 160.0


if __name__ == '__main__':
    # Test with 2 subjects to avoid long download
    dataset = PhysionetMI(subject_ids=[1, 2])
    print(dataset)
    X, y = dataset.get_data()
    print(f'X shape: {X.shape}')
    print(f'y shape: {y.shape}')
    print(f'Classes: {np.unique(y)}')