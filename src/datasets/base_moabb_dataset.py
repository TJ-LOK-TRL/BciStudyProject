from typing import Optional, List
from moabb.datasets.base import BaseDataset as MOABBBase
from moabb.paradigms import MotorImagery
from moabb.paradigms.base import BaseParadigm

from src.datasets.base_dataset import BaseDataset

class BaseMoabbDataset(BaseDataset):
    """Abstract base class for datasets loaded via MOABB."""

    def __init__(
        self,
        moabb_dataset: MOABBBase,
        paradigm: BaseParadigm,
        subject_ids: Optional[List[int]],
    ):
        super().__init__(data_path='', subject_ids=subject_ids)
        self._moabb_dataset = moabb_dataset
        self._paradigm = paradigm

    def load(self) -> None:
        """Generic MOABB load — works for all subclasses."""
        X, y, metadata = self._paradigm.get_data(
            dataset=self._moabb_dataset,
            subjects=self.subject_ids,
        )
        self.X = X
        self.y = y
        self.metadata['n_subjects'] = len(self.subject_ids)
        self.metadata['moabb_metadata'] = metadata

    def preprocess(self) -> None:
        """Placeholder — override in subclass if needed."""
        pass

class BaseMoabbMiDataset(BaseMoabbDataset):
    def __init__(
        self, 
        moabb_dataset: MOABBBase, 
        subject_ids: List[str], 
        resample: float, 
        tmin: float, 
        tmax: float,
        events: Optional[List[str]] = None,
    ):
        self.tmin = tmin
        self.tmax = tmax
        self.resample = resample

        paradigm = MotorImagery(
            events=events,
            n_classes=len(events) if events else self.n_classes,
            resample=self.resample,
            tmin=self.tmin,
            tmax=self.tmax,
        )
        super().__init__(moabb_dataset, paradigm, subject_ids)