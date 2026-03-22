
from src.datasets.base_dataset import BaseDataset


class BaseImagedSpeechDataset(BaseDataset):
    @property
    def paradigm(self) -> str:
        return 'ImagerySpeech'