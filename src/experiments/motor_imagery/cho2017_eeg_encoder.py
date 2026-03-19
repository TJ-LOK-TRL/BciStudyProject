import numpy as np
from src.experiments.base_experiment import BaseExperiment, ExperimentConfig
from src.datasets.motor_imagery.cho2017 import Cho2017
from src.models.eeg_encoder import EEGEncoderModel
from src.evaluation.validation import evaluate_intra_subject_fixed_split
from src.evaluation.results import EvaluationResult
from src.models.wrappers.nn_wrapper import NNWrapper
from src.training.callbacks.configs import EarlyStoppingCallbackConfig, LoggerCallbackConfig
from src.training.trainer_config import TrainerConfig
from src.utils.setup_seed import set_global_seed


class Cho2017EEGEncoderExperiment_9subjects(BaseExperiment):
    """
    Result: 69.3% ± 17.3% (intra-subject, 9 subjects, 2 classes)
    Subject scores: [90.0, 37.5, 87.5, 90.0, 75.0, 70.0, 56.2, 52.5, 64.6]

    EEGEncoder on Cho2017 (GigaDB) — 2 class MI (left/right hand).
    No EOG removal applied.
    Only 9 of 52 subjects tested.

    Comparison with BCI2a (same model, no EOG):
        BCI2a  (22ch, 4 classes): 76.1%
        Cho2017 (64ch, 2 classes): 69.3%

    Note: 2-class chance = 50%, 4-class chance = 25%.
    Adjusted above chance:
        BCI2a:   76.1% - 25% = +51.1pp
        Cho2017: 69.3% - 50% = +19.3pp

    Cho2017 is harder despite being binary — likely due to:
        - No EOG removal (Cho2017 has no EOG channels available)
        - Different recording setup (Biosemi, 512Hz resampled to 250Hz)
        - High subject variability (subject 2 = 37.5%, near chance)
    """

    @property
    def config(self) -> ExperimentConfig:
        return ExperimentConfig(
            name='cho2017_eeg_encoder',
            dataset='Cho2017 GigaDB (9/52 subjects, 2 classes: left/right hand)',
            model='EEGEncoder (ConvBlock + 5x DSTS: TCN + LLaMA transformer)',
            preprocessing='None — no EOG removal (Cho2017 has no EOG channels)',
            evaluation='intra-subject fixed split, test_ratio=0.2 (stratified)',
            notes=(
                'n_channels=64, n_classes=2, n_epochs=200, eegn_kern_size=65, '
                'tmin=0.0, tmax=3.0, resample=250Hz, '
                'EarlyStoppingCallback(patience=50), seed=42. '
                'Only 9 subjects tested — full 52 subjects pending.'
            ),
        )

    def prepare_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        set_global_seed()
        dataset = Cho2017(
            subject_ids=list(range(1, 10)),
            tmin=0.0,
            tmax=3.0,
        )
        X, y = dataset.get_data()
        subject_ids = dataset.metadata['moabb_metadata']['subject'].to_numpy()
        return X, y, subject_ids

    def build_model(self) -> NNWrapper:
        return NNWrapper(
            arch=EEGEncoderModel(
                n_channels=64,
                n_classes=2,
                eegn_kern_size=65,
            ),
            config=TrainerConfig(
                n_epochs=200,
                l2_scale=2.0,
                loss_scale=2.0,
                label_smoothing=0.2,
                logger=LoggerCallbackConfig(every_n_epochs=10),
                early_stopping=EarlyStoppingCallbackConfig(patience=50),
            ),
        )

    def run(self) -> EvaluationResult:
        X, y, subject_ids = self.prepare_data()
        model = self.build_model()
        result = evaluate_intra_subject_fixed_split(
            model, X, y, subject_ids, test_ratio=0.2
        )
        print(self)
        print(result)
        return result


if __name__ == '__main__':
    experiment = Cho2017EEGEncoderExperiment_9subjects()
    result = experiment.run()
