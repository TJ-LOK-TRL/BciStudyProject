import numpy as np
from src.experiments.base_experiment import BaseExperiment, ExperimentConfig
from src.datasets.motor_imagery import BNCICompIII3a
from src.models.eeg_encoder import EEGEncoderModel
from src.evaluation.validation import evaluate_intra_subject_fixed_split
from src.evaluation.results import EvaluationResult
from src.models.wrappers.nn_wrapper import NNWrapper
from src.training.callbacks.configs import EarlyStoppingCallbackConfig, LoggerCallbackConfig
from src.training.trainer_config import TrainerConfig
from src.utils.setup_seed import set_global_seed


class BNCI2014004EEGEncoderExperiment(BaseExperiment):
    """
    Result: 81.0% ± 11.8% (intra-subject, 9 subjects, 2 classes, NO EOG removal)
    Subject scores: [81.2, 66.9, 55.6, 97.3, 87.8, 86.1, 80.6, 84.9, 88.2]

    EEGEncoder on BNCI2014_004 — 2 class MI (left/right hand), no EOG removal.
    Only 3 EEG channels (C3, Cz, C4) — central channels, low EOG contamination.

    See BNCI2014004EEGEncoderEOGExperiment for comparison with EOG removal (+3.2pp).

    Key finding: Even with only 3 channels, EEGEncoder achieves 81% —
    demonstrates strong temporal modelling compensates for spatial limitations.
    """

    @property
    def config(self) -> ExperimentConfig:
        return ExperimentConfig(
            name='bnci2014_004_eeg_encoder',
            dataset='BNCI2014_004 (9 subjects, 2 classes: left/right hand)',
            model='EEGEncoder (ConvBlock + 5x DSTS: TCN + LLaMA transformer)',
            preprocessing='None — no EOG removal',
            evaluation='intra-subject fixed split, test_ratio=0.2 (stratified)',
            notes=(
                'n_channels=3, n_classes=2, n_epochs=200, eegn_kern_size=65, '
                'tmin=-0.5, tmax=4.5-1/250, resample=250Hz, '
                'EarlyStoppingCallback(patience=50), seed=42.'
            ),
        )

    def prepare_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        set_global_seed()
        dataset = BNCICompIII3a(
            subject_ids=list(range(1, 10)),
            tmin=-0.5,
            tmax=4.5 - 1/250,
            include_eog=False,
        )
        X, y = dataset.get_data()
        subject_ids = dataset.metadata['moabb_metadata']['subject'].to_numpy()
        return X, y, subject_ids

    def build_model(self) -> NNWrapper:
        return NNWrapper(
            arch=EEGEncoderModel(
                n_channels=3,
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
    experiment = BNCI2014004EEGEncoderExperiment()
    result = experiment.run()