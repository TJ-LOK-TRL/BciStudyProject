import numpy as np
from src.experiments.base_experiment import BaseExperiment, ExperimentConfig
from src.datasets.motor_imagery.bci_comp_iv_2a import BCICompIV2a
from src.models.eeg_encoder import EEGEncoderModel
from src.evaluation.validation import evaluate_intra_subject_fixed_split
from src.evaluation.results import EvaluationResult
from src.training.callbacks import EarlyStoppingCallback, LoggerCallback


class BCI2aEEGEncoderExperiment(BaseExperiment):
    """
    Result: 76.1% ± 13.6%
    Subject scores: [85.3, 67.2, 93.1, 61.2, 56.9, 60.3, 89.7, 82.8, 87.9]

    Confirmed reproducible with global seed=42.
    """

    @property
    def config(self) -> ExperimentConfig:
        return ExperimentConfig(
            name='bci2a_eeg_encoder',
            dataset='BCI Competition IV 2a (9 subjects, 4 classes)',
            model='EEGEncoder (ConvBlock + 5x DSTS: TCN + LLaMA transformer)',
            preprocessing='None — raw EEG, tmin=-0.5, tmax=4.0-1/250',
            evaluation='intra-subject fixed split, test_ratio=0.2 (stratified)',
            notes=(
                'l2_scale=2.0, label_smoothing=0.2, lr=1e-3, '
                'batch_size=64, n_epochs=200, eegn_kern_size=64, '
                'EarlyStoppingCallback(patience=50), '
                'val split 20% of train, best weights restored after ES or end of training'
            ),
        )

    def prepare_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dataset = BCICompIV2a(
            subject_ids=list(range(1, 10)),
            tmin=-0.5,
            tmax=4.0 - 1/250,
        )
        X, y = dataset.get_data()
        subject_ids = dataset.metadata['moabb_metadata']['subject'].to_numpy()
        return X, y, subject_ids

    def build_model(self) -> EEGEncoderModel:
        callbacks = [
            LoggerCallback(every_n_epochs=10, metrics=['train_loss', 'val_loss', 'train_acc', 'val_acc']),
            EarlyStoppingCallback(patience=50),
        ]
        return EEGEncoderModel(
            n_channels=22,
            n_classes=4,
            n_epochs=200,
            eegn_kern_size=64,
            callbacks=callbacks,
        )

    def run(self) -> EvaluationResult:
        from src.utils.setup_seed import set_global_seed
        set_global_seed()
        X, y, subject_ids = self.prepare_data()
        model = self.build_model()
        result = evaluate_intra_subject_fixed_split(
            model=model,
            X=X,
            y=y,
            subject_ids=subject_ids,
            test_ratio=0.2,
            #save_dir='results/models/bci2a_eeg_encoder',
        )
        print(self)
        print(result)
        return result


if __name__ == '__main__':
    experiment = BCI2aEEGEncoderExperiment()
    result = experiment.run()