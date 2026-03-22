import numpy as np
from src.experiments.base_experiment import BaseExperiment, ExperimentConfig
from src.datasets.motor_imagery.bci_comp_iv_2a import BCICompIV2a
from src.input_adapters.cnn2d_adapter import CNN2DAdapter
from src.models.eeg_encoder import EEGEncoderModel
from src.evaluation.validation import evaluate_intra_subject_fixed_split
from src.evaluation.results import EvaluationResult
from src.models.wrappers.nn_wrapper import NNWrapper
from src.preprocessing.transforms.artifacts import RegressionRemover
from src.training.callbacks.configs import EarlyStoppingCallbackConfig, LoggerCallbackConfig
from src.training.trainer_config import TrainerConfig
from src.utils.setup_seed import set_global_seed


class BCI2aEEGEncoderEOGExperiment(BaseExperiment):
    """
    Result: 85.0% ± 11.4%
    Subject scores: [88.8, 69.0, 93.1, 72.4, 92.2, 66.4, 94.8, 96.6, 91.4]

    Key findings:
    - EOG regression removal + wider time window (1125 samples) gives +9% over baseline
    - Without EOG removal but same time window: 76.1% (confirmed)
    - EOG removal alone accounts for the full improvement
    - Subjects 2 and 6 remain the most difficult

    Comparison:
    - EEGEncoder no EOG (tmin=-0.5):  76.1%
    - EEGEncoder + EOG removal:       85.0%  ← this experiment
    - RiemannianSVM no EOG:           75.8%
    - This same paper reported:       86.0%
    """

    EOG_INDICES = [22, 23, 24]

    @property
    def config(self) -> ExperimentConfig:
        return ExperimentConfig(
            name='bci2a_eeg_encoder_eog_regression',
            dataset='BCI Competition IV 2a (9 subjects, 4 classes)',
            model='EEGEncoder (ConvBlock + 5x DSTS: TCN + LLaMA transformer)',
            preprocessing='EOG regression removal (channels 22,23,24) — fit on full dataset',
            evaluation='intra-subject fixed split, test_ratio=0.2 (stratified)',
            notes=(
                'tmin=-0.5, tmax=4.0-1/250 (1125 samples), '
                'include_eog=True, RegressionRemover fit on full X (data leakage caveat), '
                'l2_scale=2.0, label_smoothing=0.2, lr=1e-3, '
                'batch_size=64, n_epochs=200, eegn_kern_size=64, '
                'EarlyStoppingCallback(patience=50), seed=42'
            ),
        )

    def prepare_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        set_global_seed()
        dataset = BCICompIV2a(
            subject_ids=list(range(1, 10)),
            tmin=-0.5,
            tmax=4.0 - 1/250,
            include_eog=True,
        )
        X, y = dataset.get_data()
        subject_ids = dataset.metadata['moabb_metadata']['subject'].to_numpy()

        remover = RegressionRemover(
            artifact_channel_indices=self.EOG_INDICES,
            artifact_type='eog',
        )
        X = remover.fit_transform(X)
        return X, y, subject_ids

    def build_model(self, 
                    eegn_kern_size: int = 64,
                    lr: float = 1e-3,
                    n_epochs: int = 200,
                    loss_scale: float = 2.0,
                    l2_scale: float = 2.0,
                    label_smoothing: float = 0.2,
                    patience: int = 50
                    ) -> NNWrapper:
        return NNWrapper(
            arch=EEGEncoderModel(
                n_channels=22,
                n_classes=4,
                eegn_kern_size = eegn_kern_size,
            ),
            config=TrainerConfig(
                lr=lr,
                n_epochs=n_epochs,
                l2_scale=l2_scale,
                loss_scale=loss_scale,
                label_smoothing=label_smoothing,
                grad_clip=0.0,
                input_adapter=CNN2DAdapter(),
                logger=LoggerCallbackConfig(every_n_epochs=10),
                early_stopping=EarlyStoppingCallbackConfig(patience=patience),
            ),
        )

    def run(self, model=None, X=None, y=None, subject_ids=None, test_ratio: int = 0.2) -> EvaluationResult:
        #X, y, subject_ids = self.prepare_data()
        #model = self.build_model()

        result = evaluate_intra_subject_fixed_split(
            model=model,
            X=X,
            y=y,
            subject_ids=subject_ids,
            test_ratio=test_ratio,
            #save_dir='results/models/bci2a_eeg_encoder_eog',
        )
        print(self)
        print(result)
        return result


if __name__ == '__main__':
    experiment = BCI2aEEGEncoderEOGExperiment()
    result = experiment.run()