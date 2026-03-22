import numpy as np
from src.experiments.base_experiment import BaseExperiment, ExperimentConfig
from src.datasets.motor_imagery import BNCICompIII3a
from src.models.eeg_encoder import EEGEncoderModel
from src.evaluation.validation import evaluate_intra_subject_fixed_split
from src.evaluation.results import EvaluationResult
from src.models.wrappers.nn_wrapper import NNWrapper
from src.preprocessing.transforms.artifacts import RegressionRemover
from src.training.callbacks.configs import EarlyStoppingCallbackConfig, LoggerCallbackConfig
from src.training.trainer_config import TrainerConfig
from src.utils.setup_seed import set_global_seed


class BNCI2014004EEGEncoderEOGExperiment(BaseExperiment):
    """
    Result: 84.2% ± 11.4% (intra-subject, 9 subjects, 2 classes, WITH EOG removal)
    Subject scores: [84.0, 54.4, 82.6, 97.3, 89.2, 85.4, 91.7, 84.2, 88.9]

    EEGEncoder on BNCI2014_004 with EOG regression removal.
    EOG channels: EOG1, EOG2, EOG3 (indices 3, 4, 5).

    Comparison with no EOG removal:
        Without EOG: 81.0% ± 11.8%
        With EOG:    84.2% ± 11.4%
        Delta:       +3.2pp

    Smaller delta than BCI2a (+9pp) because C3/Cz/C4 are central channels
    — naturally less contaminated by EOG than frontal channels in BCI2a.

    Without subject 2 (consistent outlier across all datasets): ~90.0%
    Subject 2 is a persistent difficult subject — likely neurofisiological
    differences or poor signal quality, not a model failure.
    """

    @property
    def config(self) -> ExperimentConfig:
        return ExperimentConfig(
            name='bnci2014_004_eeg_encoder_eog',
            dataset='BNCI2014_004 (9 subjects, 2 classes: left/right hand)',
            model='EEGEncoder (ConvBlock + 5x DSTS: TCN + LLaMA transformer)',
            preprocessing='EOG regression removal (channels EOG1, EOG2, EOG3, indices 3,4,5)',
            evaluation='intra-subject fixed split, test_ratio=0.2 (stratified)',
            notes=(
                'n_channels=3, n_classes=2, n_epochs=200, eegn_kern_size=65, '
                'tmin=-0.5, tmax=4.5-1/250, resample=250Hz, '
                'EarlyStoppingCallback(patience=50), seed=42. '
                'RegressionRemover fit on full dataset (data leakage caveat for final reporting).'
            ),
        )

    def prepare_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        set_global_seed()
        dataset = BNCICompIII3a(
            subject_ids=list(range(1, 10)),
            tmin=-0.5,
            tmax=4.5 - 1/250,
            include_eog=True,
        )
        X, y = dataset.get_data()
        subject_ids = dataset.metadata['moabb_metadata']['subject'].to_numpy()

        remover = RegressionRemover(
            artifact_channel_indices=BNCICompIII3a.EOG_INDICES,
            artifact_type='eog',
        )
        X = remover.fit_transform(X)
        return X, y, subject_ids

    def build_model(self,
                    eegn_kern_size: int = 65,
                    n_epochs: int = 200,
                    l2_scale: float = 2.0,
                    loss_scale: float = 2.0,
                    label_smoothing: float = 0.2,
                    lr: float = 1e-3,     
                    patience: int = 50
                        
                        ) -> NNWrapper:
        
        print(f"NO BUILD_MODEL: LABEL > {label_smoothing}")
        return NNWrapper(
            arch=EEGEncoderModel(
                n_channels=3,
                n_classes=2,
                eegn_kern_size=eegn_kern_size,
            ),
            config=TrainerConfig(
                n_epochs=n_epochs,
                l2_scale=l2_scale,
                lr=lr,
                loss_scale=loss_scale,
                label_smoothing=label_smoothing,
                logger=LoggerCallbackConfig(every_n_epochs=10),
                early_stopping=EarlyStoppingCallbackConfig(patience=patience),
            ),
        )

    def run(self, 
            model = None,
            X = None,
            y = None,
            subject_ids = None,
            test_ratio: int = 0.2
            ) -> EvaluationResult:
        
        if X is None and y is None and subject_ids is None:
            X, y, subject_ids = self.prepare_data()

        if model is None:
            model = self.build_model()


        result = evaluate_intra_subject_fixed_split(
            model=model,
            X=X,
            y=y,
            subject_ids=subject_ids,
            test_ratio=test_ratio
        )
        print(self)
        print(result)
        return result


if __name__ == '__main__':
    experiment = BNCI2014004EEGEncoderEOGExperiment()
    result = experiment.run()
