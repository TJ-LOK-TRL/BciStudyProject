import numpy as np
from src.experiments.base_experiment import BaseExperiment, ExperimentConfig
from src.datasets.motor_imagery.bci_comp_iv_2a import BCICompIV2a
from src.input_adapters.cnn2d_adapter import CNN2DAdapter
from src.models.shallow_convnet import ShallowConvNet
from src.evaluation.validation import evaluate_intra_subject_fixed_split
from src.evaluation.results import EvaluationResult
from src.models.wrappers.nn_wrapper import NNWrapper
from src.preprocessing.transforms.artifacts import RegressionRemover
from src.training.callbacks.configs import EarlyStoppingCallbackConfig, LoggerCallbackConfig
from src.training.trainer_config import TrainerConfig
from src.utils.setup_seed import set_global_seed
from src.evaluation.reporting import generate_report
from src.datasets.base_dataset import BaseDataset


class BCI2aShallowConvNetEOGExperiment(BaseExperiment):
    """
    Result: TBD
    ShallowConvNet on BCI Competition IV 2a with EOG regression removal.

    Comparison:
    - EEGEncoder no EOG:    76.1%
    - EEGEncoder + EOG:     85.0%
    - ShallowConvNet + EOG: TBD  ← this experiment
    """

    EOG_INDICES = [22, 23, 24]

    @property
    def config(self) -> ExperimentConfig:
        return ExperimentConfig(
            name='bci2a_shallow_convnet_eog_regression',
            dataset='BCI Competition IV 2a (9 subjects, 4 classes)',
            model='ShallowConvNet',
            preprocessing='EOG regression removal (channels 22,23,24) — fit on full dataset',
            evaluation='intra-subject fixed split, test_ratio=0.2 (stratified)',
            notes=(
                'tmin=-0.5, tmax=4.0-1/250 (1125 samples), '
                'include_eog=True, RegressionRemover fit on full X (data leakage caveat), '
                'EarlyStoppingCallback(patience=50), seed=42'
            ),
        )

    def prepare_data(self) -> tuple[BaseDataset, np.ndarray, np.ndarray, np.ndarray]:
        set_global_seed()
        dataset = BCICompIV2a(
            subject_ids=list(range(1, 10)),
            tmin=-0.5,
            tmax=4.0 - 1/250,
            include_eog=True,
        )
        X, y = dataset.get_data()
        subject_ids = dataset.subject_ids_array

        remover = RegressionRemover(
            artifact_channel_indices=self.EOG_INDICES,
            artifact_type='eog',
        )
        X = remover.fit_transform(X)
        return dataset, X, y, subject_ids

    def build_model(self, n_times: int) -> NNWrapper:
        return NNWrapper(
            arch=ShallowConvNet(
                n_channels=22,
                n_times=n_times,
                n_classes=4,
            ),
            config=TrainerConfig(
                n_epochs=1,
                grad_clip=0.0,
                input_adapter=CNN2DAdapter(),
                logger=LoggerCallbackConfig(every_n_epochs=10),
                early_stopping=EarlyStoppingCallbackConfig(patience=50),
            ),
        )

    def run(self) -> EvaluationResult:
        dataset, X, y, subject_ids = self.prepare_data()
        model = self.build_model(n_times=X.shape[2])

        result = evaluate_intra_subject_fixed_split(
            model=model,
            X=X,
            y=y,
            subject_ids=subject_ids,
            test_ratio=0.2,
        )
        print(self)
        print(result)

        generate_report(
            result=result,
            dataset=dataset,
            model=model,
            preprocessing='EOG regression removal (channels 22,23,24)',
            output_dir='reports',
            notes='ShallowConvNet + EOG removal on BCI2a.',
        )

        return result


if __name__ == '__main__':
    experiment = BCI2aShallowConvNetEOGExperiment()
    result = experiment.run()