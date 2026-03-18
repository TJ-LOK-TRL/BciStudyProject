import numpy as np
from src.experiments.base_experiment import BaseExperiment, ExperimentConfig
from src.datasets.imagined_speech.feis import FEIS
from src.models.riemannian_svm import RiemannianSVM
from src.evaluation.validation import evaluate_intra_subject_fixed_split
from src.evaluation.results import EvaluationResult
from src.preprocessing.transforms.filtering import FilterBankTransform
from src.utils.setup_seed import set_global_seed


class FEISRiemannianSVMExperiment(BaseExperiment):
    """
    Result: 56.3% ± 21.2% (intra-subject, 2 classes)
    Subject scores: [25.0, 50.0, 50.0, 50.0, 50.0, 75.0, 75.0, 75.0, 75.0,
                     50.0, 25.0, 33.3, 50.0, 75.0, 25.0, 50.0, 100.0, 75.0,
                     25.0, 75.0, 75.0]

    Baseline result for FEIS imagined speech dataset.
    High variance due to very few trials (20 per class per subject).
    Results barely above chance (50%) — dataset too small for robust classification.

    Other configurations tested:
        4 classes: 22.8% (chance 25%) — below chance
        16 classes: 4.6% (chance 6.3%) — below chance

    Key finding: RiemannianSVM not suitable for imagined speech with <20 trials/class.
    Deep learning models (EEGEncoder) may perform better with data augmentation.
    """

    DATA_PATH = 'data/imagined_speech/scottwellington-FEIS-7e726fd/experiments'

    @property
    def config(self) -> ExperimentConfig:
        return ExperimentConfig(
            name='feis_riemannian_svm_2class',
            dataset='FEIS (21 subjects, 2 phonemes: m vs sh)',
            model='RiemannianSVM (filter bank 6 bands, OAS covariance, tangent space, SVM rbf)',
            preprocessing='FilterBankTransform (8-32Hz, 6 bands, butter order 5)',
            evaluation='intra-subject fixed split, test_ratio=0.2, validation_ratio=0.0',
            notes=(
                'Only 20 trials per class per subject — severely data-limited. '
                'High variance expected. '
                'Chance level: 50%. '
                'Phase: thinking (imagined speech only).'
            ),
        )

    def prepare_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        set_global_seed()
        dataset = FEIS(
            data_path=self.DATA_PATH,
            subject_ids=list(range(1, 22)),
            phase='thinking',
            labels=['m', 'sh'],
        )
        X, y = dataset.get_data()
        subject_ids = dataset.metadata['subject_ids']
        X_fb = FilterBankTransform(
            bands=[(8,12),(12,16),(16,20),(20,24),(24,28),(28,32)],
            sfreq=dataset.sfreq,
        ).fit_transform(X)
        return X_fb, y, subject_ids

    def build_model(self) -> RiemannianSVM:
        return RiemannianSVM(band_mode=True)

    def run(self) -> EvaluationResult:
        X, y, subject_ids = self.prepare_data()
        model = self.build_model()
        result = evaluate_intra_subject_fixed_split(
            model, X, y, subject_ids,
            test_ratio=0.2,
            validation_ratio=0.0,
        )
        print(self)
        print(result)
        return result


if __name__ == '__main__':
    experiment = FEISRiemannianSVMExperiment()
    result = experiment.run()