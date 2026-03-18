import numpy as np
from src.experiments.base_experiment import BaseExperiment, ExperimentConfig
from src.datasets.motor_imagery.bci_comp_iv_2a import BCICompIV2a
from src.models.riemannian_svm import RiemannianSVM
from src.evaluation.validation import evaluate_intra_subject_fixed_split
from src.evaluation.results import EvaluationResult
from src.preprocessing.transforms.filtering import FilterBankTransform  

class BCI2aRiemannianSVMExperiment(BaseExperiment):
    """
    Result: 75.8% ± 14.1%
    Subject scores: [85.0, 58.4, 90.2, 68.2, 61.3, 54.3, 90.8, 85.5, 88.4]

    Best result so far on BCI Competition IV 2a.
    RiemannianSVM outperforms EEGEncoder (70%) and CSP+LDA (62.5%).
    Subjects 2, 5, 6 are consistently difficult across all models.
    """

    @property
    def config(self) -> ExperimentConfig:
        return ExperimentConfig(
            name='bci2a_riemannian_svm',
            dataset='BCI Competition IV 2a (9 subjects, 4 classes)',
            model='RiemannianSVM (filter bank 6 bands, OAS covariance, tangent space, SVM rbf)',
            preprocessing='None — RiemannianSVM filters internally per band',
            evaluation='intra-subject fixed split 80/20 (stratified)',
            notes='Raw X passed — no external preprocessing applied',
        )

    def prepare_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dataset = BCICompIV2a(
            subject_ids=list(range(1, 10)),
            tmin=0.5,
            tmax=3.5,
        )
        X, y = dataset.get_data()
        X = FilterBankTransform(bands=[(8,  12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)], sfreq=dataset.sfreq).fit_transform(X)
        subject_ids = dataset.metadata['moabb_metadata']['subject'].to_numpy()
        return X, y, subject_ids

    def build_model(self) -> RiemannianSVM:
        return RiemannianSVM()

    def run(self) -> EvaluationResult:
        X, y, subject_ids = self.prepare_data()
        model = self.build_model()
        result = evaluate_intra_subject_fixed_split(
            model, 
            X, 
            y, 
            subject_ids, 
            test_ratio=0.3, 
            validation_ratio=0.0 # Not deep learning
        )

        print(self)
        print(f'Accuracy: {result.accuracy_mean:.3f} ± {result.accuracy_std:.3f}')
        for subj, acc in result.per_subject.items():
            print(f'  Subject {subj}: {acc:.3f}')

        return result


if __name__ == '__main__':
    experiment = BCI2aRiemannianSVMExperiment()
    result = experiment.run()