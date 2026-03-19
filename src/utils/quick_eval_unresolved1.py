# quick_eval.py — Kumar RiemannianSVM
import mne
mne.set_log_level('WARNING')
import logging
logging.getLogger('moabb').setLevel(logging.WARNING)

import numpy as np
from src.utils.setup_seed import set_global_seed
from src.evaluation import evaluate_cross_subject
from src.datasets.imagined_speech.kumar_imagined_speech import KumarImagedSpeech
from src.models.riemannian_svm import RiemannianSVM

if __name__ == '__main__':
    set_global_seed()

    print('Loading dataset...')
    dataset = KumarImagedSpeech(
        data_path='data/imagined_speech',
        task='digit',
        subject_ids=list(range(1, 24)),
    )
    X, y = dataset.get_data()
    subject_ids = dataset.metadata['subject_ids']

    print(f'X: {X.shape}')
    print(f'y unique: {np.unique(y)}')
    print(f'Subjects: {np.unique(subject_ids)}')
    print(f'Trials per subject: {np.unique(subject_ids, return_counts=True)[1]}')

    model = RiemannianSVM(band_mode=False)

    print('\n--- Cross-subject (LOSO) ---')
    result = evaluate_cross_subject(model, X, y, subject_ids, validation_ratio=0.0)
    print(f'Accuracy: {result.accuracy_mean:.3f} ± {result.accuracy_std:.3f}')
    for subj, acc in result.per_subject.items():
        print(f'  Subject {subj}: {acc:.3f}')