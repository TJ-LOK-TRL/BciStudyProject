# quick_eval.py — FEIS RiemannianSVM
import mne
mne.set_log_level('WARNING')
import logging
logging.getLogger('moabb').setLevel(logging.WARNING)

import numpy as np
from src.utils.setup_seed import set_global_seed
from src.evaluation import evaluate_intra_subject_fixed_split
from src.datasets.imagined_speech.feis import FEIS
from src.models.riemannian_svm import RiemannianSVM
from src.preprocessing.transforms.filtering import FilterBankTransform

DATA_PATH = 'data/imagined_speech/scottwellington-FEIS-7e726fd/experiments'

if __name__ == '__main__':
    set_global_seed()

    print('Loading dataset...')
    dataset = FEIS(
        data_path=DATA_PATH,
        subject_ids=list(range(1, 22)),
        phase='thinking',
        labels=['m', 'sh'],
    )
    X, y = dataset.get_data()
    subject_ids = dataset.metadata['subject_ids']

    print(f'X: {X.shape}')
    print(f'y unique: {np.unique(y)}')
    print(f'Subjects: {np.unique(subject_ids)}')
    print(f'Trials per subject: {len(X) // len(np.unique(subject_ids))}')

    X_fb = FilterBankTransform(
        bands=[(8,12),(12,16),(16,20),(20,24),(24,28),(28,32)],
        sfreq=dataset.sfreq,
    ).fit_transform(X)

    model = RiemannianSVM(band_mode=True)

    print('\n--- Intra-subject (fixed split) ---')
    result = evaluate_intra_subject_fixed_split(
        model, X_fb, y, subject_ids,
        test_ratio=0.2,
        validation_ratio=0.0,
    )
    print(f'Accuracy: {result.accuracy_mean:.3f} ± {result.accuracy_std:.3f}')
    for subj, acc in result.per_subject.items():
        print(f'  Subject {subj}: {acc:.3f}')