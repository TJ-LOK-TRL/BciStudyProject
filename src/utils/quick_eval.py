import mne
mne.set_log_level('WARNING')
import logging
logging.getLogger('moabb').setLevel(logging.WARNING)

import numpy as np
from src.utils.setup_seed import set_global_seed
from src.evaluation import evaluate_intra_subject_fixed_split
from src.datasets.motor_imagery.cho2017 import Cho2017
from src.models import EEGEncoderModel
from src.training.callbacks import EarlyStoppingCallback, LoggerCallback

if __name__ == '__main__':
    set_global_seed()

    print('Loading dataset...')
    dataset = Cho2017(subject_ids=list(range(1, 10)), tmin=0.0, tmax=3.0)
    X, y = dataset.get_data()
    subject_ids = dataset.metadata['moabb_metadata']['subject'].to_numpy()

    print(f'X: {X.shape}, subjects: {np.unique(subject_ids)}')

    callbacks = [
        LoggerCallback(every_n_epochs=10, metrics=['train_loss', 'val_loss', 'train_acc', 'val_acc']),
        EarlyStoppingCallback(patience=50),
    ]

    model = EEGEncoderModel(
        n_channels=64,
        n_classes=2,
        n_epochs=200,
        eegn_kern_size=65,
        callbacks=callbacks,
    )

    print('\n--- Intra-subject ---')
    result = evaluate_intra_subject_fixed_split(
        model, X, y, subject_ids, test_ratio=0.2
    )
    print(f'Accuracy: {result.accuracy_mean:.3f} ± {result.accuracy_std:.3f}')
    for subj, acc in result.per_subject.items():
        print(f'  Subject {subj}: {acc:.3f}')