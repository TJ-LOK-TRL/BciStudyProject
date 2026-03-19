# quick_eval.py — Cho2017 EEGEncoder
import mne
mne.set_log_level('WARNING')
import logging
logging.getLogger('moabb').setLevel(logging.WARNING)

import numpy as np
from src.utils.setup_seed import set_global_seed
from src.evaluation import evaluate_intra_subject_fixed_split
from src.datasets.motor_imagery import Cho2017
from src.models.eeg_encoder import EEGEncoderModel
from src.models.wrappers.nn_wrapper import NNWrapper
from src.training.trainer_config import TrainerConfig
from src.training.callbacks import LoggerCallbackConfig, EarlyStoppingCallbackConfig

MI_CHANNELS = [
    'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C5',  'C3',  'C1',  'Cz',  'C2',  'C4',  'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
]

if __name__ == '__main__':
    set_global_seed()

    print('Loading dataset...')
    dataset = Cho2017(
        subject_ids=list(range(1, 20)),
        tmin=0.0,
        tmax=3.0,
        channels=MI_CHANNELS,
    )
    X, y = dataset.get_data()
    subject_ids = dataset.metadata['moabb_metadata']['subject'].to_numpy()

    print(f'X: {X.shape}, subjects: {np.unique(subject_ids)}')

    model = NNWrapper(
        arch=EEGEncoderModel(
            n_channels=len(MI_CHANNELS),
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

    print('\n--- Intra-subject ---')
    result = evaluate_intra_subject_fixed_split(
        model, X, y, subject_ids, test_ratio=0.2
    )
    print(f'Accuracy: {result.accuracy_mean:.3f} ± {result.accuracy_std:.3f}')
    for subj, acc in result.per_subject.items():
        print(f'  Subject {subj}: {acc:.3f}')