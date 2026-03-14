import mne
mne.set_log_level('WARNING')

import logging
logging.getLogger('moabb').setLevel(logging.WARNING)

import numpy as np
from src.evaluation import evaluate_intra_subject, evaluate_cross_subject, evaluate_intra_subject_fixed_split
from src.datasets.motor_imagery.bci_comp_iv_2a import BCICompIV2a
from src.models import CSPLDAModel, ShallowConvNet, EEGEncoderModel
from src.preprocessing.pipelines import bci_standard
from src.training.callbacks import EarlyStoppingCallback, CheckpointCallback, LoggerCallback

if __name__ == '__main__':
    print('Loading dataset...')
    dataset = BCICompIV2a(subject_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9], tmin=0.5, tmax=3.5)
    X, y = dataset.get_data()
    #X = bci_standard(sfreq=dataset.sfreq).fit_transform(X)

    callbacks = [
        LoggerCallback(every_n_epochs=10, metrics=['train_loss', 'val_loss', 'train_acc', 'val_acc']),
        EarlyStoppingCallback(patience=50),
        CheckpointCallback(checkpoint_dir='results/checkpoints/eeg_encoder', every_n_epochs=50),
    ]

    metadata = dataset.metadata['moabb_metadata']
    subject_ids = metadata['subject'].to_numpy()

    print(f'X: {X.shape}, subjects: {np.unique(subject_ids)}')

    #model = CSPLDAModel(n_components=4)
    #model = ShallowConvNet(n_channels=22, n_times=X.shape[2], n_classes=4, n_epochs=20, logging_indent='    ')
    model = EEGEncoderModel(
        n_channels=22,
        n_classes=4,
        n_epochs=200,
        eegn_kern_size=65, # To avoid warnings with c++: 'with even kernel lengths and odd dilation may require a zero-padded copy of the input be created'
        verbose=True,
        callbacks=callbacks
    )

    print('\n--- Intra-subject ---')
    #result = evaluate_intra_subject(model, X, y, subject_ids, save_dir=f'results/models/eeg_encoder_{model.n_epochs}ep')
    result = evaluate_intra_subject_fixed_split(model, X, y, subject_ids)#, save_dir=f'results/models/eeg_encoder_{model.n_epochs}ep')
    print(f'Accuracy: {result.accuracy_mean:.3f} ± {result.accuracy_std:.3f}')
    for subj, acc in result.per_subject.items():
        print(f'  Subject {subj}: {acc:.3f}')


    exit(1) # I just want intra to achive close to 86% first, then cross...


    print('\n--- Cross-subject (LOSO) ---')
    result = evaluate_cross_subject(model, X, y, subject_ids)
    print(f'Accuracy: {result.accuracy_mean:.3f} ± {result.accuracy_std:.3f}')
    for subj, acc in result.per_subject.items():
        print(f'  Subject {subj}: {acc:.3f}')
