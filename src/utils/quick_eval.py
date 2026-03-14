import mne
mne.set_log_level('WARNING')

import logging
logging.getLogger('moabb').setLevel(logging.WARNING)

import numpy as np
from src.evaluation import evaluate_intra_subject, evaluate_cross_subject
from src.datasets.motor_imagery.bci_comp_iv_2a import BCICompIV2a
from src.models import CSPLDAModel, ShallowConvNet, EEGEncoderModel

if __name__ == '__main__':
    print('Loading dataset...')
    dataset = BCICompIV2a(subject_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    X, y = dataset.get_data()

    metadata = dataset.metadata['moabb_metadata']
    subject_ids = metadata['subject'].to_numpy()

    print(f'X: {X.shape}, subjects: {np.unique(subject_ids)}')

    #model = CSPLDAModel(n_components=4)
    #model = ShallowConvNet(n_channels=22, n_times=X.shape[2], n_classes=4, n_epochs=20, logging_indent='    ')
    model = EEGEncoderModel(
        n_channels=22,
        n_classes=4,
        n_epochs=500,
        eegn_kern_size=65, # To avoid warnings with c++: 'with even kernel lengths and odd dilation may require a zero-padded copy of the input be created'
        verbose=True,
    )

    print('\n--- Intra-subject ---')
    result = evaluate_intra_subject(model, X, y, subject_ids, save_dir=f'results/models/eeg_encoder_{model.n_epochs}ep')
    print(f'Accuracy: {result.accuracy_mean:.3f} ± {result.accuracy_std:.3f}')
    for subj, acc in result.per_subject.items():
        print(f'  Subject {subj}: {acc:.3f}')


    exit(1)


    print('\n--- Cross-subject (LOSO) ---')
    result = evaluate_cross_subject(model, X, y, subject_ids)
    print(f'Accuracy: {result.accuracy_mean:.3f} ± {result.accuracy_std:.3f}')
    for subj, acc in result.per_subject.items():
        print(f'  Subject {subj}: {acc:.3f}')
