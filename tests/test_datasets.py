# tests/test_datasets.py
"""
Contract compliance tests for all BaseDataset subclasses.
Verifies that every dataset satisfies the interface defined in docs/data_contract.md.
"""
import pytest
import numpy as np
from src.datasets.base_dataset import BaseDataset 

# ─── contract checker ────────────────────────────────────────────────────────

def assert_dataset_contract(dataset: BaseDataset, n_subjects_expected: int = None):
    """
    Assert that a dataset satisfies the BaseDataset contract.
    Call after dataset.get_data() has been called.
    """
    X, y = dataset.get_data()
    subject_ids = dataset.subject_ids_array

    # shape
    assert X.ndim == 3, f'X must be 3D (n_trials, n_channels, n_times), got {X.ndim}D'
    n_trials, n_channels, n_times = X.shape

    # dtype
    assert X.dtype == np.float32, f'X must be float32, got {X.dtype}'

    # y
    assert y.ndim == 1, f'y must be 1D, got {y.ndim}D'
    assert len(y) == n_trials, f'len(y)={len(y)} != n_trials={n_trials}'

    # subject_ids
    assert subject_ids.ndim == 1, f'subject_ids_array must be 1D'
    assert len(subject_ids) == n_trials, \
        f'len(subject_ids)={len(subject_ids)} != n_trials={n_trials}'

    # consistency
    assert len(X) == len(y) == len(subject_ids), \
        'len(X), len(y), len(subject_ids_array) must all be equal'

    # properties
    assert dataset.sfreq > 0, f'sfreq must be > 0, got {dataset.sfreq}'
    assert dataset.n_channels > 0, f'n_channels must be > 0, got {dataset.n_channels}'
    assert dataset.n_classes > 0, f'n_classes must be > 0, got {dataset.n_classes}'
    assert len(dataset.class_names) == dataset.n_classes, \
        f'len(class_names)={len(dataset.class_names)} != n_classes={dataset.n_classes}'

    # n_channels consistency
    assert n_channels == dataset.n_channels, \
        f'X.shape[1]={n_channels} != dataset.n_channels={dataset.n_channels}'

    # all trials same shape — guaranteed by np.stack so just verify n_times > 0
    assert n_times > 0, f'n_times must be > 0, got {n_times}'

    # y labels match class_names
    unique_labels = set(y.tolist())
    declared_labels = set(dataset.class_names)
    assert unique_labels.issubset(declared_labels), \
        f'y contains labels not in class_names: {unique_labels - declared_labels}'

    # subject_ids subset of declared subject_ids
    unique_subjects = set(subject_ids.tolist())
    declared_subjects = set(dataset.subject_ids)
    assert unique_subjects.issubset(declared_subjects), \
        f'subject_ids_array contains subjects not in dataset.subject_ids: ' \
        f'{unique_subjects - declared_subjects}'

    if n_subjects_expected is not None:
        assert len(unique_subjects) == n_subjects_expected, \
            f'expected {n_subjects_expected} subjects, got {len(unique_subjects)}'

    print(f'  ✓ {dataset.__class__.__name__}: '
          f'X={X.shape}, sfreq={dataset.sfreq}Hz, '
          f'classes={dataset.class_names}, subjects={sorted(unique_subjects)}')


# ─── Motor Imagery ────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestBCICompIV2a:

    def test_contract_no_eog(self):
        from src.datasets.motor_imagery.bci_comp_iv_2a import BCICompIV2a
        dataset = BCICompIV2a(subject_ids=[1, 2], tmin=0.5, tmax=3.5)
        assert_dataset_contract(dataset, n_subjects_expected=2)

    def test_contract_with_eog(self):
        from src.datasets.motor_imagery.bci_comp_iv_2a import BCICompIV2a
        dataset = BCICompIV2a(subject_ids=[1], tmin=-0.5, tmax=4.0, include_eog=True)
        X, y = dataset.get_data()
        # with EOG: 22 EEG + 3 EOG = 25 channels
        assert X.shape[1] == 25, f'Expected 25 channels with EOG, got {X.shape[1]}'

    def test_subject_ids_array(self):
        from src.datasets.motor_imagery.bci_comp_iv_2a import BCICompIV2a
        dataset = BCICompIV2a(subject_ids=[1, 2], tmin=0.5, tmax=3.5)
        dataset.get_data()
        ids = dataset.subject_ids_array
        assert set(ids.tolist()) == {1, 2}

    def test_float32(self):
        from src.datasets.motor_imagery.bci_comp_iv_2a import BCICompIV2a
        dataset = BCICompIV2a(subject_ids=[1], tmin=0.5, tmax=3.5)
        X, _ = dataset.get_data()
        assert X.dtype == np.float32

@pytest.mark.slow
class TestBNCICompIII3a:

    def test_contract(self):
        from src.datasets.motor_imagery.bnci2014_004 import BNCICompIII3a
        dataset = BNCICompIII3a(subject_ids=[1, 2])
        assert_dataset_contract(dataset, n_subjects_expected=2)

    def test_contract_with_eog(self):
        from src.datasets.motor_imagery.bnci2014_004 import BNCICompIII3a
        dataset = BNCICompIII3a(subject_ids=[1], include_eog=True)
        X, _ = dataset.get_data()
        assert X.shape[1] == 6, f'Expected 6 channels with EOG, got {X.shape[1]}'

@pytest.mark.slow
class TestCho2017:

    def test_contract(self):
        from src.datasets.motor_imagery.cho2017 import Cho2017
        dataset = Cho2017(subject_ids=[1, 2])
        assert_dataset_contract(dataset, n_subjects_expected=2)


# ─── Imagined Speech ──────────────────────────────────────────────────────────

class TestFEIS:

    DATA_PATH = 'data/imagined_speech/scottwellington-FEIS-7e726fd/experiments'

    def test_contract_2classes(self):
        from src.datasets.imagined_speech.feis import FEIS
        dataset = FEIS(
            data_path=self.DATA_PATH,
            subject_ids=[1, 2],
            phase='thinking',
            labels=['m', 'sh'],
        )
        assert_dataset_contract(dataset, n_subjects_expected=2)

    def test_contract_all_phonemes(self):
        from src.datasets.imagined_speech.feis import FEIS
        dataset = FEIS(
            data_path=self.DATA_PATH,
            subject_ids=[1],
            phase='thinking',
        )
        X, y = dataset.get_data()
        assert dataset.n_classes == 16
        assert X.shape == (160, 14, 1280)

    def test_trials_per_subject(self):
        from src.datasets.imagined_speech.feis import FEIS
        dataset = FEIS(
            data_path=self.DATA_PATH,
            subject_ids=[1, 2],
            phase='thinking',
            labels=['m', 'n', 's', 'f'],
        )
        X, y = dataset.get_data()
        ids = dataset.subject_ids_array
        for subj in [1, 2]:
            n = (ids == subj).sum()
            assert n == 40, f'Subject {subj}: expected 40 trials, got {n}'


class TestKumarImagedSpeech:

    DATA_PATH = 'data/imagined_speech'

    def test_contract_digit(self):
        from src.datasets.imagined_speech.kumar_imagined_speech import KumarImagedSpeech
        dataset = KumarImagedSpeech(
            data_path=self.DATA_PATH,
            task='digit',
            subject_ids=[1, 2],
        )
        assert_dataset_contract(dataset, n_subjects_expected=2)

    def test_contract_character(self):
        from src.datasets.imagined_speech.kumar_imagined_speech import KumarImagedSpeech
        dataset = KumarImagedSpeech(
            data_path=self.DATA_PATH,
            task='character',
            subject_ids=[1],
        )
        X, y = dataset.get_data()
        assert dataset.n_classes == 26

    def test_n_channels(self):
        from src.datasets.imagined_speech.kumar_imagined_speech import KumarImagedSpeech
        dataset = KumarImagedSpeech(
            data_path=self.DATA_PATH,
            task='digit',
            subject_ids=[1],
        )
        X, _ = dataset.get_data()
        assert X.shape[1] == 14, f'Expected 14 channels, got {X.shape[1]}'