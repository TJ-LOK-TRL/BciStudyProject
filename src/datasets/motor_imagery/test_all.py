"""Quick smoke test — verifies all datasets load correctly with 1 subject."""
import mne
mne.set_log_level('WARNING')
import logging
logging.getLogger('moabb').setLevel(logging.WARNING)

from src.datasets.motor_imagery.bci_comp_iv_2a import BCICompIV2a
from src.datasets.motor_imagery.physionet import PhysionetMI
from src.datasets.motor_imagery.cho2017 import Cho2017
from src.datasets.motor_imagery.bnci2014_002 import BNCIHorizon2014002
from src.datasets.motor_imagery.bnci2014_004 import BNCICompIII3a
from src.datasets.motor_imagery.schirrmeister2017 import Schirrmeister2017
from src.datasets.motor_imagery.lee2019_mi import Lee2019MI
from src.datasets.motor_imagery.stieger2021 import Stieger2021
from src.datasets.motor_imagery.bnci2015_001 import BNCI2015001

DATASETS = [
    BCICompIV2a(subject_ids=[1]),
    PhysionetMI(subject_ids=[1]),
    Cho2017(subject_ids=[1]),
    BNCIHorizon2014002(subject_ids=[1]),
    BNCICompIII3a(subject_ids=[1]),
    Schirrmeister2017(subject_ids=[1]),
    Lee2019MI(subject_ids=[1]),
    Stieger2021(subject_ids=[1]),
    BNCI2015001(subject_ids=[1]),
]

for dataset in DATASETS:
    try:
        X, y = dataset.get_data()
        print(f'✅ {dataset.__class__.__name__}: X={X.shape}, classes={list(set(y))}')
    except Exception as e:
        print(f'❌ {dataset.__class__.__name__}: {e}')