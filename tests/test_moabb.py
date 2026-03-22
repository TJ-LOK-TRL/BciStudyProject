import os
import mne

print("=== MNE_DATA ===")
print("os.environ:  ", os.environ.get("MNE_DATA", "NÃO DEFINIDA"))
print("mne.config:  ", mne.get_config("MNE_DATA"))

print("\n=== A carregar dataset MOABB (BCI Competition IV 2a, sujeito 1) ===")
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

dataset = BNCI2014_001()
paradigm = MotorImagery()

print("A fazer download/load...")
X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[1])

print("\n=== Sucesso! ===")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Classes:", set(y))