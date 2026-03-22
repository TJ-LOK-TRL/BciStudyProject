# EEG Dataset Contract

All datasets in this project inherit from `BaseDataset` and guarantee the
following interface. Any new dataset must comply with this contract.

---

## Common Output Format

After calling `dataset.get_data()`, the following is guaranteed:

### X — EEG Signal
```
type:  np.ndarray
dtype: float32
shape: (n_trials, n_channels, n_times)
units: microvolts (µV) — raw, no normalisation applied
```

### y — Labels
```
type:  np.ndarray
dtype: str  (e.g. 'left_hand', 'right_hand', 'm', 'sh')
shape: (n_trials,)
```

### metadata — Session Info
```
type: dict
required key:
  subject_ids: accessible via dataset.subject_ids_array → np.ndarray (n_trials,)

optional keys (dataset-dependent):
  n_subjects:  int
  session_ids: np.ndarray (n_trials,)
  + any dataset-specific fields
```

---

## Dataset Properties (always available)

| Property             | Type       | Description                                  |
|----------------------|------------|----------------------------------------------|
| `sfreq`              | float      | Sampling frequency in Hz                     |
| `n_channels`         | int        | Number of EEG channels after loading         |
| `n_classes`          | int        | Number of distinct classes                   |
| `class_names`        | list[str]  | Human-readable class labels                  |
| `subject_ids`        | list[int]  | Subject IDs passed to constructor            |
| `subject_ids_array`  | np.ndarray | Per-trial subject IDs — shape (n_trials,)    |

---

## Canonical Assembly
```python
dataset = SomeDataset(subject_ids=[1, 2, 3], ...)
X, y = dataset.get_data()                    # triggers load() + preprocess()
subject_ids = dataset.subject_ids_array      # np.ndarray (n_trials,) — always works

# X:           (n_trials, n_channels, n_times)  float32
# y:           (n_trials,)                       str
# subject_ids: (n_trials,)                       int
```

---

## Normalisation Policy

**No normalisation is applied by the dataset.**

All preprocessing — bandpass filtering, z-score normalisation, EOG removal,
resampling — is the responsibility of the external preprocessing pipeline,
applied after `get_data()` and before passing data to models.

---

## Shape Consistency Guarantees

- All trials have **identical shape** `(n_channels, n_times)`
- `len(X) == len(y) == len(dataset.subject_ids_array)` always
- `n_times` is fixed per dataset instance
- `n_channels` matches `dataset.n_channels`

---

## What the Contract Does NOT Guarantee

- Class balance across subjects or sessions
- Equal number of trials per subject
- Absence of artefacts — use the preprocessing pipeline
- Specific channel ordering beyond what the source data provides

---

## Implementing a New Dataset

Subclass `BaseDataset` and implement the following:
```python
class MyDataset(BaseDataset):

    @property
    def n_classes(self) -> int: ...

    @property
    def class_names(self) -> List[str]: ...

    @property
    def n_channels(self) -> int: ...

    @property
    def sfreq(self) -> float: ...

    def load(self) -> None:
        # populate self.X, self.y, self.metadata['subject_ids']
        ...

    def preprocess(self) -> None:
        # optional — leave as pass if no preprocessing needed
        ...
```