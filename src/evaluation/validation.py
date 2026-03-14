import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.evaluation.results import EvaluationResult, EvaluationType
from src.models.base_model import BaseModel
from typing import Optional


def evaluate_intra_subject(
    model: BaseModel,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    n_splits: int = 5,
    save_dir: Optional[str] = None,
) -> EvaluationResult:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    unique_subjects = np.unique(subject_ids)

    subject_scores = []
    for i, subj in  enumerate(unique_subjects):
        print(f'  Intra-subject: subject {i}/{len(unique_subjects)}...', flush=True)
        mask = subject_ids == subj
        X_subj, y_subj = X[mask], y[mask]

        scores = []
        best_score = 0.0
        best_model = None

        for train_idx, test_idx in skf.split(X_subj, y_subj):
            cloned = model.clone()
            cloned.fit(X_subj[train_idx], y_subj[train_idx])
            preds = cloned.predict(X_subj[test_idx])
            fold_score = float(np.mean(preds == y_subj[test_idx]))
            scores.append(fold_score)

            # save best fold per subject
            if save_dir and fold_score > best_score:
                best_score = fold_score
                best_model = cloned

        if save_dir and best_model is not None:
            path = f'{save_dir}/{model.__class__.__name__}_subject{subj}_best.pt'
            best_model.save(path)

        subject_scores.append(float(np.mean(scores)))

    unique_subjects = np.unique(subject_ids)
    return EvaluationResult(
        evaluation=EvaluationType.INTRA_SUBJECT,
        accuracy_mean=float(np.mean(subject_scores)),
        accuracy_std=float(np.std(subject_scores)),
        per_subject={int(s): v for s, v in zip(unique_subjects, subject_scores)},
    )

def evaluate_cross_subject(
    model: BaseModel,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    save_dir: Optional[str] = None,
) -> EvaluationResult:
    subject_scores = []
    unique_subjects = np.unique(subject_ids)

    for i, subj in enumerate(unique_subjects):
        print(f'  Cross-subject LOSO: subject {i}/{len(unique_subjects)}...', flush=True)
        test_mask = subject_ids == subj
        cloned = model.clone()
        cloned.fit(X[~test_mask], y[~test_mask])            # train without the subject
        preds = cloned.predict(X[test_mask])                # test only in the subject
        subject_scores.append(float(np.mean(preds == y[test_mask])))

        if save_dir:
            path = f'{save_dir}/{model.__class__.__name__}_loso_subject{subj}.pt'
            cloned.save(path)

    return EvaluationResult(
        evaluation=EvaluationType.CROSS_SUBJECT,
        accuracy_mean=float(np.mean(subject_scores)),
        accuracy_std=float(np.std(subject_scores)),
        per_subject={int(s): v for s, v in zip(unique_subjects, subject_scores)},
    )