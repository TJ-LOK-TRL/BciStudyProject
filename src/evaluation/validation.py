import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from src.evaluation.results import EvaluationResult, EvaluationType
from src.models.base_model import BaseModel
from src.training.callbacks import CheckpointCallback
from src.training.trainer import Trainer
from typing import Optional
import torch

# ─── helpers ─────────────────────────────────────────────────────────────────

def _get_checkpoint_dir(model: BaseModel) -> Optional[str]:
    """Extract checkpoint_dir from model callbacks if available."""
    if not hasattr(model, '_trainer'):
        return None
    trainer: Trainer = model._trainer 
    for cb in trainer.callbacks:
        if isinstance(cb, CheckpointCallback):
            return str(cb.checkpoint_dir)
    return None


def _set_run_id(model: BaseModel, run_id: str, checkpoint_dir: Optional[str]) -> None:
    """Set run_id on trainer if model has one — enables per-run checkpointing."""
    if checkpoint_dir and hasattr(model, '_trainer'):
        model._trainer.checkpoint_dir = Path(checkpoint_dir)
        model._trainer.set_run_id(run_id)


def _is_done(run_id: str, save_dir: Optional[str]) -> Optional[float]:
    """Return score if run already completed, None otherwise."""
    if save_dir is None:
        return None
    path = Path(save_dir) / f'{run_id}_done.pt'
    if path.exists():
        return float(torch.load(path, map_location='cpu')['score'])
    return None


def _mark_done(run_id: str, save_dir: Optional[str], score: float) -> None:
    """Mark a run as completed with its score."""
    if save_dir is None:
        return
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    torch.save({'score': score}, Path(save_dir) / f'{run_id}_done.pt')


# ─── evaluation functions ─────────────────────────────────────────────────────

def evaluate_intra_subject(
    model: BaseModel,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    n_splits: int = 5,
    save_dir: Optional[str] = None,
) -> EvaluationResult:
    checkpoint_dir = _get_checkpoint_dir(model)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    unique_subjects = np.unique(subject_ids)
    subject_scores = []

    for i, subj in enumerate(unique_subjects):
        mask = subject_ids == subj
        X_subj, y_subj = X[mask], y[mask]
        scores = []
        best_score = 0.0
        best_model = None

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_subj, y_subj)):
            run_id = f'{model.__class__.__name__}_subject{subj}_fold{fold}'
            print(f'  Subject {i+1}/{len(unique_subjects)} fold {fold+1}/{n_splits}...', flush=True)

            done_score = _is_done(run_id, save_dir)
            if done_score is not None:
                print(f'    Skipping — already done (score: {done_score:.3f})')
                scores.append(done_score)
                continue

            train_idx_local, val_idx_local = train_test_split(
                train_idx, 
                test_size=0.2, 
                stratify=y_subj[train_idx],
                random_state=42
            )

            cloned = model.clone()
            _set_run_id(cloned, run_id, checkpoint_dir)
            cloned.fit(
                X_subj[train_idx_local], y_subj[train_idx_local],
                X_val=X_subj[val_idx_local],
                y_val=y_subj[val_idx_local],
            )
            preds = cloned.predict(X_subj[test_idx])
            fold_score = float(np.mean(preds == y_subj[test_idx]))
            scores.append(fold_score)
            _mark_done(run_id, save_dir, fold_score)

            if fold_score > best_score:
                best_score = fold_score
                best_model = cloned

        if save_dir and best_model is not None:
            best_model.save(f'{save_dir}/{model.__class__.__name__}_subject{subj}_best.pt')

        subject_scores.append(float(np.mean(scores)))

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
    checkpoint_dir = _get_checkpoint_dir(model)
    unique_subjects = np.unique(subject_ids)
    subject_scores = []

    for i, subj in enumerate(unique_subjects):
        run_id = f'{model.__class__.__name__}_loso_subject{subj}'
        print(f'  Cross-subject LOSO: subject {i+1}/{len(unique_subjects)}...', flush=True)

        done_score = _is_done(run_id, save_dir)
        if done_score is not None:
            print(f'    Skipping — already done (score: {done_score:.3f})')
            subject_scores.append(done_score)
            continue

        test_mask = subject_ids == subj
        X_train_all = X[~test_mask]
        y_train_all = y[~test_mask]
        train_idx_loso, val_idx_loso = train_test_split(
            np.arange(len(X_train_all)), 
            test_size=0.1,       
            stratify=y_train_all,
            random_state=42
        )

        cloned = model.clone()
        _set_run_id(cloned, run_id, checkpoint_dir)
        cloned.fit(
            X_train_all[train_idx_loso], 
            y_train_all[train_idx_loso],
            X_val=X_train_all[val_idx_loso],
            y_val=y_train_all[val_idx_loso]
        )
        preds = cloned.predict(X[test_mask])
        score = float(np.mean(preds == y[test_mask]))
        subject_scores.append(score)
        _mark_done(run_id, save_dir, score)

        if save_dir:
            cloned.save(f'{save_dir}/{run_id}.pt')

    return EvaluationResult(
        evaluation=EvaluationType.CROSS_SUBJECT,
        accuracy_mean=float(np.mean(subject_scores)),
        accuracy_std=float(np.std(subject_scores)),
        per_subject={int(s): v for s, v in zip(unique_subjects, subject_scores)},
    )


def evaluate_intra_subject_fixed_split(
    model: BaseModel,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    train_ratio: float = 0.8,
    save_dir: Optional[str] = None,
) -> EvaluationResult:
    checkpoint_dir = _get_checkpoint_dir(model)
    unique_subjects = np.unique(subject_ids)
    subject_scores = []

    for i, subj in enumerate(unique_subjects):
        run_id = f'{model.__class__.__name__}_subject{subj}_fixed'
        print(f'  Intra-subject fixed split: subject {i+1}/{len(unique_subjects)}...', flush=True)

        done_score = _is_done(run_id, save_dir)
        if done_score is not None:
            print(f'    Skipping — already done (score: {done_score:.3f})')
            subject_scores.append(done_score)
            continue

        mask = subject_ids == subj
        X_subj, y_subj = X[mask], y[mask]

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_subj, y_subj, 
            test_size=1.0 - train_ratio,
            stratify=y_subj, 
            random_state=42
        )

        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train_full, y_train_full, 
            test_size=0.2, 
            stratify=y_train_full, 
            random_state=42
        )

        cloned = model.clone()
        _set_run_id(cloned, run_id, checkpoint_dir)
        cloned.fit(X_fit, y_fit, X_val=X_val, y_val=y_val)
        preds = cloned.predict(X_test)         
        score = float(np.mean(preds == y_test))
        subject_scores.append(score)
        _mark_done(run_id, save_dir, score)

        if save_dir:
            cloned.save(f'{save_dir}/{run_id}.pt')

    return EvaluationResult(
        evaluation=EvaluationType.INTRA_SUBJECT,
        accuracy_mean=float(np.mean(subject_scores)),
        accuracy_std=float(np.std(subject_scores)),
        per_subject={int(s): v for s, v in zip(unique_subjects, subject_scores)},
    )


