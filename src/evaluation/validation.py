import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from src.evaluation.results import EvaluationResult, EvaluationType, SubjectResult, SplitType
from src.models.core import ITrainableModel 
from src.models.wrappers.nn_wrapper import NNWrapper
from typing import Optional, Tuple
import torch


# ─── helpers ─────────────────────────────────────────────────────────────────

def _get_model_name(model: ITrainableModel) -> str:
    """Get a stable model name for run_id — uses arch name for NNWrapper."""
    return model.name


def _get_checkpoint_dir(model: ITrainableModel) -> Optional[str]:
    """Extract checkpoint_dir from NNWrapper trainer if available."""
    if not isinstance(model, NNWrapper):
        return None
    if model._trainer.checkpoint_dir is None:
        return None
    return str(model._trainer.checkpoint_dir)


def _set_run_id(model: ITrainableModel, run_id: str, checkpoint_dir: Optional[str]) -> None:
    """Set run_id on trainer if model is NNWrapper — enables per-run checkpointing."""
    if checkpoint_dir and isinstance(model, NNWrapper):
        model._trainer.checkpoint_dir = Path(checkpoint_dir)
        model._trainer.set_run_id(run_id)


def _is_done(run_id: str, save_dir: Optional[str]) -> Optional[float]:
    """Return score if run already completed, None otherwise."""
    if save_dir is None:
        return None
    path = Path(save_dir) / f'{run_id}_done.pt'
    if path.exists():
        return float(torch.load(path, map_location='cpu', weights_only=True)['score'])
    return None


def _mark_done(run_id: str, save_dir: Optional[str], score: float) -> None:
    """Mark a run as completed with its score."""
    if save_dir is None:
        return
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    torch.save({'score': score}, Path(save_dir) / f'{run_id}_done.pt')


def _apply_scaling(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit scaler on train, apply to both — kept generic, no ChannelScaler import.
    Caller decides whether to apply scaling via apply_scaling=True.
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    n_trials, n_channels, n_times = X_train.shape
    # reshape to (n_trials * n_times, n_channels) for StandardScaler
    X_train_2d = X_train.transpose(0, 2, 1).reshape(-1, n_channels)
    X_test_2d = X_test.transpose(0, 2, 1).reshape(-1, n_channels)
    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(n_trials, n_times, n_channels).transpose(0, 2, 1)
    n_test = X_test.shape[0]
    X_test_scaled = scaler.transform(X_test_2d).reshape(n_test, n_times, n_channels).transpose(0, 2, 1)
    return X_train_scaled, X_test_scaled


# ─── core execution engine ───────────────────────────────────────────────────

def _execute_run(
    model: ITrainableModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    run_id: str,
    save_dir: Optional[str],
    validation_ratio: float = 0.2,
    apply_scaling: bool = False,
    save_model: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray, Optional[ITrainableModel]]:
    """
    Central execution engine.
    Returns (score, y_pred, y_test, fitted_model)
    """
    done_score = _is_done(run_id, save_dir)
    if done_score is not None:
        print(f'    Skipping {run_id} — already done (score: {done_score:.3f})')
        return done_score, np.array([]), np.array([]), None

    if apply_scaling:
        X_train, X_test = _apply_scaling(X_train, X_test)

    cloned = model.clone()
    checkpoint_dir = _get_checkpoint_dir(model)
    _set_run_id(cloned, run_id, checkpoint_dir)

    if validation_ratio > 0:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train, y_train,
            test_size=validation_ratio,
            stratify=y_train,
            random_state=42,
        )
        cloned.fit(X_fit, y_fit, X_val=X_val, y_val=y_val)
    else:
        cloned.fit(X_train, y_train)

    y_pred = cloned.predict(X_test)
    score = float(np.mean(y_pred == y_test))

    _mark_done(run_id, save_dir, score)

    if save_dir and save_model:
        ext = 'pt' if isinstance(cloned, NNWrapper) else 'pkl'
        cloned.save(f'{save_dir}/{run_id}.{ext}')

    return score, y_pred, y_test, cloned


# ─── evaluation functions ─────────────────────────────────────────────────────

def evaluate_intra_subject(
    model: ITrainableModel,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    n_splits: int = 5,
    save_dir: Optional[str] = None,
) -> EvaluationResult:
    """Intra-subject evaluation using K-Fold Cross-Validation."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    unique_subjects = np.unique(subject_ids)
    subject_scores = {}
    subject_results = {}
    

    for i, subj in enumerate(unique_subjects):
        mask = subject_ids == subj
        X_subj, y_subj = X[mask], y[mask]
        fold_scores = []
        all_y_true, all_y_pred = [], []
        best_fold_score = 0.0
        best_fold_model = None

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_subj, y_subj)):
            print(f'  Subject {i+1}/{len(unique_subjects)} fold {fold+1}/{n_splits}...', flush=True)
            run_id = f'{_get_model_name(model)}_subject{subj}_fold{fold}'

            score, y_pred, y_true, fitted = _execute_run(
                model, X_subj[train_idx], y_subj[train_idx],
                X_subj[test_idx], y_subj[test_idx],
                run_id, save_dir,
                validation_ratio=0.2,
                save_model=False,   # não guarda todos os folds
            )
            fold_scores.append(score)
            if len(y_pred) > 0:
                all_y_true.extend(y_true.tolist())
                all_y_pred.extend(y_pred.tolist())

            if fitted is not None and score > best_fold_score:
                best_fold_score = score
                best_fold_model = fitted

        # save only best fold per subject
        if save_dir and best_fold_model is not None:
            best_fold_model.save(
                f'{save_dir}/{_get_model_name(model)}_subject{subj}_best.pt'
            )

        subject_scores[int(subj)] = float(np.mean(fold_scores))
        if all_y_true:
            subject_results[int(subj)] = SubjectResult(
                subject_id=int(subj),
                y_true=all_y_true,
                y_pred=all_y_pred,
                accuracy=subject_scores[int(subj)],
            )

    return EvaluationResult(
        evaluation=EvaluationType.INTRA_SUBJECT,
        split_type=SplitType.KFOLD,
        accuracy_mean=float(np.mean(list(subject_scores.values()))),
        accuracy_std=float(np.std(list(subject_scores.values()))),
        per_subject=subject_scores,
        per_subject_results=subject_results,
    )


def evaluate_cross_subject(
    model: ITrainableModel,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    save_dir: Optional[str] = None,
    validation_ratio: float = 0.1,
) -> EvaluationResult:
    """Cross-subject evaluation using Leave-One-Subject-Out (LOSO) protocol."""
    unique_subjects = np.unique(subject_ids)
    subject_scores = {}
    subject_results = {}

    for i, subj in enumerate(unique_subjects):
        print(f'  Cross-subject LOSO: subject {i+1}/{len(unique_subjects)}...', flush=True)
        run_id = f'{_get_model_name(model)}_loso_subject{subj}'

        test_mask = subject_ids == subj
        score, y_pred, y_true, _ = _execute_run(
            model, X[~test_mask], y[~test_mask],
            X[test_mask], y[test_mask],
            run_id, save_dir,
            validation_ratio=validation_ratio,
        )
        subject_scores[int(subj)] = score

        if len(y_pred) > 0:
            subject_results[int(subj)] = SubjectResult(
                subject_id=int(subj),
                y_true=y_true.tolist(),
                y_pred=y_pred.tolist(),
                accuracy=score,
            )

    return EvaluationResult(
        evaluation=EvaluationType.CROSS_SUBJECT,
        split_type=SplitType.NONE,
        accuracy_mean=float(np.mean(list(subject_scores.values()))),
        accuracy_std=float(np.std(list(subject_scores.values()))),
        per_subject=subject_scores,
        per_subject_results=subject_results,
    )


def evaluate_intra_subject_fixed_split(
    model: ITrainableModel,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    test_ratio: float = 0.2,
    validation_ratio: float = 0.2,
    save_dir: Optional[str] = None,
) -> EvaluationResult:
    """Intra-subject evaluation using a single fixed train/test split per subject."""
    unique_subjects = np.unique(subject_ids)
    subject_scores = {}
    subject_results = {}

    for i, subj in enumerate(unique_subjects):
        print(f'  Intra-subject fixed split: subject {i+1}/{len(unique_subjects)}...', flush=True)
        run_id = f'{_get_model_name(model)}_subject{subj}_fixed'

        mask = subject_ids == subj
        X_subj, y_subj = X[mask], y[mask]
        X_train, X_test, y_train, y_test = train_test_split(
            X_subj, y_subj,
            test_size=test_ratio,
            stratify=y_subj,
            random_state=42,
        )

        score, y_pred, y_true, _ = _execute_run(
            model, X_train, y_train, X_test, y_test,
            run_id, save_dir,
            validation_ratio=validation_ratio,
        )
        subject_scores[int(subj)] = score

        if len(y_pred) > 0:
            subject_results[int(subj)] = SubjectResult(
                subject_id=int(subj),
                y_true=y_true.tolist(),
                y_pred=y_pred.tolist(),
                accuracy=score,
            )

    return EvaluationResult(
        evaluation=EvaluationType.INTRA_SUBJECT,
        split_type=SplitType.FIXED,
        accuracy_mean=float(np.mean(list(subject_scores.values()))),
        accuracy_std=float(np.std(list(subject_scores.values()))),
        per_subject=subject_scores,
        per_subject_results=subject_results,
    )


def evaluate_session_split(
    model: ITrainableModel,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    session_ids: np.ndarray,
    train_session: str = '0train',
    test_session: str = '1test',
    validation_ratio: float = 0.2,
    save_dir: Optional[str] = None,
) -> EvaluationResult:
    """Exact BCI Competition IV 2a protocol: Train on Session 1, Test on Session 2."""
    unique_subjects = np.unique(subject_ids)
    subject_scores = {}
    subject_results = {}

    for i, subj in enumerate(unique_subjects):
        print(f'  Session split: subject {i+1}/{len(unique_subjects)}...', flush=True)
        # fix 4 — run_id inclui as sessões para evitar conflito de cache
        run_id = f'{_get_model_name(model)}_subject{subj}_{train_session}_vs_{test_session}'

        subj_mask = subject_ids == subj
        train_mask = subj_mask & (session_ids == train_session)
        test_mask = subj_mask & (session_ids == test_session)

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            print(f'    Skipping subject {subj} — missing session data')
            continue

        score, y_pred, y_true, _ = _execute_run(
            model, X[train_mask], y[train_mask],
            X[test_mask], y[test_mask],
            run_id, save_dir,
            validation_ratio=validation_ratio,
            apply_scaling=True,
        )
        subject_scores[int(subj)] = score

        if len(y_pred) > 0:
            subject_results[int(subj)] = SubjectResult(
                subject_id=int(subj),
                y_true=y_true.tolist(),
                y_pred=y_pred.tolist(),
                accuracy=score,
            )

    return EvaluationResult(
        evaluation=EvaluationType.INTRA_SUBJECT,
        split_type=SplitType.SESSION,
        accuracy_mean=float(np.mean(list(subject_scores.values()))),
        accuracy_std=float(np.std(list(subject_scores.values()))),
        per_subject=subject_scores,
        per_subject_results=subject_results,
    )