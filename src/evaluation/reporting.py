from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.evaluation.results import EvaluationResult
from src.evaluation.metrics import metrics_to_row
from src.models.core import ITrainableModel
from src.datasets.base_dataset import BaseDataset
from src.models.wrappers.nn_wrapper import NNWrapper 


def generate_report(
    result: EvaluationResult,
    dataset: BaseDataset,
    model: ITrainableModel,
    preprocessing: str = 'none',
    output_dir: str = 'reports',
    notes: str = '',
    seed: int = 42,
) -> None:
    """
    Generate a full report from an EvaluationResult:
      - reports/tables/{dataset}_{model}_results.csv
      - reports/{paradigm}_{dataset}_{model}_report.md
      - reports/figures/{dataset}_{model}_confusion_matrix.png
    """

    dataset_name      = dataset.name
    paradigm          = dataset.paradigm
    model_name        = model.name
    dataset_info      = dataset.dataset_info
    model_hyperparams = model.get_hyperparams()
    trainer_config    = model.train_config.to_dict() if isinstance(model, NNWrapper) else None

    output_dir: Path = Path(output_dir)
    tables_dir  = output_dir / 'tables'
    figures_dir = output_dir / 'figures'
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    slug = f'{dataset_name}_{model_name}'.lower().replace(' ', '_')
    date_str = datetime.now().strftime('%Y-%m-%d')

    # ── 1. compute metrics ────────────────────────────────────────────────────
    if not result.per_subject_results:
        raise RuntimeError(
            'EvaluationResult.per_subject_results is empty. '
            'Re-run evaluation without cache to populate predictions.'
        )

    global_metrics = result.compute_all_metrics(paradigm=paradigm)
    per_subj_metrics = result.per_subject_metrics(paradigm=paradigm)

    # ── 2. confusion matrix figure ────────────────────────────────────────────
    cm = np.array(global_metrics['confusion_matrix'])
    classes = global_metrics['classes']
    fig_path = figures_dir / f'{slug}_confusion_matrix.png'
    _plot_confusion_matrix(cm, classes, dataset_name, model_name, fig_path)

    # ── 3. CSV row ────────────────────────────────────────────────────────────
    csv_path = tables_dir / f'{slug}_results.csv'
    row = metrics_to_row(
        global_metrics,
        subject_id='all',
        model_name=model_name,
        dataset_name=dataset_name,
    )
    row.update({
        'date':          date_str,
        'paradigm':      paradigm,
        'preprocessing': preprocessing,
        'eval_type':     result.evaluation.value,
        'split_type':    result.split_type.value,
        'trainer_config': json.dumps(trainer_config) if trainer_config else 'N/A',
        'notes':         notes,
    })
    _append_csv(row, csv_path)

    # ── 4. summary_all_models.csv ─────────────────────────────────────────────
    summary_path = output_dir / 'tables' / 'summary_all_models.csv'
    summary_row = {
        'date':               date_str,
        'paradigm':           paradigm,
        'dataset':            dataset_name,
        'model':              model_name,
        'preprocessing':      preprocessing,
        'eval_type':          result.evaluation.value,
        'split_type':         result.split_type.value,
        'accuracy':           round(result.accuracy_mean, 4),
        'balanced_accuracy':  round(global_metrics['balanced_accuracy'], 4),
        'f1_macro':           round(global_metrics['f1_macro'], 4),
        'kappa':              round(global_metrics['kappa'], 4),
        'mcc':                round(global_metrics['mcc'], 4),
        'roc_auc':            round(global_metrics.get('roc_auc', float('nan')), 4),
        'pr_auc':             round(global_metrics.get('pr_auc', float('nan')), 4),
        'n_subjects':         dataset_info.get('n_subjects', '?') if dataset_info else '?',
        'n_classes':          dataset_info.get('n_classes', '?') if dataset_info else '?',
        'seed':               seed,
        'notes':              notes,
    }
    _append_csv(summary_row, summary_path)

    # ── 5. markdown report ────────────────────────────────────────────────────
    md_path = output_dir / f'{paradigm}_{slug}_report.md'
    _write_markdown(
        path=md_path,
        slug=slug,
        date_str=date_str,
        dataset_name=dataset_name,
        model_name=model_name,
        paradigm=paradigm,
        preprocessing=preprocessing,
        trainer_config=trainer_config,
        model_hyperparams=model_hyperparams,
        dataset_info=dataset_info or {},
        result=result,
        global_metrics=global_metrics,
        per_subj_metrics=per_subj_metrics,
        notes=notes,
        figures_dir=figures_dir,
    )

    print(f'  ✓ Report saved:')
    print(f'    CSV:     {csv_path}')
    print(f'    Summary: {summary_path}')
    print(f'    Figure:  {fig_path}')
    print(f'    Report:  {md_path}')


# ─── helpers ──────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    dataset_name: str,
    model_name: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(classes)), max(5, len(classes) - 1)))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=classes, yticklabels=classes, ax=ax,
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{dataset_name} × {model_name}')
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _append_csv(row: dict, path: Path) -> None:
    DEDUP_KEYS = ['dataset', 'model', 'preprocessing', 'eval_type', 'split_type', 'seed']
    df_new = pd.DataFrame([row])
    if path.exists():
        df_existing = pd.read_csv(path)
        for col in df_new.columns:
            if col not in df_existing.columns:
                df_existing[col] = float('nan')
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        # dedup — keep last run for identical experiment config
        dedup_cols = [c for c in DEDUP_KEYS if c in df_combined.columns]
        df_combined = df_combined.drop_duplicates(subset=dedup_cols, keep='last')
    else:
        df_combined = df_new
    df_combined.to_csv(path, index=False)


def _write_markdown(
    path: Path,
    slug: str,
    date_str: str,
    dataset_name: str,
    model_name: str,
    paradigm: str,
    preprocessing: str,
    trainer_config: Optional[Dict],
    model_hyperparams: Optional[Dict],
    dataset_info: dict,
    result: EvaluationResult,
    global_metrics: dict,
    per_subj_metrics: dict,
    notes: str,
    figures_dir: Path,
    **kwargs,
) -> None:

    chance = round(1.0 / global_metrics.get('n_classes', len(global_metrics['classes'])), 3) \
        if 'n_classes' not in global_metrics else round(1.0 / len(global_metrics['classes']), 3)

    # per-subject table rows
    per_subj_rows = []
    for subj, m in sorted(per_subj_metrics.items()):
        m: Dict
        per_subj_rows.append(
            f"| {subj} "
            f"| {m['accuracy']:.3f} "
            f"| {m.get('balanced_accuracy', float('nan')):.3f} "
            f"| {m['f1_macro']:.3f} "
            f"| {m['kappa']:.3f} "
            f"| {m['mcc']:.3f} |"
        )
    per_subj_table = '\n'.join(per_subj_rows)

    # f1 per class section
    f1_per_class: Dict = global_metrics.get('f1_per_class', {})
    f1_class_rows = '\n'.join(
        f'| {cls} | {score:.3f} |'
        for cls, score in f1_per_class.items()
    )

    # optional sections
    top_k_section = ''
    if 'top_k_accuracy' in global_metrics:
        top_k_section = f"""
### 4.2 Top-k Accuracy *(imagined speech)*

| k | Top-k Accuracy |
|---|----------------|
| {result.per_subject_results and 3} | {global_metrics['top_k_accuracy']:.3f} |
"""

    clinical_section = ''
    if 'roc_auc' in global_metrics:
        clinical_section = f"""
### 4.2 Clinical Metrics

| Metric      | Value  |
|-------------|--------|
| ROC-AUC     | {global_metrics.get('roc_auc', 'N/A'):.3f} |
| PR-AUC      | {global_metrics.get('pr_auc', 'N/A'):.3f} |
| Brier Score | {global_metrics.get('brier', 'N/A'):.3f} |
| ECE         | {global_metrics.get('ece', 'N/A'):.3f} |
"""

    trainer_str = (
        '\n'.join(f'# {k}: {v}' for k, v in trainer_config.items())
        if trainer_config else '# N/A — classical model'
    )
    model_hp_str = (
        '\n'.join(f'# {k}: {v}' for k, v in model_hyperparams.items())
        if model_hyperparams else '# not provided'
    )
    cm_raw = '\n'.join(str(row) for row in global_metrics['confusion_matrix'])

    md = f"""# EEG Classification Report — {dataset_name} × {model_name}

**Date:** {date_str}
**Paradigm:** {paradigm}
**Seed:** 42

---

## 1. Dataset

| Property        | Value |
|-----------------|-------|
| Name            | {dataset_name} |
| Subjects        | {dataset_info.get('n_subjects', '?')} |
| Classes         | {', '.join(global_metrics['classes'])} |
| Channels        | {dataset_info.get('n_channels', '?')} |
| Sampling Rate   | {dataset_info.get('sfreq', '?')} Hz |
| Trials/Subject  | {dataset_info.get('trials_per_subject', '?')} |

---

## 2. Pipeline

| Stage         | Details |
|---------------|---------|
| Preprocessing | {preprocessing} |
| Model         | {model_name} |
| Trainer       | {'provided' if trainer_config else 'N/A — classical model'} |

### 2.1 Preprocessing Details
```
{preprocessing}
```

### 2.2 Model Hyperparameters
```
{model_hp_str}
```

### 2.3 Trainer Config
```
{trainer_str}
```

---

## 3. Evaluation Protocol

| Property      | Value |
|---------------|-------|
| Type          | {result.evaluation.value} |
| Split         | {result.split_type.value} |

---

## 4. Results Summary

### 4.1 Core Metrics

| Metric            | Value  |
|-------------------|--------|
| Accuracy          | {result.accuracy_mean:.3f} ± {result.accuracy_std:.3f} |
| Balanced Accuracy | {global_metrics.get('balanced_accuracy', float('nan')):.3f} |
| F1-Macro          | {global_metrics['f1_macro']:.3f} |
| Cohen's Kappa     | {global_metrics['kappa']:.3f} |
| MCC               | {global_metrics['mcc']:.3f} |
| Chance Level      | {chance:.3f} |

{top_k_section}
{clinical_section}

---

## 5. Per-Subject Results

| Subject | Accuracy | Balanced Acc | F1-Macro | Kappa | MCC |
|---------|----------|--------------|----------|-------|-----|
{per_subj_table}

---

## 6. Confusion Matrix

![Confusion Matrix](figures/{slug}_confusion_matrix.png)
```
{cm_raw}
```

---

## 7. F1 Per Class

| Class | F1 |
|-------|----|
{f1_class_rows}

---

## 8. Notes & Observations

{notes if notes else '_No notes provided._'}

---

## 9. References

- Dataset: {dataset_name}
- Model: {model_name}
"""
    path.write_text(md)