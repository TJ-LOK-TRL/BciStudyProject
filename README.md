# EEG-AI — Arquitetura Modular para Classificação de EEG

Projeto scaffold criado para suportar **vários datasets** (Imaginação de Palavras, Motor Imagery, Estados Clínicos), **múltiplos modelos** (LSTM, GRU, CNN1D, Liquid NN) e **otimização de hiperparâmetros com Optuna**.

## Visão Geral
- **DataAdapters por dataset** → convertem dados brutos num **formato comum** `[B, C, T]` + metadados.
- **Input Model Adapters** → convertem o formato comum para o formato de entrada do modelo (p.ex. `[B, T, C]` para LSTM/GRU/Liquid).
- **Modelos modulares** → `BaseModel` + implementações `LSTMModel`, `GRUModel`, `CNN1DModel`, `LiquidNNBasic`.
- **Treino & Avaliação** → `Trainer` com métricas robustas e relatórios standard.
- **HPO (Optuna)** → estudos por Dataset×Modelo, com pruning e logging em SQLite.

> **Nota**: Este scaffold não inclui dados. Coloque os ficheiros sob `data/raw/<dataset>/` ou ajuste os DataAdapters.

## Requisitos
- Python 3.10+
- Ver `requirements.txt` para dependências sugeridas (PyTorch, numpy, mne opcional, optuna).

## Como usar
1. Crie e ative um ambiente virtual.
2. Instale dependências: `pip install -r requirements.txt`.
3. Ajuste configs em `configs/`.
4. Pré-proc (opcional): `python scripts/preprocess.py --dataset motor_imagery`.
5. Treinar: `python scripts/train.py --config configs/base.yaml`.
6. Avaliar: `python scripts/evaluate.py --config configs/base.yaml`.
7. Otimizar (Optuna): `python scripts/run_optuna.py --dataset motor_imagery --model lstm --n-trials 10`.
