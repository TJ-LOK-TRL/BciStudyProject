import argparse
from src.hpo.optuna_runner import run_study


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['imagined_words','motor_imagery','clinical_states'])
    p.add_argument('--model', required=True, choices=['lstm','gru','cnn1d','liquid'])
    p.add_argument('--n-trials', type=int, default=10)
    p.add_argument('--study-name', type=str, default=None)
    args = p.parse_args()

    study = run_study(args.dataset, args.model, n_trials=args.n_trials, study_name=args.study_name)
    print("[OK] Best value:", study.best_value)
    print("[OK] Best params:", study.best_params)

if __name__ == '__main__':
    main()
