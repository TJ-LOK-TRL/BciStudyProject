import argparse
import optuna
import numpy as np
#from src.hpo.optuna_runner import run_study


from src.experiments.motor_imagery.bci2a_eeg_encoder_eog import BCI2aEEGEncoderEOGExperiment as Experiment
from src.experiments.motor_imagery.bnci2014_004_eeg_encoder_eog import BNCI2014004EEGEncoderEOGExperiment as Experiment2
#NOMES NAO FINAIS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['imagined_words','motor_imagery','clinical_states'])
    p.add_argument('--model', required=False, choices=['lstm','gru','cnn1d','liquid'])
    p.add_argument('--n-trials', type=int, default=10)
    p.add_argument('--study-name', required= True ,type=str, default=None)
    p.add_argument('--n-jobs', type=int, default=1)
    p.add_argument('--direction', default='maximize')
    args = p.parse_args()

    
    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True, group=True)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)


    study = optuna.create_study(                
        study_name=args.study_name,
        direction=args.direction,
        sampler=sampler,
        pruner=pruner
    )

    study.optimize(objective, n_trials = args.n_trials, n_jobs = args.n_jobs, show_progress_bar=True)

    #study = run_study(args.dataset, args.model, n_trials=args.n_trials, study_name=args.study_name)
    print("[OK] Best value:", study.best_value)
    print("[OK] Best params:", study.best_params)

    with(open(f"optuna_studies/{args.study_name}_trials.txt", "a")) as f:
        for t in study.trials:
            f.write(f"Trial {t.number}: value={t.value}, params={t.params}\n")

    with(open(f"optuna_studies/{args.study_name}_best_parameters.txt", "a")) as f:
        f.write(f"Best Results for the study \"{args.study_name}\".\n\n")
        f.write(f"Best Value: {study.best_value}\nBest Parameters: {study.best_params}")


def suggest_params(trial: optuna.Trial) -> dict:
    params = {
        "eegn_kern_size": trial.suggest_categorical("eegn_kern_size", [33,49,65]),
        "n_epochs": trial.suggest_int("n_epochs", 40,120),
        "loss_scale": trial.suggest_float("loss_scale", 0.0, 0.2),
        "l2_scale": trial.suggest_float("l2_scale", 0.0, 0.2),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
        "patience": trial.suggest_int("patience", 10, 60),
    }
    #FALTA LR!

    return params

def objective(trial: optuna.Trial) -> float:

    p = suggest_params(trial)

    exp = Experiment2() 
    X,y, subject_ids = exp.prepare_data()

    print(f"EPOCHS SUGGESTED > {p['n_epochs']}")

    model = exp.build_model(eegn_kern_size=p['eegn_kern_size'],
                                n_epochs=p['n_epochs'],
                                loss_scale=p['loss_scale'],
                                l2_scale=p['l2_scale'],
                                label_smoothing=p['label_smoothing'],
                                patience=p['patience'])
    #FALTA LR
    
    result: EvaluationResult = exp.run(model=model,
                     X=X,
                     y=y,
                     subject_ids=subject_ids)
    
    if result is None or result.accuracy_mean is None or not np.isfinite(result.accuracy_mean):
        return None
    
    return result.accuracy_mean


if __name__ == '__main__':
    main()
