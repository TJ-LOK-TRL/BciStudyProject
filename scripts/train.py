import argparse
from src.utils.io import load_yaml
from src.utils.seed import set_seed
from src.models.lstm import LSTMModel
from src.models.gru import GRUModel
from src.models.cnn1d import CNN1DModel
from src.models.liquid import LiquidNNBasic
from src.input_adapters.temporal import to_sequence_input
from src.training.trainer import Trainer, TrainerConfig

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get('seed', 42))

    import torch
    B, C, T, n_classes = 64, 32, cfg.get('window_size', 500), 4
    model_name = cfg.get('training', {}).get('model', 'lstm')

    if model_name in ['lstm','gru','liquid']:
        Xtr = torch.randn(B, T, C); Xva = torch.randn(B//2, T, C)
    else:
        Xtr = torch.randn(B, C, T); Xva = torch.randn(B//2, C, T)
    ytr = torch.randint(0, n_classes, (B,))
    yva = torch.randint(0, n_classes, (B//2,))

    if model_name == 'lstm':
        model = LSTMModel(C, n_classes)
    elif model_name == 'gru':
        model = GRUModel(C, n_classes)
    elif model_name == 'cnn1d':
        model = CNN1DModel(C, n_classes)
    else:
        model = LiquidNNBasic(C, n_classes)

    train_loader = [(Xtr, ytr)]
    val_loader = [(Xva, yva)]

    tcfg = TrainerConfig(**cfg.get('training', {}))
    trainer = Trainer(model, train_loader, val_loader, tcfg)
    score, hist = trainer.train_and_evaluate(single_epoch=True)
    print({"best_score": score})

if __name__ == '__main__':
    main()
