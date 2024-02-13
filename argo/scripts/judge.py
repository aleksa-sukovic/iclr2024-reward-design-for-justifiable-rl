import torch
import click
import random
import numpy as np

from pickle import dump
from typing import Optional
from os import makedirs
from os.path import join
from typing import Optional
from torch.optim.lr_scheduler import ReduceLROnPlateau

from argo.models.judge import Judge
from argo.datasets.sepsis import get_judge_dataset
from argo.library.logging import configure_logger
from argo.library.training import EarlyStopping


@click.group()
def main():
    pass


@main.command()
@click.option("--artifacts-dir", required=True, type=click.Path(), help="Path to directory where generated artifacts will be saved.")
@click.option("--train-dict-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="Path to exported train tensor dict (see `generate_dataset` script).")
@click.option("--val-dict-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="Path to exported val tensor dict (see `generate_dataset` script).")
@click.option("--test-dict-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="Path to exported test tensor dict (see `generate_dataset` script).")
@click.option("--train-preferences-path", required=False, default=None, type=click.Path(file_okay=True, dir_okay=False), help="Path to exported training preferences. Used to ensure judges trained with different arguments use the same data.")
@click.option("--val-preferences-path", required=False, default=None, type=click.Path(file_okay=True, dir_okay=False), help="Path to exported validation preferences. Used to ensure judges trained with different arguments use the same data.")
@click.option("--test-preferences-path", required=False, default=None, type=click.Path(file_okay=True, dir_okay=False), help="Path to exported test preferences. Used to ensure judges trained with different arguments use the same data.")
@click.option("--base-model", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="Path to base model weights to load.")
@click.option("--use-dem", default=True, type=click.BOOL, help="Use demographic features as part of the state-space.")
@click.option("--perform-scaling", default=True, type=click.BOOL, help="Perform standardization of dataset before training.")
@click.option("--num-arguments", default=-1, type=int, help="Number of state-features (arguments) to be sampled during training, setting rest to zero. Use -1 to utilize all state features as arguments.")
@click.option("--batch-size", default=256, type=int, help="Batch size to use during training and evaluation.")
@click.option("--hidden-dim", default=64, type=int, help="Dimension of the network embedding hidden layer.")
@click.option("--optimizer", type=click.Choice(["adam", "sgd", "rmsprop"]), default="adam", help="Determines the optimizer to use.")
@click.option("--preference-generation-method", type=click.Choice(["random", "offset", "exhaustive"]), default="random", help="Determines how are actions from the dataset paired to generate a synthetic preference dataset.")
@click.option("--lr", type=float, default=1e-3, help="Learning rate to use.")
@click.option("--lr-patience", type=int, default=None, help="Number of epochs with no improvement after which learning rate will be reduced by a factor of 0.1.")
@click.option("--weight-decay", type=float, default=0.0, help="Amount of weight decay regularization to use.")
@click.option("--epochs", type=int, default=100, help="Number of epochs to run the training.")
@click.option("--es-patience", type=int, default=None, help="Early stopping patience, or None to disable early stopping.")
@click.option("--train-device", type=str, default="cpu", help="Target device on which to run the training.")
@click.option("--debug", type=click.BOOL, default=False, help="Uses only a single batch for training. Useful for debugging the model.")
@click.option("--seed", type=int, default=25, help="Random seed to use for reproducibility.")
def train(artifacts_dir: str, train_dict_path: str, val_dict_path: str, test_dict_path: str, train_preferences_path: str, val_preferences_path: str, test_preferences_path: str, base_model: str, use_dem: bool, perform_scaling: bool, num_arguments: Optional[int], batch_size: int, hidden_dim: int, optimizer: str, preference_generation_method: str, lr: float, lr_patience: int, weight_decay: float, epochs: int, es_patience: Optional[int], train_device: str, debug: bool, seed: int):
    logger = configure_logger(log_dir=artifacts_dir)
    makedirs(artifacts_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Sets up datasets
    train_dict, val_dict, test_dict = torch.load(train_dict_path).cpu(), torch.load(val_dict_path).cpu(), torch.load(test_dict_path).cpu()
    state_dim = train_dict["obs"].size(2) if not use_dem else train_dict["obs"].size(2) + train_dict["dem"].size(2)
    num_actions = train_dict["actions"].size(2)

    train_preferences_path = train_preferences_path or join(artifacts_dir, "train_preferences.pt")
    val_preferences_path = val_preferences_path or join(artifacts_dir, "val_preferences.pt")
    test_preferences_path = test_preferences_path or join(artifacts_dir, "test_preferences.pt")

    _, train_loader, scaler = get_judge_dataset(data_dict=train_dict, use_dem=use_dem, batch_size=batch_size, debug=debug, perform_scaling=perform_scaling, method=preference_generation_method, weighted_sampling=True, load_path=train_preferences_path)
    _, val_loader, _ = get_judge_dataset(data_dict=test_dict, use_dem=use_dem, batch_size=batch_size, scaler=scaler, debug=debug, perform_scaling=perform_scaling, method=preference_generation_method, weighted_sampling=False, load_path=val_preferences_path)
    _, test_loader, _ = get_judge_dataset(data_dict=val_dict, use_dem=use_dem, batch_size=batch_size, scaler=scaler, debug=debug, perform_scaling=perform_scaling, method=preference_generation_method, weighted_sampling=False, load_path=test_preferences_path)

    # Loads model and prepares for training
    device = torch.device(train_device)
    model = Judge(state_dim=state_dim, num_actions=num_actions, hidden_dim=hidden_dim, device=device,
                  lr=lr, optimizer=optimizer, weight_decay=weight_decay, num_arguments=None if num_arguments == -1 else num_arguments)
    model = model.to(device)

    if base_model:
        model.load_state_dict(torch.load(base_model, map_location=device).state_dict())

    scheduler = ReduceLROnPlateau(model.optimizer, "min", patience=lr_patience, verbose=True) if lr_patience else None
    early_stopping = EarlyStopping(patience=es_patience, improvement="higher") if es_patience else None
    val_frequency = 20
    metrics = {"train": [], "val": [], "test": {}}

    for epoch in range(1, epochs + 1):
        result = model.epoch_train(train_loader)
        logger.bind(mode="TRAIN", epoch=epoch).info(" ".join([f"{k}={v.item():.4f}" for k, v in result.items()]))
        metrics["train"].append(result)

        if (epoch) % val_frequency == 0:
            # Runs evaluation over the entire validation set
            result = model.epoch_eval(val_loader)
            result = {f"val_{k}": v for k, v in result.items()}
            metrics["val"].append(result)

            # Implements early stopping and LR decay
            logger.bind(mode="VALIDATION", epoch=epoch).info(f" ".join([f"{k}={v.item():.4f}" for k, v in result.items()]))

            if lr_patience:
                scheduler.step(result["val_accuracy"])
            if es_patience:
                early_stopping.step(result["val_accuracy"], logger=logger.bind(mode="VALIDATION", epoch=epoch))
                if early_stopping.stop: break

    # Performs final evaluation on the test set
    result = model.epoch_eval(test_loader)
    result = {f"test_{k}": v for k, v in result.items()}
    metrics["test"] = result
    logger.bind(mode="TEST", epoch=epoch).info(f" ".join([f"{k}={v.item():.4f}" for k, v in result.items()]))

    # Persists trained model and data scaler
    logger.bind(mode="TRAIN", epoch=epoch).info(f"Training finished, saving model to {join(artifacts_dir, 'model.pt')}")
    torch.save(model, join(artifacts_dir, "judge.pt"))
    torch.save(model.state_dict(), join(artifacts_dir, "judge.weights.pt"))
    torch.save(metrics, join(artifacts_dir, "metrics.pt"))
    if perform_scaling:
        dump(scaler, open(join(artifacts_dir, "scaler.pkl"), "wb"))


if __name__ == "__main__":
    main()
