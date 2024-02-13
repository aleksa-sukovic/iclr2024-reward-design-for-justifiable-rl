import torch
import click
import pickle
import random
import numpy as np

from typing import Optional
from os import makedirs
from os.path import join
from typing import Optional
from torch.optim.lr_scheduler import ReduceLROnPlateau

from argo.models.clinician import ClinicianPolicy
from argo.datasets.sepsis import get_clinician_dataset
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
@click.option("--base-model", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="Path to base model checkpoint to load.")
@click.option("--use-dem", default=True, type=click.BOOL, help="Use demographic features as part of the state-space.")
@click.option("--perform-scaling", default=True, type=click.BOOL, help="Perform standardization of dataset before training.")
@click.option("--batch-size", default=128, type=int, help="Batch size to use during training and evaluation.")
@click.option("--hidden-dim", default=256, type=int, help="Dimension of the network hidden layer.")
@click.option("--optimizer", type=click.Choice(["adam", "sgd", "rmsprop"]), default="adam", help="Determines the optimizer to use.")
@click.option("--lr", type=float, default=1e-3, help="Learning rate to use.")
@click.option("--lr-patience", type=int, default=None, help="Number of epochs with no improvement after which learning rate will be reduced by a factor of 0.1.")
@click.option("--weight-decay", type=float, default=1e-1, help="Amount of weight decay regularization to use.")
@click.option("--epochs", type=int, default=100, help="Number of epochs to run the training.")
@click.option("--es-patience", type=int, default=None, help="Early stopping patience, or None to disable early stopping.")
@click.option("--train-device", type=str, default="cpu", help="Target device on which to run the training.")
@click.option("--debug", type=bool, default=False, help="Uses only a single batch for training. Useful for debugging the model.")
@click.option("--seed", type=int, default=25, help="Random seed to use for reproducibility.")
def train(artifacts_dir: str, train_dict_path: str, val_dict_path: str, test_dict_path: str, base_model: str, use_dem: bool, perform_scaling: bool, batch_size: int, hidden_dim: int, optimizer: str, lr: float, lr_patience: int, weight_decay: float, epochs: int, es_patience: Optional[int], train_device: str, debug: bool, seed: int):
    # sets up logging and output directories
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    makedirs(artifacts_dir, exist_ok=True)
    logger = configure_logger(log_dir=artifacts_dir)
    device = torch.device(train_device)

    # sets up training and evaluation datasets
    train_dict, val_dict, test_dict = torch.load(train_dict_path).cpu(), torch.load(val_dict_path).cpu(), torch.load(test_dict_path).cpu()
    state_dim = train_dict["obs"].size(-1) if not use_dem else train_dict["obs"].size(-1) + train_dict["dem"].size(-1)
    action_dim = train_dict["actions"].size(-1)

    _, train_loader, scaler = get_clinician_dataset(data_dict=train_dict, use_dem=use_dem, batch_size=batch_size, perform_scaling=perform_scaling, debug=debug)
    _, val_loader, _ = get_clinician_dataset(data_dict=test_dict, use_dem=use_dem, batch_size=batch_size, scaler=scaler, perform_scaling=perform_scaling, debug=debug)
    _, test_loader, _ = get_clinician_dataset(data_dict=val_dict, use_dem=use_dem, batch_size=batch_size, scaler=scaler, perform_scaling=perform_scaling, debug=debug)

    # sets up model
    clinician = ClinicianPolicy(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, optimizer=optimizer, lr=lr, weight_decay=weight_decay, device=device)
    if base_model:
        clinician_data = torch.load(base_model, map_location=device)
        clinician.model.load_state_dict(clinician_data["weights"])

    # sets up training schedulers
    scheduler = ReduceLROnPlateau(clinician.optimizer, "min", patience=lr_patience, verbose=True) if lr_patience else None
    early_stopping = EarlyStopping(patience=es_patience, improvement="higher") if es_patience else None
    val_frequency = 1

    # runs training and evaluation
    for epoch in range(1, epochs + 1):
        result = clinician.epoch_train(train_loader)
        logger.bind(mode="TRAIN", epoch=epoch).info(f" ".join([f"{k}={v.item():.4f}" for k, v in result.items()]))

        if epoch % val_frequency == 0:
            # runs evaluation and saves model
            result = clinician.epoch_eval(val_loader)
            result = {f"val_{k}": v for k, v in result.items()}
            torch.save({"model": clinician, "weights": clinician.model.state_dict(), "optim": clinician.optimizer.state_dict()}, join(artifacts_dir, "clinician.pt"))

            # implements early stopping and LR decay
            logger.bind(mode="VALIDATION", epoch=epoch).info(f" ".join([f"{k}={v.item():.4f}" for k, v in result.items()]))

            if lr_patience:
                scheduler.step(result["val_loss"])
            if es_patience:
                early_stopping.step(result["val_accuracy"], logger=logger.bind(mode="VALIDATION", epoch=epoch))
                if early_stopping.stop: break

    # performs final evaluation on the test set
    result = clinician.epoch_eval(test_loader)
    result = {f"test_{k}": v for k, v in result.items()}
    logger.bind(mode="TEST", epoch=epoch).info(f" ".join([f"{k}={v.item():.4f}" for k, v in result.items()]))

    # Persists trained model and data scaler
    logger.bind(mode="TRAIN", epoch=epoch).info(f"Training finished, saving model to {join(artifacts_dir, 'clinician.pt')}")
    torch.save({"model": clinician, "weights": clinician.model.state_dict(), "optim": clinician.optimizer.state_dict()}, join(artifacts_dir, "clinician.pt"))
    if perform_scaling:
        pickle.dump(scaler, open(join(artifacts_dir, "scaler.pkl"), "wb"))


if __name__ == "__main__":
    main()
