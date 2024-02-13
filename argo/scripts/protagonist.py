import torch
import click
import pickle
import random
import numpy as np

from os.path import join
from tianshou.trainer import OfflineTrainer

from argo.library.evaluation import wis
from argo.datasets.sepsis import get_patient_dataset, get_patient_buffer
from argo.models.protagonist import make_ddqn_protagonist
from argo.library.logging import configure_logger


@click.group()
def main():
    pass


@main.command()
@click.option("--artifacts-dir", required=True, type=click.Path(), help="Path to directory where generated artifacts will be saved.")
@click.option("--train-dict-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="Path to exported train tensor dict (see `generate_dataset` script).")
@click.option("--test-dict-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="Path to exported test tensor dict (see `generate_dataset` script).")
@click.option("--buffer-path", required=False, type=click.Path(file_okay=True, dir_okay=False), help="Path to training buffer to use. If not provided, buffer will be generated from the training dataset.")
@click.option("--clinician-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="Path to exported model for behavior-cloning clinician policy.")
@click.option("--argumentator-path", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="Path to exported argumentator model.")
@click.option("--baseline-path", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="Path to exported baseline policy model used to debate against.")
@click.option("--judge-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="Path to exported judge model.")
@click.option("--use-dem", default=True, type=click.BOOL, help="Use demographic features as part of the state-space.")
@click.option("--hidden-dim", default=128, type=int, help="Dimension of the policy network hidden layer.")
@click.option("--hidden-depth", default=2, type=int, help="Number of hidden layers in the policy network.")
@click.option("--lr", default=1e-5, type=float, help="Policy learning rate.")
@click.option("--epochs", default=800, type=int, help="Number of training iterations to run using just the offline dataset.")
@click.option("--update-per-epoch", default=1, type=int, help="Number of gradient steps (of size 'batch_size') to perform per epoch.")
@click.option("--batch-size", default=128, type=int, help="The batch size of sample data, which is going to feed in the policy network.")
@click.option("--tau", default=1e-3, type=float, help="Polyak update coefficient (tau) for updating the target network.")
@click.option("--n-estimation-step", default=3, type=int, help="Number of n-step estimation performed when calculating returns.")
@click.option("--relu-slope", default=0.01, type=float, help="Slope of the LeakyReLU activation function.")
@click.option("--reward-multiplier", default=15.0, type=float, help="Multiplier for the reward scalar observed in the dataset, to facilitate more stable learning.")
@click.option("--debate-multiplier", default=10.0, type=float, help="Multiplier for the justifiability reward, to facilitate more stable learning.")
@click.option("--debate-deterministic", default=True, type=click.BOOL, help="Indicates if debate arguments are sampled deterministically (argmax) or stochastically (from distribution).")
@click.option("--num-arguments", default=6, type=int, help="Number of arguments to be proposed during debate.")
@click.option("--lmbd-justifiability", default=0.0, type=float, help="Lambda parameter from the paper, which determines the justifiability trade-off.")
@click.option("--dense-reward", default=True, type=click.BOOL, help="Use patient's SOFA as an intermediate reward.")
@click.option("--gamma", default=0.99, type=float, help="Discount factor.")
@click.option("--train-device", type=str, default="cpu", help="Target device on which to run the training.")
@click.option("--seed", type=str, default="25", help="Random seed to use for reproducibility.")
def train_ddqn(artifacts_dir: str, train_dict_path: str, test_dict_path: str, buffer_path: str, clinician_path: str, argumentator_path: str, baseline_path: str, judge_path: str, use_dem: bool, hidden_dim: int, hidden_depth: int, lr: float, epochs: int, update_per_epoch: int, batch_size: int, tau: float, n_estimation_step: int, relu_slope: float, reward_multiplier: float, debate_multiplier: float, debate_deterministic: bool, num_arguments: float, lmbd_justifiability: float, dense_reward: bool, gamma: float, train_device: str, seed: str):
    train_dict, test_dict = torch.load(train_dict_path).cpu(), torch.load(test_dict_path).cpu()
    logger = configure_logger(log_dir=artifacts_dir, log_epochs=False, log_iters=True, log_training_stage=False)

    state_dim = test_dict["obs"].size(-1) if not use_dem else test_dict["obs"].size(-1) + test_dict["dem"].size(-1)
    action_dim = test_dict["actions"].size(-1)

    # === initializes static models ===
    clinician = torch.load(clinician_path, map_location=train_device)["model"]
    clinician = clinician.eval()

    judge = torch.load(judge_path, map_location=train_device)
    judge = judge.eval()

    argumentator = torch.load(argumentator_path, map_location=train_device)["model"].eval()
    argumentator.ret_rms = torch.load(argumentator_path, map_location=train_device)["ret_rms"]
    argumentator._deterministic_eval = debate_deterministic

    baseline = torch.load(baseline_path, map_location=train_device)["model"] if baseline_path is not None else None
    baseline = baseline.eval() if baseline is not None else None

    for s in seed.split(","):
        # === sets seeds ===
        np.random.seed(int(s))
        torch.manual_seed(int(s))
        random.seed(int(s))

        # === initializes data ===
        _, test_patients, _ = get_patient_dataset(
            data_dict=test_dict, use_dem=use_dem, batch_size=batch_size, device=train_device,
            shuffle=False, dense_reward=False, mask=False, num_workers=0)
        _, test_debates, _ = get_patient_dataset(
            data_dict=test_dict, use_dem=use_dem, batch_size=batch_size, lmbd_justifiability=1.0, argumentator=argumentator,
            judge=judge, baseline=baseline, debate_multiplier=1.0, num_arguments=num_arguments, device=train_device, shuffle=False,
            mask=False, dense_reward=False, num_workers=0)
        buffer = get_patient_buffer(
            data_dict=train_dict, use_dem=use_dem, device=train_device, save_path=buffer_path, lmbd_justifiability=lmbd_justifiability,
            argumentator=argumentator, judge=judge, baseline=baseline, debate_multiplier=debate_multiplier, num_arguments=num_arguments,
            dense_reward=dense_reward, reward_multiplier=reward_multiplier)

        # === initializes policy ===
        policy = make_ddqn_protagonist(
            state_dim=state_dim, action_dim=action_dim, lr=lr, tau=tau, relu_slope=relu_slope,
            hidden_dim=hidden_dim, hidden_depth=hidden_depth, n_estimation_step=n_estimation_step,
            discount=gamma, device=train_device)

        # === initializes trainer ===
        trainer = OfflineTrainer(
            policy=policy, buffer=buffer,
            max_epoch=epochs,                  # train for 'epochs' epochs
            update_per_epoch=update_per_epoch, # where each epoch repeats 'update_per_epoch' times
            batch_size=batch_size,             # fetching of 'batch_size' samples of shape (s, a, r, s', d) and performing gradient update
            test_collector=None, episode_per_test=0, verbose=False, show_progress=False)

        # === performs initial evaluation ===
        evaluations = {"wis": [], "jstf": []}

        # === runs the training ===
        for epoch, epoch_stat, _ in trainer:
            # performs policy evaluation
            with torch.no_grad():
                policy = policy.eval()
                wis_eval = wis(target=policy, behavioral=clinician, patients=test_patients, discount=gamma, device=train_device)
                jstf_eval = wis(target=policy, behavioral=clinician, patients=test_debates, discount=0.0, device=train_device)
                policy = policy.train()

            # logs the evaluation results
            logger.bind(iter=epoch).info(f"test/{s}/wis={wis_eval:.2f} test/{s}/jstf={jstf_eval:.2f} train/{s}/loss={epoch_stat['loss']:.2f}")
            evaluations["wis"].append(wis_eval)
            evaluations["jstf"].append(jstf_eval)

            # saves current version of the policy
            pickle.dump(evaluations, open(join(artifacts_dir, f"evaluations-s{s}.pkl"), "wb"))
            torch.save({"model": policy, "weights": policy.state_dict(), "optimizer": policy.optim.state_dict()}, join(artifacts_dir, f"protagonist-s{s}.pt"))

    print("Finished training.")

if __name__ == "__main__":
    main()
