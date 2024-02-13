import pickle
import torch
import click
import random
import numpy as np

from os.path import join, exists
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger

from argo.datasets.sepsis import get_debate_dataset, get_xai_debate_dataset
from argo.envs.sepsis import SepsisArgumentationEnv
from argo.models.argumentator import make_argumentator


@click.group()
def main():
    pass


@main.command()
@click.option("--artifacts-dir", required=True, type=click.Path(), help="Path to directory where generated artifacts will be saved.")
@click.option("--judge-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="Path to exported PyTorch judge model.")
@click.option("--train-dataset-path", required=True, type=click.Path(file_okay=True, dir_okay=False), help="Path to training dataset exported during judge training (see `scripts.judge.train`).")
@click.option("--test-dataset-path", required=True, type=click.Path(file_okay=True, dir_okay=False), help="Path to test dataset exported during judge training (see `scripts.judge.train`).")
@click.option("--opponent-path", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="Path to opponent PyTorch model.")
@click.option("--hidden-dim", default=256, type=int, help="Dimension of the policy network hidden layer.")
@click.option("--hidden-depth", default=2, type=int, help="Number of hidden layers.")
@click.option("--num-arguments", default=6, type=int, help="Number of state-features (arguments) to be used.")
@click.option("--num-train-envs", default=1, help="Number of parallel environments to use.")
@click.option("--num-test-envs", default=1, help="Number of parallel environments to use.")
@click.option("--epochs", default=100, help="Number of training epochs to run. Total of epochs * step_per_epoch steps will be executed.")
@click.option("--step-per-epoch", default=4096, help="Number of transitions to collect per each epoch.")
@click.option("--step-per-collect", default=None, type=int, help="Number of transitions to collect before performing the network update.")
@click.option("--episode-per-collect", default=384, type=int, help="Number of complete episodes to collect before performing the network update.")
@click.option("--repeat-per-collect", default=10, help="Number of policy learning updates to perform per one collect batch.")
@click.option("--lr", default=5e-4, type=float, help="Policy learning rate.")
@click.option("--lr-schedule", default="constant", type=click.Choice(["constant", "linear", "step"]), help="Learning rate scheduler.")
@click.option("--ent-coef", default=1e-4, type=float, help="Strength of entropy regularization.")
@click.option("--clip-range", default=0.2, type=float, help="PPO clip range (epsilon).")
@click.option("--gamma", default=0.8, type=float, help="Discount factor.")
@click.option("--gae-lambda", default=0.95, type=float, help="Factor for trade-off of bias vs variance for Generalized Advantage Estimator.")
@click.option("--vf-coef", default=0.5, type=float, help="Value function coefficient for the loss calculation.")
@click.option("--max-grad-norm", default=0.5, type=float, help="The maximum value for the gradient clipping.")
@click.option("--normalize-rewards", default=False, type=click.BOOL, help="Determines if rewards passed to the agents are normalized to have std (close to) 1.")
@click.option("--use-judge-diff-as-reward", default=False, type=click.BOOL, help="Determines if difference between judge's evaluations is used as a reward.")
@click.option("--ortho-init", default=True, type=click.BOOL, help="Determines if orthogonal layer initialization is used.")
@click.option("--xai-method", default=None, type=click.Choice(["shap", "lime"]), help="Indicates that the confuser agent should be trained against arguments proposed by the specified XAI method.")
@click.option("--xai-bg-dataset", default=None, type=click.Path(exists=False, dir_okay=False), help="Path to exported dataset of samples used as background data for XAI method. If not specified, it will be automatically generated and saved in the output directory.")
@click.option("--xai-policy", default=None, type=click.Path(exists=True, dir_okay=False), help="Path to exported clinician policy which will be explained by the XAI method.")
@click.option("--xai-num-arguments", default=3, type=int, help="Number of arguments/evidence to be proposed by XAI method, typically L//2.")
@click.option("--propose-evidence-upfront", default=False, type=click.BOOL, help="Determines if the opponent proposes num_arguments/3 evidence upfront, or if it proposes one evidence per turn.")
@click.option("--batch-size", default=64, type=int, help="Number of transitions to use when performing network update.")
@click.option("--limit", default=None, type=int, help="Maximum number of patient transitions to use for training.")
@click.option("--train-device", type=str, default="cpu", help="Target device on which to run the training.")
@click.option("--resume-path", type=click.Path(exists=True, file_okay=True), required=False, default=None, help="Indicates if training should be resumed from specified checkpoint.")
@click.option("--seed", type=int, default=25, help="Random seed to use for reproducibility.")
def train(artifacts_dir: str, judge_path: str, train_dataset_path: str, test_dataset_path: str, opponent_path: str, hidden_dim: int, hidden_depth: int, num_arguments: int, lr: float, lr_schedule: str, ent_coef: float, clip_range: float, gae_lambda: float, vf_coef: float, max_grad_norm: float, normalize_rewards: bool, use_judge_diff_as_reward: bool, gamma: float, ortho_init: str, xai_method: str, xai_bg_dataset: str, xai_policy: str, xai_num_arguments: int, propose_evidence_upfront: bool, num_train_envs: int, num_test_envs: int, epochs: int, step_per_epoch: int, step_per_collect: int, episode_per_collect: int, repeat_per_collect: int, batch_size: int, limit: int, train_device: str, resume_path: str, seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # === initializes data ===
    if xai_method:
        xai_clinician = torch.load(xai_policy, map_location=train_device)["model"].eval().to("cpu")
        dataset, train_dataloader, _ = get_xai_debate_dataset(method=xai_method, clinician=xai_clinician, background_dataset_path=xai_bg_dataset, num_arguments=xai_num_arguments, preferences_path=train_dataset_path, batch_size=1, num_workers=0, limit=limit)
        _, test_dataloader, _ = get_xai_debate_dataset(method=xai_method, clinician=xai_clinician, background_dataset_path=xai_bg_dataset, num_arguments=xai_num_arguments, preferences_path=test_dataset_path, batch_size=1, num_workers=0, limit=limit)
    else:
        dataset, train_dataloader, _ = get_debate_dataset(load_path=train_dataset_path, batch_size=1, num_workers=0, limit=limit)
        _, test_dataloader, _ = get_debate_dataset(load_path=test_dataset_path, batch_size=1, num_workers=0, limit=limit)

    action_dim = dataset[:][1].unique().size(0)
    state_dim = dataset[:][0].size(-1)

    # === sets up models ===
    judge = torch.load(judge_path, map_location="cpu")
    judge = judge.eval()

    if opponent_path is not None:
        opponent_data = torch.load(opponent_path, map_location=torch.device(train_device))
        opponent = opponent_data["model"]
        opponent.ret_rms = opponent_data["ret_rms"]
        opponent = opponent.eval()
    else:
        opponent = None

    # === training callback functions ===
    def get_env_factory(dataloader: DataLoader):
        def factory():
            return SepsisArgumentationEnv(inverse_reward=True, xai=xai_method is not None, dataloader=dataloader, opponent=opponent, start_player=1 if xai_method else None, judge=judge, action_dim=action_dim, state_dim=state_dim, num_arguments=num_arguments, propose_evidence_upfront=propose_evidence_upfront, use_judge_diff_as_reward=use_judge_diff_as_reward)
        return factory

    def save_best_fn(policy):
        torch.save({"model": policy, "weights": policy.state_dict(), "ret_rms": policy.ret_rms, "optim": policy.optim.state_dict()}, join(artifacts_dir, "confuser.pt"))

    def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int):
        torch.save({"model": policy, "weights": policy.state_dict(), "ret_rms": policy.ret_rms, "optim": policy.optim.state_dict()}, join(artifacts_dir, f"confuser.checkpoint.pt"))

    # === sets up train and test environments ===
    train_envs = DummyVectorEnv([get_env_factory(train_dataloader) for _ in range(num_train_envs)])
    test_envs = DummyVectorEnv([get_env_factory(test_dataloader) for _ in range(num_test_envs)])
    train_envs.seed(seed)
    test_envs.seed(seed)

    # === sets up policy ===
    policy = make_argumentator(
        state_dim=state_dim,
        num_actions=action_dim,
        hidden_dim=hidden_dim,
        hidden_depth=hidden_depth,
        lr=lr,
        lr_schedule=lr_schedule,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        clip_range=clip_range,
        gae_lambda=gae_lambda,
        max_grad_norm=max_grad_norm,
        gamma=gamma,
        device=train_device,
        normalize_rewards=normalize_rewards,
        ortho_init=ortho_init,
    )

    # === restores policy and optimizer state, if requested ===
    if resume_path:
        if exists(resume_path):
            checkpoint = torch.load(resume_path, map_location=train_device)
            policy.load_state_dict(checkpoint["weights"])
            policy.optim.load_state_dict(checkpoint["optim"])
            print(f"Loaded model checkpoint from '{resume_path}'")
        else:
            print(f"Checkpoint not found under '{resume_path}'")

    # === sets up train and test data collectors ===
    train_collector = Collector(policy, train_envs, VectorReplayBuffer(1e5, len(train_envs)))
    test_collector = Collector(policy, test_envs)
    train_collector.collect(n_step=batch_size * num_train_envs)

    # === sets up additional logging ===
    writer = SummaryWriter(join(artifacts_dir, "tensorboard"))
    logger = TensorboardLogger(writer, save_interval=1)
    log_dict = {}

    # === sets up trainer ===
    trainer = OnpolicyTrainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=epochs,
        step_per_epoch=step_per_epoch,
        repeat_per_collect=repeat_per_collect,
        step_per_collect=step_per_collect,
        episode_per_collect=episode_per_collect,
        batch_size=batch_size,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        logger=logger,
        episode_per_test=100,
        test_in_train=False,
        verbose=True,
    )

    # === starts training ===
    for epoch, epoch_stat, _ in trainer:
        epoch_stat = dict(**epoch_stat, epoch=epoch)
        for k in epoch_stat:
            if k in log_dict:
                log_dict[k].append(epoch_stat[k])
            else:
                log_dict[k] = [epoch_stat[k]]
        pickle.dump(log_dict, open(join(artifacts_dir, "logs.pkl"), "wb"))


if __name__ == "__main__":
    main()
