import torch
import click
import random
import numpy as np

from copy import deepcopy
from pprint import pprint
from os.path import join, exists
from torch.utils.tensorboard import SummaryWriter
from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger

from argo.envs.sepsis import SepsisArgumentationEnv
from argo.datasets.sepsis import get_debate_dataset
from argo.models.argumentator import MaskedPPO, make_argumentator
from argo.models.judge import Judge


@click.group()
def main():
    pass


@main.command()
@click.option("--artifacts-dir", required=True, type=click.Path(), help="Path to directory where generated artifacts will be saved.")
@click.option("--judge-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="Path to exported PyTorch judge model.")
@click.option("--dataset-path", required=True, type=click.Path(file_okay=True, dir_okay=False), help="Path to dataset exported during judge training (see `scripts.judge.train`).")
@click.option("--hidden-dim", default=256, type=int, help="Dimension of the policy network hidden layer.")
@click.option("--hidden-depth", default=2, type=int, help="Number of hidden layers.")
@click.option("--num-arguments", default=6, type=int, help="Number of state-features (arguments) to be used.")
@click.option("--num-train-envs", default=1, help="Number of parallel environments to use.")
@click.option("--num-test-envs", default=1, help="Number of parallel environments to use.")
@click.option("--epochs", default=100, help="Number of training epochs to run. Total of epochs * step_per_epoch steps will be executed.")
@click.option("--step-per-epoch", default=4096, help="Number of transitions to collect per each epoch.")
@click.option("--step-per-collect", default=None, help="Number of transitions to collect before performing the network update.")
@click.option("--episode-per-collect", default=256, type=int, help="Number of complete episodes to collect before performing the network update.")
@click.option("--repeat-per-collect", default=10, help="Number of policy learning updates to perform per one collect batch.")
@click.option("--lr", default=5e-4, type=float, help="Policy learning rate.")
@click.option("--lr-schedule", default="constant", type=click.Choice(["constant", "step"]), help="Learning rate scheduler.")
@click.option("--ent-coef", default=1e-4, type=float, help="Strength of entropy regularization.")
@click.option("--clip-range", default=0.2, type=float, help="PPO clip range (epsilon).")
@click.option("--gamma", default=0.8, type=float, help="Discount factor.")
@click.option("--gae-lambda", default=0.95, type=float, help="Factor for trade-off of bias vs variance for Generalized Advantage Estimator.")
@click.option("--vf-coef", default=0.5, type=float, help="Value function coefficient for the loss calculation.")
@click.option("--max-grad-norm", default=0.5, type=float, help="The maximum value for the gradient clipping.")
@click.option("--normalize-rewards", default=False, type=click.BOOL, help="Determines if rewards passed to the agents are normalized to have std (close to) 1.")
@click.option("--ortho-init", default=True, type=click.BOOL, help="Determines if orthogonal layer initialization is used.")
@click.option("--batch-size", default=64, type=int, help="Number of transitions to use when performing network update.")
@click.option("--limit", default=None, type=int, help="Maximum number of patient transitions to use for training.")
@click.option("--train-device", type=str, default="cpu", help="Target device on which to run the training.")
@click.option("--resume-path", type=click.Path(exists=True, file_okay=True), required=False, default=None, help="Indicates if training should be resumed from specified checkpoint.")
@click.option("--seed", type=int, default=25, help="Random seed to use for reproducibility.")
def train(artifacts_dir: str, judge_path: str, dataset_path: str, hidden_dim: int, hidden_depth: int, num_arguments: int, lr: float, lr_schedule: str, ent_coef: float, clip_range: float, gae_lambda: float, vf_coef: float, max_grad_norm: float, normalize_rewards: bool, gamma: float, ortho_init: str, num_train_envs: int, num_test_envs: int, epochs: int, step_per_epoch: int, step_per_collect: int, episode_per_collect: int, repeat_per_collect: int, batch_size: int, limit: int, train_device: str, resume_path: str, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # === initializes data ===
    dataset, dataloader, _ = get_debate_dataset(load_path=dataset_path, batch_size=1, num_workers=0, limit=limit)
    action_dim = dataset[:][1].unique().size(0)
    state_dim = dataset[:][0].size(-1)

    # === initializes judge model ===
    judge = torch.load(judge_path, map_location=torch.device("cpu"))
    judge = judge.eval()

    # === sets up environment ===
    def get_env():
        return SepsisArgumentationEnv(dataloader=dataloader, judge=judge, action_dim=action_dim, state_dim=state_dim, num_arguments=num_arguments)

    # === sets up training callback functions ===
    def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int):
        torch.save({"model": policy, "weights": policy.state_dict(), "ret_rms": policy.ret_rms, "optim": policy.optim.state_dict()}, join(artifacts_dir, "argumentator.isolated.checkpoint.pt"))

    def save_best_fn(policy):
        torch.save({"model": policy, "weights": policy.state_dict(), "ret_rms": policy.ret_rms, "optim": policy.optim.state_dict()}, join(artifacts_dir, "argumentator.isolated.pt"))

    # === sets up train and test environments ===
    train_envs = DummyVectorEnv([get_env for _ in range(num_train_envs)])
    test_envs = DummyVectorEnv([get_env for _ in range(num_test_envs)])
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
            policy.ret_rms = checkpoint["ret_rms"]
            print(f"Loaded model checkpoint from '{resume_path}'")
        else:
            print(f"Checkpoint not found under '{resume_path}'")

    # === sets up train and test data collectors ===
    train_collector = Collector(policy, train_envs, VectorReplayBuffer(1e5, len(train_envs)))
    test_collector = Collector(policy, test_envs)
    train_collector.collect(n_step=batch_size * num_train_envs)

    # === sets up additional Tensorboard logging, also used for resuming training ===
    writer = SummaryWriter(join(artifacts_dir, "tensorboard"))
    logger = TensorboardLogger(writer, save_interval=1)

    # === trains the agent ===
    result = onpolicy_trainer(
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
    pprint(result)


@main.command()
@click.option("--artifacts-dir", required=True, type=click.Path(), help="Path to directory where generated artifacts will be saved.")
@click.option("--judge-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="Path to exported PyTorch judge model.")
@click.option("--dataset-path", required=True, type=click.Path(file_okay=True, dir_okay=False), help="Path to dataset exported during judge training (see `scripts.judge.train`).")
@click.option("--hidden-dim", default=256, type=int, help="Dimension of the policy network hidden layer.")
@click.option("--hidden-depth", default=2, type=int, help="Number of hidden layers.")
@click.option("--num-arguments", default=6, type=int, help="Number of state-features (arguments) to be used.")
@click.option("--num-train-envs", default=1, help="Number of parallel environments to use.")
@click.option("--num-test-envs", default=1, help="Number of parallel environments to use.")
@click.option("--epochs", default=100, help="Number of training epochs to run. Total of epochs * step_per_epoch steps will be executed.")
@click.option("--generations", default=5, help="Number of self-play generations.")
@click.option("--step-per-epoch", default=4096, help="Number of transitions to collect per each epoch.")
@click.option("--step-per-collect", default=None, type=int, help="Number of transitions to collect before performing the network update.")
@click.option("--episode-per-collect", default=384, type=int, help="Number of complete episodes to collect before performing the network update.")
@click.option("--repeat-per-collect", default=10, help="Number of policy learning updates to perform per one collect batch.")
@click.option("--lr", default=5e-4, type=float, help="Policy learning rate.")
@click.option("--lr-schedule", default="constant", type=click.Choice(["constant", "step"]), help="Learning rate scheduler.")
@click.option("--ent-coef", default=1e-4, type=float, help="Strength of entropy regularization.")
@click.option("--clip-range", default=0.2, type=float, help="PPO clip range (epsilon).")
@click.option("--gamma", default=0.8, type=float, help="Discount factor.")
@click.option("--gae-lambda", default=0.95, type=float, help="Factor for trade-off of bias vs variance for Generalized Advantage Estimator.")
@click.option("--vf-coef", default=0.5, type=float, help="Value function coefficient for the loss calculation.")
@click.option("--max-grad-norm", default=0.5, type=float, help="The maximum value for the gradient clipping.")
@click.option("--normalize-rewards", default=False, type=click.BOOL, help="Determines if rewards passed to the agents are normalized to have std (close to) 1.")
@click.option("--use-judge-diff-as-reward", default=False, type=click.BOOL, help="Determines if difference between judge's evaluations is used as a reward.")
@click.option("--ortho-init", default=True, type=click.BOOL, help="Determines if orthogonal layer initialization is used.")
@click.option("--batch-size", default=64, type=int, help="Number of transitions to use when performing network update.")
@click.option("--limit", default=None, type=int, help="Maximum number of patient transitions to use for training.")
@click.option("--train-device", type=str, default="cpu", help="Target device on which to run the training.")
@click.option("--resume-path", type=click.Path(exists=True, file_okay=True), required=False, default=None, help="Indicates if training should be resumed from specified checkpoint.")
@click.option("--seed", type=int, default=25, help="Random seed to use for reproducibility.")
def train_debate(artifacts_dir: str, judge_path: str, dataset_path: str, hidden_dim: int, hidden_depth: int, num_arguments: int, lr: float, lr_schedule: str, ent_coef: float, clip_range: float, gae_lambda: float, vf_coef: float, max_grad_norm: float, normalize_rewards: bool, use_judge_diff_as_reward: bool, gamma: float, ortho_init: str, num_train_envs: int, num_test_envs: int, epochs: int, generations: int, step_per_epoch: int, step_per_collect: int, episode_per_collect: int, repeat_per_collect: int, batch_size: int, limit: int, train_device: str, resume_path: str, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # === initializes data ===
    dataset, dataloader, _ = get_debate_dataset(load_path=dataset_path, batch_size=1, num_workers=0, limit=limit)
    action_dim = dataset[:][1].unique().size(0)
    state_dim = dataset[:][0].size(-1)

    # === initializes judge model ===
    judge = torch.load(judge_path, map_location=torch.device("cpu"))
    judge = judge.eval()

    # === sets up environment ===
    def get_env(judge: Judge, opponent: MaskedPPO):
        def factory():
            return SepsisArgumentationEnv(dataloader=dataloader, judge=judge, opponent=opponent, action_dim=action_dim, state_dim=state_dim, num_arguments=num_arguments, use_judge_diff_as_reward=use_judge_diff_as_reward)
        return factory

    # === sets up training callback functions ===
    def get_save_checkpoint_fn(model: MaskedPPO, name: str):
        def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int):
            torch.save({"model": model, "weights": model.state_dict(), "ret_rms": model.ret_rms, "optim": model.optim.state_dict()}, join(artifacts_dir, f"{name}.debate.checkpoint.pt"))
        return save_checkpoint_fn

    def get_save_best_fn(model: MaskedPPO, name: str):
        def save_best_fn(_):
            torch.save({"model": model, "weights": model.state_dict(), "ret_rms": model.ret_rms, "optim": model.optim.state_dict(), "ret_rms": model.ret_rms}, join(artifacts_dir, f"{name}.debate.pt"))
        return save_best_fn

    # === sets up collectors ===
    def get_collectors(p1: MaskedPPO, p2: MaskedPPO, judge: Judge):
        train_envs = DummyVectorEnv([get_env(judge, p2) for _ in range(num_train_envs)])
        test_envs = DummyVectorEnv([get_env(judge, p2) for _ in range(num_test_envs)])
        train_envs.seed(seed)
        test_envs.seed(seed)

        train_collector = Collector(p1, train_envs, VectorReplayBuffer(1e5, len(train_envs)))
        test_collector = Collector(p1, test_envs)
        return train_collector, test_collector

    # === sets up policies ===
    def get_agent():
        return make_argumentator(
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

    argumentator, opponent = get_agent().train(), get_agent().eval()

    # === restores policy and optimizer state, if requested ===
    if resume_path:
        if exists(resume_path):
            checkpoint = torch.load(resume_path, map_location=train_device)

            argumentator.load_state_dict(deepcopy(checkpoint["weights"]))
            argumentator.optim.load_state_dict(checkpoint["optim"])
            argumentator.ret_rms = deepcopy(checkpoint["ret_rms"])

            opponent.load_state_dict(deepcopy(checkpoint["weights"]))
            opponent.ret_rms = deepcopy(checkpoint["ret_rms"])
            print(f"Loaded model checkpoint from '{resume_path}'")
        else:
            print(f"Checkpoint not found under '{resume_path}'")

    # === specifies main train function ===
    def train_fn(p1: MaskedPPO, p2: MaskedPPO, epochs: int, log_name: str):
        train_collector, test_collector = get_collectors(p1, p2, judge)
        train_collector.collect(n_step=batch_size * num_train_envs)

        writer = SummaryWriter(join(artifacts_dir, f"{log_name}-tensorboard"))
        logger = TensorboardLogger(writer, save_interval=1)

        onpolicy_trainer(
            p1,
            train_collector,
            test_collector,
            max_epoch=epochs,
            step_per_epoch=step_per_epoch,
            repeat_per_collect=repeat_per_collect,
            step_per_collect=step_per_collect,
            episode_per_collect=episode_per_collect,
            batch_size=batch_size,
            save_best_fn=get_save_best_fn(p1, log_name),
            save_checkpoint_fn=get_save_checkpoint_fn(p1, log_name),
            logger=logger,
            resume_from_log=False,
            episode_per_test=10,
            test_in_train=False,
            verbose=True,
            show_progress=True,
        )

    # === trains the agent ===
    for gen in range(1, generations + 1):
        train_fn(argumentator, opponent, epochs=epochs, log_name="argumentator")

        opponent = get_agent()
        opponent.load_state_dict(torch.load(join(artifacts_dir, "argumentator.debate.checkpoint.pt"))["weights"])
        opponent.eval()

        print(f"Generation {gen} completed.", flush=True)


@main.command()
@click.option("--artifacts-dir", required=True, type=click.Path(), help="Path to directory where generated artifacts will be saved.")
@click.option("--judge-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), help="Path to exported PyTorch judge model.")
@click.option("--dataset-path", required=True, type=click.Path(file_okay=True, dir_okay=False), help="Path to dataset exported during judge training (see `scripts.judge.train`).")
@click.option("--hidden-dim", default=256, type=int, help="Dimension of the policy network hidden layer.")
@click.option("--hidden-depth", default=2, type=int, help="Number of hidden layers.")
@click.option("--num-arguments", default=6, type=int, help="Number of state-features (arguments) to be used.")
@click.option("--num-train-envs", default=1, help="Number of parallel environments to use.")
@click.option("--num-test-envs", default=1, help="Number of parallel environments to use.")
@click.option("--epochs-argumentator", default=100, help="Number of training epochs to train the argumentator.")
@click.option("--epochs-confuser", default=100, help="Number of training epochs to train the confuser.")
@click.option("--generations", default=5, help="Number of alternating optimization rounds to run.")
@click.option("--step-per-epoch", default=4096, help="Number of transitions to collect per each epoch.")
@click.option("--step-per-collect", default=None, type=int, help="Number of transitions to collect before performing the network update.")
@click.option("--episode-per-collect", default=384, type=int, help="Number of complete episodes to collect before performing the network update.")
@click.option("--repeat-per-collect", default=10, help="Number of policy learning updates to perform per one collect batch.")
@click.option("--lr", default=5e-4, type=float, help="Policy learning rate.")
@click.option("--lr-schedule", default="constant", type=click.Choice(["constant", "step"]), help="Learning rate scheduler.")
@click.option("--ent-coef", default=1e-4, type=float, help="Strength of entropy regularization.")
@click.option("--clip-range", default=0.2, type=float, help="PPO clip range (epsilon).")
@click.option("--gamma", default=0.8, type=float, help="Discount factor.")
@click.option("--gae-lambda", default=0.95, type=float, help="Factor for trade-off of bias vs variance for Generalized Advantage Estimator.")
@click.option("--vf-coef", default=0.5, type=float, help="Value function coefficient for the loss calculation.")
@click.option("--max-grad-norm", default=0.5, type=float, help="The maximum value for the gradient clipping.")
@click.option("--normalize-rewards", default=False, type=click.BOOL, help="Determines if rewards passed to the agents are normalized to have std (close to) 1.")
@click.option("--use-judge-diff-as-reward", default=False, type=click.BOOL, help="Determines if difference between judge's evaluations is used as a reward.")
@click.option("--ortho-init", default=True, type=click.BOOL, help="Determines if orthogonal layer initialization is used.")
@click.option("--batch-size", default=64, type=int, help="Number of transitions to use when performing network update.")
@click.option("--limit", default=None, type=int, help="Maximum number of patient transitions to use for training.")
@click.option("--train-device", type=str, default="cpu", help="Target device on which to run the training.")
@click.option("--argumentator-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), help="Path to the exported confuser model weights to resume training from.")
@click.option("--confuser-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), help="Path to the exported confuser model weights to resume training from.")
@click.option("--seed", type=int, default=25, help="Random seed to use for reproducibility.")
def train_minimax(artifacts_dir: str, judge_path: str, dataset_path: str, hidden_dim: int, hidden_depth: int, num_arguments: int, lr: float, lr_schedule: str, ent_coef: float, clip_range: float, gae_lambda: float, vf_coef: float, max_grad_norm: float, normalize_rewards: bool, use_judge_diff_as_reward: bool, gamma: float, ortho_init: str, num_train_envs: int, num_test_envs: int, epochs_argumentator: int, epochs_confuser: int, generations: int, step_per_epoch: int, step_per_collect: int, episode_per_collect: int, repeat_per_collect: int, batch_size: int, limit: int, train_device: str, argumentator_path: str, confuser_path: str, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # === initializes data ===
    dataset, dataloader, _ = get_debate_dataset(load_path=dataset_path, batch_size=1, num_workers=0, limit=limit)
    action_dim = dataset[:][1].unique().size(0)
    state_dim = dataset[:][0].size(-1)

    # === initializes judge model ===
    judge = torch.load(judge_path, map_location=torch.device("cpu"))
    judge = judge.eval()

    # === sets up environment ===
    def get_env(judge: Judge, opponent: MaskedPPO, inverse_reward: bool = False):
        def factory():
            return SepsisArgumentationEnv(dataloader=dataloader, judge=judge, opponent=opponent, action_dim=action_dim, state_dim=state_dim, num_arguments=num_arguments, inverse_reward=inverse_reward, use_judge_diff_as_reward=use_judge_diff_as_reward)
        return factory

    # === sets up training callback functions ===
    def get_save_best_fn(model: MaskedPPO, name: str):
        def save_best_fn(policy):
            torch.save({"model": model, "weights": model.state_dict(), "ret_rms": model.ret_rms, "optim": model.optim.state_dict(), "ret_rms": model.ret_rms}, join(artifacts_dir, f"{name}.debate.pt"))
        return save_best_fn

    def get_save_checkpoint_fn(model: MaskedPPO, name: str):
        def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int):
            torch.save({"model": model, "weights": model.state_dict(), "ret_rms": model.ret_rms, "optim": model.optim.state_dict()}, join(artifacts_dir, f"{name}.debate.checkpoint.pt"))
        return save_checkpoint_fn

    # === sets up collectors ===
    def get_collectors(p1: MaskedPPO, p2: MaskedPPO, judge: Judge, inverse_reward: bool):
        train_envs = DummyVectorEnv([get_env(judge, p2, inverse_reward) for _ in range(num_train_envs)])
        test_envs = DummyVectorEnv([get_env(judge, p2, inverse_reward) for _ in range(num_test_envs)])
        train_envs.seed(seed)
        test_envs.seed(seed)

        train_collector = Collector(p1, train_envs, VectorReplayBuffer(1e5, len(train_envs)), exploration_noise=True)
        test_collector = Collector(p1, test_envs, exploration_noise=True)
        return train_collector, test_collector

    # === sets up policies ===
    def get_agent():
        return make_argumentator(
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

    argumentator, confuser = get_agent().train(), get_agent().train()

    # === loads weights if available ===
    if argumentator_path:
        data = torch.load(argumentator_path)
        argumentator.load_state_dict(data["weights"])
        argumentator.ret_rms = data["ret_rms"]
    if confuser_path:
        data = torch.load(confuser_path)
        confuser.load_state_dict(data["weights"])
        confuser.ret_rms = data["ret_rms"]

    # === specifies main train function ===
    def train_fn(p1: MaskedPPO, p2: MaskedPPO, epochs: int, inverse_reward: bool, log_name: str):
        train_collector, test_collector = get_collectors(p1, p2, judge, inverse_reward=inverse_reward)
        train_collector.collect(n_step=batch_size * num_train_envs)

        writer = SummaryWriter(join(artifacts_dir, f"{log_name}-tensorboard"))
        logger = TensorboardLogger(writer, save_interval=1)

        onpolicy_trainer(
            p1,
            train_collector,
            test_collector,
            max_epoch=epochs,
            step_per_epoch=step_per_epoch,
            repeat_per_collect=repeat_per_collect,
            step_per_collect=step_per_collect,
            episode_per_collect=episode_per_collect,
            batch_size=batch_size,
            save_best_fn=get_save_best_fn(p1, log_name),
            save_checkpoint_fn=get_save_checkpoint_fn(p1, log_name),
            logger=logger,
            resume_from_log=False,
            episode_per_test=50,
            test_in_train=False,
            verbose=True,
            show_progress=True,
        )

    # === trains the agent ===
    for gen in range(1, generations + 1):
        # trains confuser, keeping the argumentator fixed
        confuser = confuser.train()
        argumentator = argumentator.eval()
        train_fn(confuser, argumentator, epochs=epochs_confuser, inverse_reward=True, log_name="confuser")

        # trains argumentator, keeping the confuser fixed
        argumentator = argumentator.train()
        confuser = confuser.eval()
        train_fn(argumentator, confuser, epochs=epochs_argumentator, inverse_reward=False, log_name="argumentator")

        print(f"Generation {gen} completed.", flush=True)


if __name__ == "__main__":
    main()
