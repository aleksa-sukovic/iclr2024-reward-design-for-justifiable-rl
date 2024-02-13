"""
This script processes the MIMIC-III cohort into a format that can be used to learn RL policies.
    - The *.csv file representing sepsis cohort as described in (Komorowski2018 et. al) is extracted using https://github.com/microsoft/mimic_sepsis
    - The cohort is split into train, validation and test sets
    - This script was inspired by https://github.com/MLforHealth/rl_representations
"""

import click
import torch
import pandas as pd
import numpy as np

from os.path import join
from sklearn.model_selection import train_test_split
from tensordict import TensorDict
from typing import List, Optional, Tuple

from argo.datasets.sepsis import (
    SEPSIS_FEATURES_IGNORED,
    SEPSIS_FEATURES_DEMOGRAPHIC,
    SEPSIS_FEATURES_OBSERVATIONAL,
    SEPSIS_FEATURES_ACTION,
    SEPSIS_FEATURES_REWARD
)


def _get_batch(data: pd.DataFrame, trajectories: List[int], horizon: int = 21, num_actions: int = 25, device: torch.device = torch.device("cpu")) -> TensorDict:
    items: List[dict] = []
    action_map = torch.eye(num_actions).to(device)

    for id in trajectories:
        trajectory = data[data["traj"] == id].sort_values(by="step")
        item_dem = torch.Tensor(trajectory[SEPSIS_FEATURES_DEMOGRAPHIC].values).to(device)
        item_dem_prot = torch.Tensor(trajectory[SEPSIS_FEATURES_DEMOGRAPHIC].values).to(device)
        item_obs = torch.Tensor(trajectory[SEPSIS_FEATURES_OBSERVATIONAL].values).to(device)
        item_obs_prot = torch.Tensor(trajectory[SEPSIS_FEATURES_OBSERVATIONAL].values).to(device)
        item_act = torch.Tensor(trajectory[SEPSIS_FEATURES_ACTION].values.astype(np.int32)).long().view(-1, 1).to(device)
        item_rew = torch.Tensor(trajectory[SEPSIS_FEATURES_REWARD].values).to(device)
        item_length = item_obs.shape[0] # number of steps for this trajectory (i.e., patient)
        if item_length <= 1:
            continue
        items.append({
            "dem": torch.cat((item_dem, torch.zeros((horizon - item_length, item_dem.shape[1]), dtype=torch.float).to(device))),
            "dem_prot": torch.cat((item_dem_prot, torch.zeros((horizon - item_length, item_dem_prot.shape[1]), dtype=torch.float).to(device))),
            "obs": torch.cat((item_obs, torch.zeros((horizon - item_length, item_obs.shape[1]), dtype=torch.float).to(device))),
            "obs_prot": torch.cat((item_obs_prot, torch.zeros((horizon - item_length, item_obs_prot.shape[1]), dtype=torch.float).to(device))),
            "actions": torch.cat((action_map[item_act].squeeze(1), torch.zeros((horizon - item_length, num_actions), dtype=torch.float).to(device))),
            "rewards": torch.cat((item_rew, torch.zeros((horizon - item_length), dtype=torch.float).to(device))),
            "steps": torch.Tensor(range(horizon)).to(device),
            "trajectory_id": torch.Tensor([id]).int().to(device),
            "trajectory_length": torch.Tensor([item_length]).to(device),
        })

    return TensorDict({
        "dem": torch.cat(tuple(map(lambda i: i["dem"].unsqueeze(dim=0), items))),
        "dem_prot": torch.cat(tuple(map(lambda i: i["dem_prot"].unsqueeze(dim=0), items))),
        "obs": torch.cat(tuple(map(lambda i: i["obs"].unsqueeze(dim=0), items)), dim=0),
        "obs_prot": torch.cat(tuple(map(lambda i: i["obs_prot"].unsqueeze(dim=0), items)), dim=0),
        "actions": torch.cat(tuple(map(lambda i: i["actions"].unsqueeze(dim=0), items)), dim=0),
        "rewards": torch.cat(tuple(map(lambda i: i["rewards"].unsqueeze(dim=0), items)), dim=0),
        "steps": torch.cat(tuple(map(lambda i: i["steps"].unsqueeze(dim=0), items)), dim=0),
        "trajectory_id": torch.cat(tuple(map(lambda i: i["trajectory_id"].unsqueeze(dim=0), items)), dim=0),
        "trajectory_length": torch.cat(tuple(map(lambda i: i["trajectory_length"].unsqueeze(dim=0), items)), dim=0),
    }, batch_size=len(items), device=device)


@click.group()
def main():
    pass


@main.command()
@click.option("--artifacts-dir", required=True, help="Path to directory where generated artifacts will be saved.")
@click.option("--sepsis-cohort", required=True, help="Path to the .csv file of sepsis cohort.")
@click.option("--train-chunk", default=0.7, type=float, help="Percentage of total data, used for training.")
@click.option("--val-chunk", default=0.5, type=float, help="Percentage of validation chunk, used for testing.")
@click.option("--train-file", default="train_dict.pt", help="File name of the train split.")
@click.option("--val-file", default="val_dict.pt", help="File name of the validation split.")
@click.option("--test-file", default="test_dict.pt", help="File name of the test split.")
@click.option("--include-action", "-a", type=int, multiple=True, help="Include only samples with specified actions.")
@click.option("--limit", default=None, type=int, help="Maximum number of rows to read from the input file.")
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default="cpu", help="Target device on which to run the training.")
@click.option("--seed", type=int, default=25, help="Random seed to use for reproducibility.")
def generate(artifacts_dir: str, sepsis_cohort: str, train_chunk: float, val_chunk: float, train_file: str, val_file: str, test_file: str, limit: Optional[int], include_action: Tuple[int], device: str, seed: int):
    df = pd.read_csv(sepsis_cohort, nrows=limit)
    df = df.drop(SEPSIS_FEATURES_IGNORED, axis=1)
    if include_action:
        df = df[df["a:action"].isin(list(include_action))]
        df = df.replace({k: v for k, v in zip(include_action, range(len(include_action)))})

    # Splits the dataset into (train, val, test) splits, stratified by
    # patient's survival outcome where outcome is survival (+1.0) or
    # death (-1.0) at the end of trajectory.
    outcomes = df.groupby("traj")["r:reward"].sum()

    x = outcomes.index.values
    y = outcomes.values

    x_train, x_val, y_train, y_val = train_test_split(x, y, stratify=y, test_size=1 - train_chunk, random_state=seed)
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, stratify=y_val, test_size=val_chunk, random_state=seed)

    train_data = df[df["traj"].isin(x_train)]
    train_trajectories = train_data["traj"].unique()

    val_data = df[df["traj"].isin(x_val)]
    val_trajectories = val_data["traj"].unique()

    test_data = df[df["traj"].isin(x_test)]
    test_trajectories = test_data["traj"].unique()

    # Defines dataset features
    num_actions = 25 if not include_action else len(include_action)
    horizon = 21
    device = torch.device(device)

    # Define splits
    train_data = _get_batch(data=train_data, trajectories=train_trajectories, horizon=horizon, num_actions=num_actions, device=device)
    val_data = _get_batch(data=val_data, trajectories=val_trajectories, horizon=horizon, num_actions=num_actions, device=device)
    test_data = _get_batch(data=test_data, trajectories=test_trajectories, horizon=horizon, num_actions=num_actions, device=device)

    # Save splits
    torch.save(train_data, join(artifacts_dir, train_file))
    torch.save(val_data, join(artifacts_dir, val_file))
    torch.save(test_data, join(artifacts_dir, test_file))


if __name__ == "__main__":
    main()
