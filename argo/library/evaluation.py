import tqdm
import torch
import random
import numpy as np
import torch.nn.functional as F

from typing import List, Optional, Tuple, TYPE_CHECKING, Union
from torch.utils.data import DataLoader
from tianshou.policy import DQNPolicy, BasePolicy
from tianshou.data import Batch

if TYPE_CHECKING:
    from argo.models.judge import Judge
    from argo.models.argumentator import MaskedPPO
    from argo.models.clinician import ClinicianPolicy


def wis(target: Union["DQNPolicy", List["DQNPolicy"]], behavioral: "ClinicianPolicy", patients: DataLoader, discount: float = 0.99, device: str = "cuda") -> float:
    if isinstance(target, list):
        return np.mean([wis(t, behavioral, patients, discount, device) for t in target])

    wis_return = 0.0
    wis_weight = 0.0

    for s_t, a_t, r_t, _, _ in patients:
        s_t, a_t, r_t = s_t.to(device), a_t.argmax(dim=-1, keepdim=True).to(device), r_t.to(device)

        mask = (s_t == 0).all(dim=-1)
        horizon = s_t.size(1)

        # calculates discounted cumulative reward
        discounts = discount**torch.arange(horizon)[None, :].to(device)
        rew_discounted = (discounts * r_t).sum(dim=-1).squeeze()

        # calculates probabilities of actions as per behavioral and target policies
        with torch.no_grad():
            behavioral = behavioral.eval()
            p_behavioral = F.softmax(behavioral(s_t.flatten(end_dim=1)), dim=-1)
            p_behavioral = p_behavioral.gather(dim=1, index=a_t.flatten()[:, None]).reshape(s_t.shape[:2])

            target = target.eval()
            p_target = target(Batch(obs=s_t.flatten(end_dim=1), info={})).logits
            p_target = F.softmax(p_target, dim=-1)
            p_target = p_target.gather(dim=1, index=a_t.flatten()[:, None]).reshape(s_t.shape[:2])

        # handles violation of the WIS estimator assumptions: pi(a|s) > 0 -> b(a|s) > 0
        # c.f. Sutton et.al, Reinforcement learning: an introduction, page 103
        if not (p_behavioral > 0).all():
            p_behavioral[p_behavioral == 0] = 0.1

        # eliminates spurious probabilities due to padded observations
        p_behavioral[mask] = 1.0
        p_target[mask] = 1.0

        # c.f. Jagannatha A. et.al, Towards High Confidence Off-Policy Reinforcement Learning for Clinical Applications
        is_ratio = torch.clamp((p_target / p_behavioral).prod(axis=1), 1e-8, 1e2)

        # appends wis estimate
        wis_rewards = is_ratio.cpu() * rew_discounted.cpu()
        wis_return += wis_rewards.sum().item()
        wis_weight += is_ratio.sum().item()

    return wis_return / wis_weight


def wis_clinician(clinician: "ClinicianPolicy", patients: DataLoader, discount: float = 0.99, device: str = "cpu") -> float:
    wis_return = 0.0
    wis_weight = 0.0

    for s_t, a_t, r_t, _, _ in patients:
        s_t, a_t, r_t = s_t.to(device), a_t.argmax(dim=-1, keepdim=True).to(device), r_t.to(device)
        s_t, a_t, r_t = s_t[:, :-1, :], a_t[:, :-1], r_t[:, :-1]

        mask = (s_t == 0).all(dim=-1)
        horizon = s_t.size(1)

        # calculates discounted cumulative reward
        discounts = discount**torch.arange(horizon)[None, :].to(device)
        rew_discounted = (discounts * r_t).sum(dim=-1).squeeze()

        # calculates probabilities of actions as per behavioral and target policies
        with torch.no_grad():
            clinician = clinician.eval()

            p_behavioral = F.softmax(clinician(s_t.flatten(end_dim=1)), dim=-1)
            p_behavioral = p_behavioral.gather(dim=1, index=a_t.flatten()[:, None]).reshape(s_t.shape[:2])

            p_clinician = F.softmax(clinician(s_t.flatten(end_dim=1)), dim=-1)
            p_clinician = p_clinician.gather(dim=1, index=a_t.flatten()[:, None]).reshape(s_t.shape[:2])

        # handles violation of the WIS estimator assumptions: pi(a|s) > 0 -> b(a|s) > 0
        # c.f. Sutton et.al, Reinforcement learning: an introduction, page 103
        if not (p_behavioral > 0).all():
            p_behavioral[p_behavioral == 0] = 0.1

        # eliminates spurious probabilities due to padded observations
        p_behavioral[mask] = 1.0
        p_clinician[mask] = 1.0

        # c.f. Jagannatha A. et.al, Towards High Confidence Off-Policy Reinforcement Learning for Clinical Applications
        is_ratio = torch.clamp((p_clinician / p_behavioral).prod(axis=1), 1e-8, 1e2)

        # appends wis estimate
        wis_rewards = is_ratio.cpu() * rew_discounted.cpu()
        wis_return += wis_rewards.sum().item()
        wis_weight += is_ratio.sum().item()

    return wis_return / wis_weight


def jstf(justifiable: "DQNPolicy", baseline: "DQNPolicy", dataloader: DataLoader, argumentator: "MaskedPPO", judge: "Judge", num_arguments: int = 6, device: str = "cuda", report_percentage: bool = True) -> Union[float, List[int]]:
    wins = torch.empty((0, ), dtype=torch.float32, device=device)

    for s_t, _, _, _, _ in dataloader:
        s_t = s_t.to(device)
        a_justifiable = torch.tensor(justifiable(Batch(obs=s_t, info={})).act, device=device)
        a_baseline = torch.tensor(baseline(Batch(obs=s_t, info={})).act, device=device)

        rew_target, rew_baseline = run_debate(s_t=s_t, a_p1=a_justifiable, a_p2=a_baseline, judge=judge, argumentator=argumentator, num_arguments=num_arguments, device=device)

        if report_percentage:
            a_mask = (a_justifiable != a_baseline)
            rew_target, rew_baseline = rew_target[a_mask], rew_baseline[a_mask]
            reward = (rew_target > rew_baseline).float()
        else:
            reward = torch.zeros_like(rew_target, device=device)
            reward = torch.where(rew_target > rew_baseline, torch.ones_like(reward, device=device), reward)
            reward = torch.where(rew_target < rew_baseline, -torch.ones_like(reward, device=device), reward)

        wins = torch.cat((wins, reward))

    return wins.mean().item() if report_percentage else wins.flatten().tolist()


def full_vs_partial_context(policy: Union["DQNPolicy", List["DQNPolicy"]], dataloader: DataLoader, device: str = "cuda") -> float:
    result = torch.empty((0, ), dtype=torch.float32, device=device)

    if isinstance(policy, list):
        return np.mean([full_vs_partial_context(t, dataloader, device) for t in policy])

    for s_t, a_p1, a_p2, pref in tqdm.tqdm(dataloader):
        a_t = torch.cat((a_p1[pref[:, 0] == 1], a_p2[pref[:, 1] == 1])).flatten()
        a_r = torch.cat((a_p1[pref[:, 0] == 0], a_p2[pref[:, 1] == 0])).flatten()
        s_t, a_t, a_r = s_t.to(device), a_t.to(device), a_r.to(device)
        batch = Batch(obs=s_t, info={})

        batch_logits = policy(batch).logits
        a_t_logits, a_r_logits = batch_logits.gather(dim=1, index=a_t[:, None]).flatten(), batch_logits.gather(dim=1, index=a_r[:, None]).flatten()
        aligned = (a_t_logits >= a_r_logits).float()

        result = torch.cat((result, aligned.float()))

    return result.mean().item()


def run_debate(s_t: torch.Tensor, a_p1: torch.Tensor, a_p2: Optional[torch.Tensor], judge: "Judge", argumentator: "MaskedPPO", num_arguments: int = 6, return_args: bool = False, device: str = "cuda") -> Tuple[torch.Tensor]:
    args = torch.zeros(s_t.shape, dtype=torch.float32, device=device)
    args_mask = torch.zeros(s_t.shape, dtype=torch.int8, device=device)
    args_count = 0
    state_dim = s_t.shape[-1]
    player_state = {
        "player_0": {"action": a_p1},
        "player_1": {"action": a_p2 if a_p2 is not None else a_p1}
    }
    player = random.choice([0, 1])

    if judge.num_arguments == s_t.shape[0]:
        # skips debate if judge takes as input the entire state
        args_count = judge.num_arguments
        args = s_t
        args_mask = torch.ones_like(args_mask, device=device, dtype=torch.int8)

    while args_count < num_arguments:
        state = player_state[f"player_{player}"]

        obs = Batch(obs={
            "args": args.cpu().numpy(),
            "args_mask": args_mask.cpu().numpy(),
            "obs": s_t.cpu().numpy(),
            "act": state["action"].cpu().numpy(),
            "mask": (1 - args_mask).bool().cpu().numpy(),
        }, info={})
        act = argumentator(obs).act

        mask = (F.one_hot(act, num_classes=state_dim) == 1)
        args = torch.where(mask, s_t, args)
        args_mask = torch.where(mask, torch.ones_like(args_mask, device=device, dtype=torch.int8), args_mask)
        args_count += 1
        player = 1 - player

    if a_p2 is not None:
        if return_args:
            return judge((args, args_mask, a_p1)), judge((args, args_mask, a_p2)), args, args_mask
        else:
            return judge((args, args_mask, a_p1)), judge((args, args_mask, a_p2))
    else:
        if return_args:
            return judge((args, args_mask, a_p1)), args, args_mask
        else:
            return judge((args, args_mask, a_p1))


def get_action_counts(dataloader: DataLoader, policy: Optional[Union[torch.nn.Module, List[torch.nn.Module]]], device: str = "cpu"):
    counts = torch.zeros((25, ), dtype=torch.int32, device=device)

    for s_t, a_t, _, _, _ in dataloader:
        s_t, a_t = s_t.to(device), a_t.to(device)

        if policy is None:
            a_t_pred = a_t.argmax(dim=-1)
            counts.scatter_add_(0, a_t_pred, torch.ones_like(a_t_pred, dtype=torch.int32, device=device))
        elif isinstance(policy, BasePolicy):
            a_t_pred = policy(Batch(obs=s_t, info={})).act
            a_t_pred = torch.tensor(a_t_pred).to(device)
            counts.scatter_add_(0, a_t_pred, torch.ones_like(a_t_pred, dtype=torch.int32, device=device))
        elif isinstance(policy, torch.nn.Module):
            a_t_pred = policy(s_t).argmax(dim=-1)
            counts.scatter_add_(0, a_t_pred, torch.ones_like(a_t_pred, dtype=torch.int32, device=device))
        else:
            for policy_seed in policy:
                a_t_pred = torch.tensor(policy_seed(Batch(obs=s_t, info={})).act).to(device) if isinstance(policy_seed, BasePolicy) else policy_seed(s_t).argmax(dim=-1)
                counts.scatter_add_(0, a_t_pred, torch.ones_like(a_t_pred, dtype=torch.int32, device=device))

    if isinstance(policy, List):
        counts = torch.div(counts, len(policy), rounding_mode="floor")

    return counts.cpu().numpy()
