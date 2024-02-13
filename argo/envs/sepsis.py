import random
import torch
import numpy as np

from typing import Optional, Literal
from gymnasium import spaces, Env
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tianshou.data import Batch

from argo.models.judge import Judge
from argo.models.argumentator import MaskedPPO


class SepsisArgumentationEnv(Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        dataloader: DataLoader,
        judge: Judge,
        opponent: Optional[MaskedPPO] = None,
        scaler: Optional[StandardScaler] = None,
        action_dim: int = 25,
        state_dim: int = 44,
        num_arguments: int = 6,
        render_mode: str = None,
        start_player: Literal[1, 2, None] = None,
        inverse_reward: bool = False,
        xai: bool = False,
        propose_evidence_upfront: bool = False,
        use_judge_diff_as_reward: bool = False,
        device: str = None,
        eval: bool = False,
    ) -> None:
        assert num_arguments % 2 == 0 if opponent else True, "Number of arguments must be even when opponent is provided."
        assert start_player == 1 if xai else True, "Start player must be 1 when XAI is enabled."

        self.judge = judge
        self.scaler = scaler
        self.inverse_reward = inverse_reward
        self.dataloader = dataloader
        self.dataloader_iterator = None
        self.opponent = opponent
        self.start_player = start_player
        self.render_mode = render_mode
        self.num_arguments = num_arguments
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.xai = xai
        self.eval = eval
        self.propose_evidence_upfront = propose_evidence_upfront
        self.use_judge_diff_as_reward = use_judge_diff_as_reward
        self.device = device
        self.action_space = spaces.Discrete(self.state_dim)
        self.observation_space = spaces.Dict({
            "args": spaces.Box(low=np.finfo(np.float32).min, high=np.finfo(np.float32).max, shape=(self.state_dim, ), dtype=np.float32),
            "args_mask": spaces.Box(low=0, high=1, shape=(self.state_dim, ), dtype=np.int8),
            "obs": spaces.Box(low=np.finfo(np.float32).min, high=np.finfo(np.float32).max, shape=(self.state_dim, ), dtype=np.float32),
            "act": spaces.Discrete(self.action_dim),
            "mask": spaces.Box(low=0, high=1, shape=(self.state_dim, ), dtype=np.int8),
        })
        self.args = torch.zeros((self.state_dim, ), dtype=torch.int8, device=self.device)
        self.obs = torch.zeros((self.state_dim, ), device=self.device)

    def reset(self, seed=None, options=None, item=None):
        super().reset(seed=seed)

        data = self._get_next_item() if item is None else item
        s_t, a_p1, a_p2, pref = data[:4]
        xai_args = data[-1] if self.xai else None

        a_t = a_p1 if pref.flatten()[0].item() == 1.0 else a_p2
        a_r = a_p1 if pref.flatten()[0].item() == 0.0 else a_p2

        self.args = torch.zeros(self.state_dim, dtype=torch.float32, device=self.device)
        self.args_mask = torch.zeros(self.state_dim, dtype=torch.int8, device=self.device)
        self.obs = s_t.flatten()
        self.act, self.act_opp = (a_r, a_t) if self.inverse_reward else (a_t, a_r)

        if xai_args is not None:
            xai_args = torch.nonzero(xai_args, as_tuple=True)[1].to(self.device)
            self.args = self.args.scatter(dim=0, src=s_t.flatten(), index=xai_args).to(self.device)
            self.args_mask = self.args_mask.scatter(dim=0, src=torch.ones_like(self.args_mask), index=xai_args).to(self.device)

        if not self.propose_evidence_upfront and self.opponent and (self.start_player or random.choice([1, 2])) == 2:
            # if we have an opponent and it is its turn, play out its response
            with torch.no_grad():
                obs = self._get_obs(unsqueeze=True, opponent=True)
                act = self.opponent(Batch(obs=obs, info={})).act.item()
            self.args[act] = self.obs[act]
            self.args_mask[act] = 1
        elif self.propose_evidence_upfront and self.opponent:
            # if we are not in turn-based mode, propose half of the evidence
            with torch.no_grad():
                for _ in range(self.num_arguments // 2):
                    obs = self._get_obs(unsqueeze=True, opponent=True)
                    act = self.opponent(Batch(obs=obs, info={})).act.item()
                    self.args[act] = self.obs[act]
                    self.args_mask[act] = 1

        return self._get_obs(), {}

    def step(self, action):
        num_args_before = torch.count_nonzero(self.args_mask).item()
        self.args[action] = self.obs[action]
        self.args_mask[action] = 1
        num_args_after = torch.count_nonzero(self.args_mask).item()

        if not self.propose_evidence_upfront and self.opponent and num_args_after < self.num_arguments:
            # if opponent was provided, play out its response
            with torch.no_grad():
                obs = self._get_obs(unsqueeze=True, opponent=True)
                act = self.opponent(Batch(obs=obs, info={})).act.item()
            self.args[act] = self.obs[act]
            self.args_mask[act] = 1
            num_args_after = torch.count_nonzero(self.args_mask).item()

        if num_args_after == self.num_arguments:
            # the necessary number of arguments is proposed
            obs = self._get_obs()

            rew_p1 = self.judge((self.args.unsqueeze(dim=0), self.args_mask.unsqueeze(dim=0), self.act)).item()
            rew_p2 = self.judge((self.args.unsqueeze(dim=0), self.args_mask.unsqueeze(dim=0), self.act_opp)).item()

            if self.use_judge_diff_as_reward:
                assert rew_p1 * rew_p2 > 0, "Rewards must have the same sign. We do not ensure this when training a judge model, this was added as an additional experiment during rebuttal."
                reward = rew_p1 - rew_p2
            elif self.eval:
                reward = 1.0 if rew_p1 > rew_p2 else 0.0 # just counts the wins when evaluating
            else:
                reward = 1.0 if rew_p1 > rew_p2 else -1.0 if rew_p2 > rew_p1 else 0.0 # zero-sum rewards when training

            terminated = True
            truncated = False
            info = {"num_args_before": num_args_before, "num_args_after": num_args_after, "rew": reward, "rew_p1": rew_p1, "rew_p2": rew_p2}
        else:
            # agent proposed an argument, but necessary number of arguments wasn't reached
            obs = self._get_obs()
            reward = 0.0
            terminated = False
            truncated = False
            info = {"num_args_before": num_args_before, "num_args_after": num_args_after, "reward": reward}

        return obs, reward, terminated, truncated, info

    def _get_obs(self, unsqueeze: bool = False, opponent: bool = False):
        return {
            "args": self.args.clone().cpu().numpy() if not unsqueeze else self.args.clone().unsqueeze(dim=0).numpy(),
            "args_mask": self.args_mask.clone().cpu().numpy() if not unsqueeze else self.args_mask.clone().unsqueeze(dim=0).numpy(),
            "obs": self.obs.clone().cpu().numpy() if not unsqueeze else self.obs.clone().unsqueeze(dim=0).numpy(),
            "act": self.act.clone().cpu().numpy() if not opponent else self.act_opp.clone().numpy(),
            "mask": (1 - self.args_mask).clone().cpu().flatten().bool().numpy(),
        }

    def _get_next_item(self):
        try:
            if self.dataloader_iterator is None:
                self.dataloader_iterator = iter(self.dataloader)
            return next(self.dataloader_iterator)
        except StopIteration:
            self.dataloader_iterator = iter(self.dataloader)
            return next(self.dataloader_iterator)

