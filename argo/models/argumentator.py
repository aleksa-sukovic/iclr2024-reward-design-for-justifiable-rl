import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from gymnasium import spaces
from typing import Literal, Any, Optional, Union

from tianshou.data import Batch
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor, Critic

from argo.library.training import dist_categorical


def make_argumentator(
    state_dim: int,
    num_actions: int,
    hidden_dim: int = 256,
    hidden_depth: int = 2,
    lr: float = 5e-4,
    lr_schedule: Literal["constant", "step"] = None,
    gamma: float = 0.9,
    gae_lambda: float = 0.92,
    max_grad_norm: float = 0.7,
    vf_coef: float = 0.5,
    ent_coef: float = 4e-4,
    normalize_rewards: bool = False,
    normalize_advantage: bool = True,
    clip_range: float = 0.2,
    value_clip: bool = True,
    dual_clip: float = 5.0,
    device: str = "cuda",
    masked: bool = True,
    ortho_init: bool = True,
    frozen: bool = False,
):
    net = ObsPreprocessor(state_dim=state_dim, num_actions=num_actions, device=device)

    actor = Actor(net, action_shape=state_dim, hidden_sizes=[hidden_dim] * hidden_depth, softmax_output=False, device=device)
    actor = actor.to(device)

    critic = Critic(net, hidden_sizes=[hidden_dim] * hidden_depth, device=device)
    critic = critic.to(device)

    actor_critic = ActorCritic(actor, critic)

    if frozen:
        optim = torch.optim.SGD(actor_critic.parameters(), lr=0) # c.f. https://github.com/thu-ml/tianshou/issues/381
    else:
        optim = torch.optim.Adam(actor_critic.parameters(), lr=lr, eps=1e-5)

    lr_scheduler = None
    if lr_schedule == "step":
        lr_scheduler = MultiStepLR(optim, milestones=[1500, 10000], gamma=0.1)

    if ortho_init:
        for m in actor_critic.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    policy_cls = MaskedPPO if masked else PPOPolicy
    policy = policy_cls(
        actor, critic, optim,
        dist_fn=dist_categorical,
        discount_factor=gamma,
        gae_lambda=gae_lambda,
        max_grad_norm=max_grad_norm,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        reward_normalization=normalize_rewards,
        action_scaling=False,
        lr_scheduler=lr_scheduler,
        action_space=spaces.Discrete(state_dim),
        eps_clip=clip_range,
        value_clip=value_clip,
        dual_clip=dual_clip,
        advantage_normalization=normalize_advantage,
    )

    if frozen:
        policy = policy.eval()

    return policy


class MaskedPPO(PPOPolicy):
    def forward(self, batch: Union[Batch, torch.Tensor], state: Optional[Union[dict, Batch, np.ndarray]] = None, **kwargs: Any) -> Batch:
        if isinstance(batch, torch.Tensor):
            logits, hidden = self.actor(batch, state=state, info={})
        else:
            logits, hidden = self.actor(batch.obs, state=state, info=batch.info)

        if isinstance(batch, Batch) and hasattr(batch.obs, "mask"):
            # N.B. this does not handle a case where logits are a tuple (e.g., (mean, std) for continuous actions), not relevant here.
            mask = batch.obs.mask if hasattr(batch.obs, "mask") else batch.obs.mask
            mask = mask.astype(int)
            mask = torch.tensor(mask, device=logits.device, dtype=torch.int8)
            inft = torch.ones_like(mask, device=logits.device) * torch.finfo().min
            logits = torch.where(mask == 0, inft, logits)
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)

        if self._deterministic_eval and not self.training:
            if self.action_type == "discrete":
                act = logits.argmax(-1)
            elif self.action_type == "continuous":
                act = logits[0]
        else:
            act = dist.sample()

        if isinstance(batch, Batch):
            return Batch(logits=logits, act=act, state=hidden, dist=dist)
        else:
            return logits


class ObsPreprocessor(nn.Module):
    def __init__(self, state_dim: int, num_actions: int = 25, device: str = "cuda"):
        super().__init__()
        self.num_actions = num_actions
        self.output_dim = 2 * state_dim + num_actions
        self.device = device
        self.to(self.device)

    def forward(self, observation: Union[Batch, torch.Tensor], state: Any = None):
        if isinstance(observation, torch.Tensor):
            return observation, None
        if isinstance(observation.obs, Batch):
            observation = observation.obs

        obs = observation.obs if torch.is_tensor(observation.obs) else torch.tensor(observation.obs, device=self.device)
        args_mask = torch.tensor(observation.args_mask, device=self.device).int()

        act = observation.act.flatten().to(self.device) if torch.is_tensor(observation.act) else torch.tensor(observation.act.flatten(), device=self.device)
        act = F.one_hot(act, num_classes=self.num_actions).int()

        return torch.cat((obs, args_mask, act), dim=1), None
