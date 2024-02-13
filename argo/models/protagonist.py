import torch
import numpy as np
import torch.nn as nn

from typing import Any, Dict
from tianshou.policy import DQNPolicy
from tianshou.data import Batch, to_torch_as, ReplayBuffer
from tianshou.utils.net.common import Net


def make_ddqn_protagonist(
    state_dim: int = 44,
    action_dim: int = 25,
    lr: float = 1e-4,
    hidden_dim: int = 128,
    hidden_depth: int = 2,
    tau: float = 1e-3,
    n_estimation_step: int = 3,
    relu_slope: float = 0.01,
    discount: float = 0.99,
    device: str = "cuda",
) -> DQNPolicy:
    net = Net(
        state_shape=state_dim, action_shape=action_dim,
        hidden_sizes=[hidden_dim] * hidden_depth,
        norm_layer=nn.BatchNorm1d, device=device, softmax=False,
        activation=nn.LeakyReLU, act_args={"negative_slope": relu_slope},
        dueling_param=({"hidden_sizes": [hidden_dim // 2]}, {"hidden_sizes": [hidden_dim // 2]}),
    )
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.constant_(m.bias, 0.0)
    optim = torch.optim.Adam(net.parameters(), lr=lr, eps=15e-5)
    policy = DDQNPolicy(
        model=net, optim=optim,
        discount_factor=discount, estimation_step=n_estimation_step,
        is_double=True, clip_loss_grad=False, tau=tau,
    )
    return policy.to(device)


class DDQNPolicy(DQNPolicy):
    def __init__(
        self,
        model: torch.nn.Module, optim: torch.optim.Optimizer,
        discount_factor: float = 0.99, estimation_step: int = 1,
        reward_normalization: bool = False, is_double: bool = True,
        tau: float = 1e-3, clip_loss_grad: bool = False,
        **kwargs: Any,
    ) -> None:
        target_update_freq = 100 # needs to be > 0 to use the target network
        self.tau = tau
        super().__init__(model, optim, discount_factor, estimation_step, target_update_freq, reward_normalization, is_double, clip_loss_grad, **kwargs)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        # we perform Polyak averaging on every gradient step
        self.sync_weight()

        # calculates td-error
        self.optim.zero_grad()
        weight = batch.pop("weight", 1.0)
        q = self(batch).logits
        q = q[np.arange(len(q)), batch.act]
        returns = to_torch_as(batch.returns.flatten(), q)
        td_error = returns - q

        # c.f. A. Raghu et al., “Deep Reinforcement Learning for Sepsis Treatment.”
        q_lmbd_reg = 5.0 # regularizer parameter for clipping q-values
        q_thresh = 20.0  # threshold for clipping q-values

        if self._clip_loss_grad:
            # Huber loss with clipping-based gradient and q-values
            y = q.reshape(-1, 1)
            t = returns.reshape(-1, 1)
            reg_term = torch.maximum(torch.abs(q) - q_thresh, torch.zeros_like(q, device=q.device))
            loss = torch.nn.functional.huber_loss(y, t, reduction="mean") + q_lmbd_reg * reg_term.sum()
        else:
            # Standard MSE loss, with q-values clipping regularization
            reg_term = torch.maximum(torch.abs(q) - q_thresh, torch.zeros_like(q, device=q.device))
            err_term = td_error.pow(2) * weight
            loss = err_term.mean() + q_lmbd_reg * reg_term.sum()

        # sets weight for prioritized experience replay
        batch.weight = td_error

        # performs gradient step
        loss.backward()
        self.optim.step()
        self._iter += 1
        return {"loss": loss.item()}

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        # c.f. A. Raghu et al., “Deep Reinforcement Learning for Sepsis Treatment.”
        # clip target q-values to given threshold
        q_thresh = 20.0

        batch = buffer[indices]
        result = self(batch, input="obs_next")

        if self._target:
            target_q = self(batch, model="model_old", input="obs_next").logits
        else:
            target_q = result.logits

        if self._is_double:
            target_q = target_q[np.arange(len(result.act)), result.act]
            target_q[target_q > q_thresh] = q_thresh
            target_q[target_q < -q_thresh] = -q_thresh
            return target_q
        else:  # Nature DQN, over estimates
            return target_q.max(dim=1)[0]

    def sync_weight(self) -> None:
        # performs Polyak averaging
        for param, target_param in zip(self.model.parameters(), self.model_old.parameters()):
           target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
