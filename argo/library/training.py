import torch
import numpy as np

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class EarlyStopping:
    patience: int = field(
        default=10,
        metadata={"help": "Number of epochs without improvement, before early stopping."})
    improvement: Literal["lower", "higher"] = field(
        default="higher",
        metadata={"help": "Indicates if higher or lower values are considered improvement."})
    delta: float = field(
        default=1e-5,
        metadata={"help": "Minimum change in monitored quantity which classifies as improvement."})
    value_best: float = field(
        default=0.0,
        init=False,
        metadata={"help": "Current best value of the monitored quantity."})
    value_plateau: int = field(
        default=0,
        init=False,
        metadata={"help": "Number of epochs without improvement."})
    stop: bool = field(
        default=False,
        init=False,
        metadata={"help": "Indicates if early stopping criteria has been met."})

    def reset(self):
        self.value_best = np.inf if self.improvement == "lower" else -np.inf
        self.value_plateau = 0
        self.stop = False

    def step(self, value: float, logger: Optional["Logger"] = None):
        threshold = self.value_best - self.delta if self.improvement == "lower" else self.value_best + self.delta
        is_improving = value < threshold if self.improvement == "lower" else value > threshold

        if is_improving:
            self.value_best = value
            self.value_plateau = 0
        else:
            self.value_plateau += 1
            if logger: logger.info(f"Early stopping, plateaued for {self.value_plateau}/{self.patience} epochs")

        if self.value_plateau == self.patience:
            self.stop = True
            if logger: logger.info(f"Early stopping, patience reached")


def weight_init(module: torch.nn.Module):
    classname = module.__class__.__name__
    if classname.find('Linear') != -1:
        weight_shape = list(module.weight.data.size())
        fan_in, fan_out = weight_shape[1], weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        module.weight.data.uniform_(-w_bound, w_bound)
        module.bias.data.fill_(0)
    elif classname.find('GRUCell') != -1:
        for param in module.parameters():
            if len(param.shape) >= 2:
                torch.init.orthogonal_(param.data)
            else:
                torch.init.normal_(param.data)


def dist_categorical(logits):
   return torch.distributions.Categorical(logits=logits)

