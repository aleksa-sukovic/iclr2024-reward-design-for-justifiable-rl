import torch
import numpy as np
import torchmetrics as tm
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from typing import List, Literal, Optional


class Judge(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_actions: int,
        action_dim: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 0,
        num_arguments: Optional[int] = 6,
        optimizer: Literal["adam", "sgd", "rmsprop"] = "adam",
        device = torch.device("cpu"),
    ):
        super().__init__()
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.num_arguments = num_arguments if num_arguments is not None else state_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_actions = num_actions

        assert self.num_arguments <= self.state_dim, "Number of arguments cannot be greater than number of state features."

        self.l1 = nn.Linear(2 * state_dim + num_actions, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.prelu1 = nn.PReLU(hidden_dim)

        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.prelu2 = nn.PReLU(hidden_dim)

        self.lin_proj = nn.Linear(hidden_dim, 1)

        self.device = device
        self.loss = nn.CrossEntropyLoss()
        self.train_metrics = tm.MetricCollection({
            "accuracy": tm.Accuracy(task="binary"),
            "precision": tm.Precision(task="binary"),
            "recall": tm.Recall(task="binary"),
            "f1_score": tm.F1Score(task="binary"),
        })
        self.val_metrics = tm.MetricCollection({
            "accuracy": tm.Accuracy(task="binary"),
            "precision": tm.Precision(task="binary"),
            "recall": tm.Recall(task="binary"),
            "f1_score": tm.F1Score(task="binary"),
        })

        parameters = [
            {"params": self.l1.parameters()},
            {"params": self.bn1.parameters()},
            {"params": self.l2.parameters()},
            {"params": self.bn2.parameters()},
            {"params": self.lin_proj.parameters()},
            {"params": self.prelu1.parameters(), "weight_decay": 0.0},
            {"params": self.prelu2.parameters(), "weight_decay": 0.0},
        ]

        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif optimizer == "rmsprop":
            self.optimizer = torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
        else:
            raise RuntimeError(f"Unknown optimizer: '{self.optimizer}'")

    def forward(self, state):
        s_t_args, s_t_args_mask, a_t = state
        a_t = F.one_hot(a_t, num_classes=self.num_actions)
        s_t = torch.cat((s_t_args, s_t_args_mask, a_t), dim=1)

        # hidden layer 1
        p = self.prelu1(self.l1(s_t))
        p = self.bn1(p)

        # hidden layer 2
        p = self.prelu2(self.l2(p))
        p = self.bn2(p)

        return self.lin_proj(p)

    def epoch_train(self, dataloader: DataLoader):
        losses: List[float] = []

        self.train()
        self.train_metrics.reset()

        for s_t, a1, a2, pref in dataloader:
            s_t, a1, a2, pref = s_t.to(self.device), a1.to(self.device), a2.to(self.device), pref.to(self.device)

            if self.num_arguments < self.state_dim:
                # Randomly sample requested number of arguments from current state
                num_items = s_t.size(0)
                index = torch.multinomial(torch.ones((num_items, self.state_dim)), num_samples=self.num_arguments, replacement=False)
                src = torch.ones((num_items, self.num_arguments))
                arg_mask = torch.zeros((num_items, self.state_dim)).scatter(dim=1, index=index, src=src).to(self.device)
                s_t = s_t * arg_mask
            else:
                # Use all features from a current state as arguments
                num_items = s_t.size(0)
                arg_mask = torch.ones_like(s_t)

            self.optimizer.zero_grad()
            r1, r2 = self.forward((s_t, arg_mask, a1)), self.forward((s_t, arg_mask, a2))
            rew = torch.cat((r1, r2), dim=1)
            loss = self.loss(rew, pref)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.detach().item())
            self.train_metrics.update(rew.detach().argmax(dim=1), pref.argmax(dim=1))

        return {"loss": np.mean(losses), **self.train_metrics.compute()}

    def epoch_eval(self, dataloader: DataLoader):
        losses: List[float] = []

        self.eval()
        self.val_metrics.reset()

        for s_t, a1, a2, pref in dataloader:
            s_t, a1, a2, pref = s_t.to(self.device), a1.to(self.device), a2.to(self.device), pref.to(self.device)

            if self.num_arguments < self.state_dim:
                # Randomly sample requested number of arguments from current state
                num_items = s_t.size(0)
                index = torch.multinomial(torch.ones((num_items, self.state_dim)), num_samples=self.num_arguments, replacement=False)
                src = torch.ones((num_items, self.num_arguments))
                arg_mask = torch.zeros((num_items, self.state_dim)).scatter(dim=1, index=index, src=src).to(self.device)
                s_t = s_t * arg_mask
            else:
                num_items = s_t.size(0)
                arg_mask = torch.ones_like(s_t)

            r1, r2 = self.forward((s_t, arg_mask, a1)), self.forward((s_t, arg_mask, a2))
            rew = torch.cat((r1, r2), dim=1)
            loss = self.loss(rew, pref)

            loss = self.loss(rew, pref)
            losses.append(loss.detach().cpu().item())
            self.val_metrics.update(rew.argmax(dim=1), pref.argmax(dim=1))

        return {"loss": np.mean(losses), **self.train_metrics.compute()}
