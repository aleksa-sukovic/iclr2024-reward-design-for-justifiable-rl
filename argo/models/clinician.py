import torch
import numpy as np
import torchmetrics as tm
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from typing import List, Literal


class FC(nn.Module):
    def __init__(self, state_dim=33, action_dim=25, hidden_dim: int = 64):
        super(FC, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        out = F.relu(self.l1(state))
        out = self.bn1(out)
        out = F.relu(self.l2(out))
        out = self.bn2(out)
        return self.l3(out)


class ClinicianPolicy(nn.Module):
    def __init__(self, state_dim: int = 44, action_dim: int = 25, hidden_dim: int = 256, lr: float = 1e-3, weight_decay: float = 0.1, optimizer: Literal["adam", "sgd"] = "adam", device: str = "cpu"):
        super(ClinicianPolicy, self).__init__()

        self.device = device
        self.state_shape = state_dim
        self.action_dim = action_dim
        self.lr = lr

        self.model = FC(state_dim, action_dim, hidden_dim).to(self.device)
        self.loss_func = nn.CrossEntropyLoss()

        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=weight_decay)

    def forward(self, state):
        return self.model(state)

    def epoch_train(self, dataloader: DataLoader):
        losses: List[float] = []
        metrics: tm.MetricCollection = tm.MetricCollection({
            "accuracy": tm.Accuracy(task="multiclass", num_classes=self.action_dim),
            "precision": tm.Precision(task="multiclass", num_classes=self.action_dim),
            "recall": tm.Recall(task="multiclass", num_classes=self.action_dim),
            "f1_score": tm.F1Score(task="multiclass", num_classes=self.action_dim),
        }).to(self.device)

        for s_t, a_t in dataloader:
            s_t, a_t = s_t.to(self.device), a_t.to(self.device)

            a_t_pred = self.model(s_t)
            loss = self.loss_func(a_t_pred, a_t.flatten())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            metrics.update(a_t_pred.detach().argmax(dim=1), a_t)

        return {"loss": np.mean(losses), **metrics.compute()}

    def epoch_eval(self, dataloader: DataLoader):
        losses: List[float] = []
        metrics: tm.MetricCollection = tm.MetricCollection({
            "accuracy": tm.Accuracy(task="multiclass", num_classes=self.action_dim),
            "precision": tm.Precision(task="multiclass", num_classes=self.action_dim),
            "recall": tm.Recall(task="multiclass", num_classes=self.action_dim),
            "f1_score": tm.F1Score(task="multiclass", num_classes=self.action_dim),
        }).to(self.device)

        for s_t, a_t in dataloader:
            s_t, a_t = s_t.to(self.device), a_t.to(self.device)
            a_t_pred = self.model(s_t)
            loss = self.loss_func(a_t_pred, a_t.flatten())
            losses.append(loss.detach().item())
            metrics.update(a_t_pred.detach().argmax(dim=1), a_t)

        return {"loss": np.mean(losses), **metrics.compute()}
