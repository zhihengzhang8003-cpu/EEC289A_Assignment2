"""Student world model — Run 7 reproduction (Attempt 2 residual MLP + Attempt 3 KNOBS).

Architecture (starter Attempt 2):
* Drop cart_pos from MLP input (translation invariance)
* MLP predicts 2 velocity deltas [d_cart_vel, d_pole_vel], tanh-bounded ±5
* Position deltas via Linear(2,1) on [vel, d_vel] — exact kinematic integration
* hidden_dim=256, num_layers=3 ResidualBlocks (LayerNorm + SiLU)

KNOBS = Run 7 (Attempt 3):
  H=20, W=1.0, DISC=0.95, OSW=25, UPDATES=24000, LR=7e-4
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(self.act(self.fc1(self.norm(x))))
        return x + h


class StudentWorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int = 4,
        act_dim: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 3,
        use_gru: bool = False,
        delta_limit: float = 5.0,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.use_gru = bool(use_gru)
        self.delta_limit = float(delta_limit)

        hidden = int(hidden_dim)
        in_dim = (self.obs_dim - 1) + self.act_dim
        self.input_proj = nn.Linear(in_dim, hidden)
        self.input_act = nn.SiLU()
        self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(max(1, int(num_layers)))])
        self.gru = nn.GRUCell(hidden, hidden) if self.use_gru else None
        self.vel_head = nn.Linear(hidden, 2)
        self.pos_cart = nn.Linear(2, 1)
        self.pos_pole = nn.Linear(2, 1)

    def initial_hidden(self, batch_size: int, device: torch.device):
        if self.gru is None:
            return None
        return torch.zeros(batch_size, self.gru.hidden_size, device=device)

    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        cart_vel = obs_norm[:, 2:3]
        pole_vel = obs_norm[:, 3:4]

        feat = self.input_act(self.input_proj(torch.cat([obs_norm[:, 1:], act_norm], dim=-1)))
        feat = self.blocks(feat)
        if self.gru is not None:
            if hidden is None:
                hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)
            hidden = self.gru(feat, hidden)
            feat = hidden

        raw_vel = self.vel_head(feat)
        d_vel = self.delta_limit * torch.tanh(raw_vel / self.delta_limit)
        d_cart_vel = d_vel[:, 0:1]
        d_pole_vel = d_vel[:, 1:2]

        d_cart_pos = self.pos_cart(torch.cat([cart_vel, d_cart_vel], dim=-1))
        d_pole_ang = self.pos_pole(torch.cat([pole_vel, d_pole_vel], dim=-1))

        delta = torch.cat([d_cart_pos, d_pole_ang, d_cart_vel, d_pole_vel], dim=-1)
        return delta, hidden
