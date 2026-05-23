"""Student world model for InvertedPendulum-v5 dynamics.

Attempt 1 Run 2 -- residual MLP with default init.

Keeps the starter's residual delta-prediction interface (the network predicts a
normalized 4-D state delta, and the locked ``predict_next`` adds it to the
current observation), but strengthens the model so it can be trained for long,
stable open-loop rollouts:

* Deeper / wider trunk built from pre-norm residual blocks (LayerNorm + SiLU),
  which keeps gradients well-behaved at depth.
* An optional GRU cell (``use_gru``) gives a recurrent variant.
* ``tanh`` delta clamp keeps per-step deltas bounded.

Run 1 used a small (std=1e-2) output head initialization to start near
"identity dynamics", but combined with a long-rollout loss that pushed the
model into a "predict-smooth" shortcut (low long-horizon nMSE, high per-step
error). Run 2 reverts to PyTorch's default Linear init so the model outputs
deltas of natural scale from the first update.

The public interface (``forward(obs_norm, act_norm, hidden) -> (delta, hidden)``
and ``initial_hidden``) is unchanged, so the locked rollout / evaluation /
checkpoint code keeps working. The architecture is fully determined by
``hidden_dim``, ``num_layers`` and ``use_gru`` (the only fields ``build_model``
passes), so checkpoints reconstruct correctly.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Pre-norm residual MLP block: x + W2(SiLU(W1(LayerNorm(x))))."""

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
        self.input_proj = nn.Linear(self.obs_dim + self.act_dim, hidden)
        self.input_act = nn.SiLU()
        self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(max(1, int(num_layers)))])
        self.gru = nn.GRUCell(hidden, hidden) if self.use_gru else None
        self.head = nn.Linear(hidden, self.obs_dim)
        # Default PyTorch Linear init for self.head -- intentionally NOT zeroed,
        # so the network outputs deltas of natural scale from update 1 and
        # avoids the Run-1 "predict-smooth shortcut" local minimum.

    def initial_hidden(self, batch_size: int, device: torch.device):
        if self.gru is None:
            return None
        return torch.zeros(batch_size, self.gru.hidden_size, device=device)

    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        feat = self.input_act(self.input_proj(torch.cat([obs_norm, act_norm], dim=-1)))
        feat = self.blocks(feat)
        if self.gru is not None:
            if hidden is None:
                hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)
            hidden = self.gru(feat, hidden)
            feat = hidden
        raw_delta = self.head(feat)
        delta = self.delta_limit * torch.tanh(raw_delta / self.delta_limit)
        return delta, hidden
