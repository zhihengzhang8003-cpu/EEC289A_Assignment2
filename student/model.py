"""Student world model for InvertedPendulum-v5 dynamics.

Attempt 2 -- physics-structured residual model.

Design rationale (compared with Attempt 1's free 4-D delta MLP):

* InvertedPendulum's dynamics are *translation-invariant in cart position* (the
  cart slides on a rail with no position-dependent force inside the data
  distribution). So the cart-pos coordinate is dropped from the MLP input,
  removing a feedback channel that could only ever destabilise long rollouts.

* The MLP head predicts only the two **velocity** deltas
  ``[d_cart_vel_norm, d_pole_vel_norm]`` -- the genuine degrees of freedom of
  the system.

* The two **position** deltas are *exact kinematic integrals* of velocity
  (``dx = dt * v_{t+1}``). Because the normalizer is affine, this exact
  relation becomes an affine function of ``[v_norm, dv_norm]`` after
  normalization; a tiny learnable ``Linear(2, 1)`` absorbs the integrator step
  ``dt`` and all normalization constants. Once velocities are correct,
  positions are exact too -- they cannot independently compound error.

* PyTorch's default ``nn.Linear`` initialization is used (intentionally NOT a
  small init, learning from the Run-1 "predict-smooth shortcut" failure).

* Architecture is fully determined by ``hidden_dim``, ``num_layers``,
  ``use_gru`` (the only fields ``build_model`` passes), so checkpoints
  reconstruct correctly.

* Public interface ``forward(obs_norm, act_norm, hidden) -> (delta_norm, hidden)``
  and ``initial_hidden`` is unchanged.
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
        # MLP input = [pole_angle, cart_vel, pole_vel, action] -- cart_pos dropped.
        in_dim = (self.obs_dim - 1) + self.act_dim
        self.input_proj = nn.Linear(in_dim, hidden)
        self.input_act = nn.SiLU()
        self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(max(1, int(num_layers)))])
        self.gru = nn.GRUCell(hidden, hidden) if self.use_gru else None
        # Predicts the two normalized velocity deltas: [d_cart_vel, d_pole_vel].
        self.vel_head = nn.Linear(hidden, 2)
        # Exact kinematic read-out for positions: each position delta is an
        # affine function of the current velocity and the predicted velocity
        # delta of the same body. The Linear absorbs the integrator step and
        # normalization constants.
        self.pos_cart = nn.Linear(2, 1)   # d_cart_pos   <- [cart_vel, d_cart_vel]
        self.pos_pole = nn.Linear(2, 1)   # d_pole_angle <- [pole_vel, d_pole_vel]
        # Run 10: softened structure -- learnable small residual on position.
        # Zero-initialised so behaviour starts identical to pure kinematic integration;
        # the model can grow it during training to correct small MuJoCo non-linearities.
        self.pos_residual = nn.Linear(hidden, 2)
        nn.init.zeros_(self.pos_residual.weight)
        nn.init.zeros_(self.pos_residual.bias)

    def initial_hidden(self, batch_size: int, device: torch.device):
        if self.gru is None:
            return None
        return torch.zeros(batch_size, self.gru.hidden_size, device=device)

    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        # obs_norm = [cart_pos, pole_angle, cart_vel, pole_vel] (normalized).
        cart_vel = obs_norm[:, 2:3]
        pole_vel = obs_norm[:, 3:4]

        # Drop cart_pos; MLP sees [pole_angle, cart_vel, pole_vel, action].
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

        d_cart_pos_phys = self.pos_cart(torch.cat([cart_vel, d_cart_vel], dim=-1))
        d_pole_ang_phys = self.pos_pole(torch.cat([pole_vel, d_pole_vel], dim=-1))
        # Run 10: add small learnable residual on top of kinematic integration.
        # Zero-init at start, so initial behaviour == pure physics integration.
        residual = self.pos_residual(feat)
        d_cart_pos = d_cart_pos_phys + residual[:, 0:1]
        d_pole_ang = d_pole_ang_phys + residual[:, 1:2]

        delta = torch.cat([d_cart_pos, d_pole_ang, d_cart_vel, d_pole_vel], dim=-1)
        return delta, hidden
