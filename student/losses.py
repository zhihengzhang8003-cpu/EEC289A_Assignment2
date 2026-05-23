"""Student one-step + curriculum rollout loss for the InvertedPendulum world model.

Improvements over the starter:

* The open-loop rollout horizon follows a *curriculum*: it ramps from a short
  horizon up to the full training horizon over the first part of training. The
  model first nails one-step dynamics, then learns to stay stable over long
  open-loop rollouts -- which is what VPT / nMSE actually reward.
* ``compute_loss`` receives no step index, so a tiny process-local counter
  drives the curriculum. A fresh training process restarts it at zero.
* The rollout loss optionally supports a per-step discount; with the default
  ``rollout_discount = 1.0`` it is a plain mean and behaves like the starter.

All config keys are read with ``.get`` defaults, so older configs (and the unit
tests) keep working unchanged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .rollout import open_loop_rollout


# Process-local training-step counter for the rollout-horizon curriculum.
_TRAIN_STEP = [0]


def one_step_delta_loss(model, states: torch.Tensor, actions: torch.Tensor, normalizer) -> torch.Tensor:
    obs = states[:, :-1].reshape(-1, states.shape[-1])
    act = actions.reshape(-1, actions.shape[-1])
    target_delta = (states[:, 1:] - states[:, :-1]).reshape(-1, states.shape[-1])
    obs_norm = normalizer.normalize_obs(obs)
    act_norm = normalizer.normalize_act(act)
    target_norm = normalizer.normalize_delta(target_delta)
    pred_norm, _ = model(obs_norm, act_norm, None)
    return F.mse_loss(pred_norm, target_norm)


def rollout_loss(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    warmup_steps: int,
    horizon: int,
    discount: float = 1.0,
) -> torch.Tensor:
    # Train open-loop stability at random positions inside each stored window.
    needed_states = int(warmup_steps) + int(horizon) + 1
    if states.shape[1] < needed_states:
        raise ValueError(
            "training.train_sequence_length is too short for rollout loss: "
            f"need at least {needed_states - 1} actions for warmup={warmup_steps}, horizon={horizon}."
        )
    max_start = states.shape[1] - needed_states
    if max_start > 0:
        start = int(torch.randint(0, max_start + 1, (), device=states.device).item())
    else:
        start = 0
    sub_states = states[:, start : start + needed_states]
    sub_actions = actions[:, start : start + int(warmup_steps) + int(horizon)]
    preds = open_loop_rollout(model, sub_states, sub_actions, normalizer, warmup_steps=warmup_steps, horizon=horizon)
    targets = sub_states[:, warmup_steps + 1 : warmup_steps + 1 + horizon]
    pred_norm = normalizer.normalize_obs(preds)
    target_norm = normalizer.normalize_obs(targets)

    per_step = ((pred_norm - target_norm) ** 2).mean(dim=(0, 2))  # [horizon]
    discount = float(discount)
    if abs(discount - 1.0) < 1e-9:
        return per_step.mean()
    weights = discount ** torch.arange(per_step.shape[0], device=per_step.device, dtype=per_step.dtype)
    return (per_step * weights).sum() / weights.sum()


def _curriculum_horizon(loss_cfg: dict, step: int) -> int:
    """Linearly ramp the rollout horizon from min to max over the curriculum."""
    max_h = int(loss_cfg.get("rollout_train_horizon", 5))
    min_h = int(loss_cfg.get("rollout_min_horizon", max_h))
    ramp = int(loss_cfg.get("rollout_curriculum_updates", 0))
    min_h = max(1, min(min_h, max_h))
    if ramp <= 0:
        return max_h
    frac = min(1.0, float(step) / float(ramp))
    return int(round(min_h + frac * (max_h - min_h)))


def compute_loss(model, batch: dict[str, torch.Tensor], normalizer, cfg: dict):
    loss_cfg = cfg["loss"]
    states = batch["states"]
    actions = batch["actions"]

    _TRAIN_STEP[0] += 1
    horizon = _curriculum_horizon(loss_cfg, _TRAIN_STEP[0])
    warmup = int(cfg["eval"].get("warmup_steps", 5))

    one = one_step_delta_loss(model, states, actions, normalizer)
    roll = rollout_loss(
        model,
        states,
        actions,
        normalizer,
        warmup_steps=warmup,
        horizon=horizon,
        discount=loss_cfg.get("rollout_discount", 1.0),
    )
    total = (
        float(loss_cfg.get("one_step_weight", 1.0)) * one
        + float(loss_cfg.get("rollout_weight", 0.3)) * roll
    )
    return total, {
        "loss/total": float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/rollout": float(roll.detach().cpu()),
        "rollout_horizon": float(horizon),
    }
