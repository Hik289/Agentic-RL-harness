"""Offline Advantage-Weighted policy training (readme §16).

Workflow:
  1. Collect a buffer of (task_id, trajectory, return_G) tuples using a
     behavioral policy (e.g. perturbed Base Harness).
  2. For each task, compute baseline b(task) = mean return over trajectories
     of that task; per-traj advantage A_i = G_i - b(task_i).
  3. Per-traj weight w_i = clip(exp(A_i / temperature), w_min, w_max).
  4. Convert each step into (state, action_idx, weight) — all steps within
     a trajectory inherit w_i.
  5. Train a 1-layer MLP policy with weighted cross-entropy + entropy reg.

Default hyperparameters from readme §16.6.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field, asdict
from typing import Any

import torch
import torch.nn.functional as F

from .policy import MLPPolicy
# state_features module is domain-specific; caller provides a featurize
# function via build_dataset(traj_featurizer=...). Backward compat: default
# to the coding featurizer.
from .state_features import trajectory_to_features as _coding_featurize


@dataclass
class AWConfig:
    batch_size: int = 256
    learning_rate: float = 1e-3
    epochs: int = 20
    temperature: float = 0.2
    weight_clip_min: float = 0.1
    weight_clip_max: float = 10.0
    entropy_coef: float = 0.01
    hidden: int = 64
    weight_decay: float = 0.0
    seed: int = 0


def _group_by_task(buffer: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for ent in buffer:
        out.setdefault(ent["task_id"], []).append(ent)
    return out


def compute_advantages(buffer: list[dict]) -> list[float]:
    """Return per-entry advantage = G_i - b(task_id)."""
    by_task = _group_by_task(buffer)
    baseline = {tid: statistics.fmean(e["return_G"] for e in lst)
                for tid, lst in by_task.items()}
    return [e["return_G"] - baseline[e["task_id"]] for e in buffer]


def build_dataset(buffer: list[dict], n_criteria_by_task: dict[str, int],
                  cfg: AWConfig,
                  traj_featurizer=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (X states, y action_idx, w weights).

    traj_featurizer(records, n_criteria) -> list[(state_vec, action_idx)]
    Defaults to the coding featurizer for backward compat.
    """
    if traj_featurizer is None:
        def traj_featurizer(records, n_criteria):
            return _coding_featurize(records, action_space=None, n_criteria=n_criteria)
    advs = compute_advantages(buffer)
    weights_per_traj = []
    for a in advs:
        w = math.exp(a / cfg.temperature)
        w = max(cfg.weight_clip_min, min(w, cfg.weight_clip_max))
        weights_per_traj.append(w)

    xs, ys, ws = [], [], []
    for ent, w in zip(buffer, weights_per_traj):
        nc = n_criteria_by_task.get(ent["task_id"], 1)
        pairs = traj_featurizer(ent["records"], nc)
        for feats, aidx in pairs:
            xs.append(feats)
            ys.append(aidx)
            ws.append(w)
    X = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.long)
    w = torch.tensor(ws, dtype=torch.float32)
    return X, y, w


def train_aw(buffer: list[dict], n_criteria_by_task: dict[str, int],
             cfg: AWConfig | None = None,
             *, traj_featurizer=None) -> tuple[MLPPolicy, dict]:
    cfg = cfg or AWConfig()
    torch.manual_seed(cfg.seed)
    X, y, w = build_dataset(buffer, n_criteria_by_task, cfg,
                              traj_featurizer=traj_featurizer)
    if len(X) == 0:
        raise RuntimeError("empty buffer / no usable steps")
    policy = MLPPolicy(hidden=cfg.hidden)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate,
                            weight_decay=cfg.weight_decay)

    n = len(X)
    losses = []
    for ep in range(cfg.epochs):
        perm = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            xb, yb, wb = X[idx], y[idx], w[idx]
            logits = policy(xb)
            log_probs = F.log_softmax(logits, dim=-1)
            # Weighted negative log-likelihood (action policy loss)
            nll = -log_probs.gather(1, yb.unsqueeze(1)).squeeze(1)
            policy_loss = (wb * nll).mean()
            # Entropy bonus
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            loss = policy_loss - cfg.entropy_coef * entropy
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
        ep_loss /= n
        losses.append(ep_loss)

    diag = {
        "n_steps": int(len(X)),
        "n_trajectories": len(buffer),
        "config": asdict(cfg),
        "loss_per_epoch": losses,
        "weight_stats": {
            "min": float(w.min().item()),
            "max": float(w.max().item()),
            "mean": float(w.mean().item()),
            "std": float(w.std().item()),
        },
        "advantage_stats": {
            "mean": float(sum(compute_advantages(buffer)) / len(buffer)),
        },
    }
    return policy, diag
