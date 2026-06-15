"""1-layer MLP policy over the coding action space.

Pure PyTorch. CPU-only for anchor_5 (very small model, ~hundreds of
samples). Forward: state ∈ R^18 → logits ∈ R^8.

Inference returns a categorical distribution; the harness draws actions
by sampling (training) or argmax (eval, optional).

We also clip action probabilities so that an action mask can zero out
disallowed actions (e.g. revise_code when no test has been run, or
submit when no draft yet) and renormalize.
"""
from __future__ import annotations

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .state_features import CODING_ACTION_SPACE, STATE_DIM


class MLPPolicy(nn.Module):
    def __init__(self, hidden: int = 64,
                 n_actions: int = len(CODING_ACTION_SPACE)):
        super().__init__()
        self.fc1 = nn.Linear(STATE_DIM, hidden)
        self.fc2 = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        return self.fc2(h)

    def action_logits(self, state: list[float]) -> torch.Tensor:
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            return self.forward(x).squeeze(0)


def masked_probs(logits: torch.Tensor, mask: list[bool],
                 eps: float = 1e-9) -> torch.Tensor:
    mask_t = torch.tensor(mask, dtype=torch.bool)
    very_neg = torch.full_like(logits, -1e9)
    masked = torch.where(mask_t, logits, very_neg)
    return F.softmax(masked, dim=-1) + eps


def sample_action(logits: torch.Tensor, mask: list[bool],
                  greedy: bool = False, rng: random.Random | None = None) -> int:
    probs = masked_probs(logits, mask)
    if greedy:
        return int(torch.argmax(probs).item())
    if rng is None:
        return int(torch.multinomial(probs, 1).item())
    # deterministic sample via rng (for reproducible eval)
    r = rng.random()
    csum = 0.0
    p_np = probs.tolist()
    for i, p in enumerate(p_np):
        csum += p
        if r <= csum:
            return i
    return len(p_np) - 1
