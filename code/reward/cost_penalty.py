"""Cost penalty (readme §12.6).

P_cost = min(cost_so_far / cost_budget, 1.0)
"""
from __future__ import annotations


def cost_penalty(cost_so_far: float, cost_budget: float) -> float:
    if cost_budget <= 0:
        return 0.0
    return min(max(cost_so_far, 0.0) / cost_budget, 1.0)
