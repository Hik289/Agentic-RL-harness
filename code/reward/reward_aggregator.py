"""Episode-level reward aggregator (readme §12).

R_total =
    R_rubric
  + α * R_verify
  + β * R_format
  + δ * R_task
  - γ * P_error
  - λ * P_cost
  - μ * P_early_submit

Default coefficients (readme §12):
  α = 0.20  (verification)
  β = 0.10  (format)
  δ = 0.20  (task native)
  γ = 0.25  (error)
  λ = 0.05  (cost)
  μ = 0.10  (early submit)

All sub-rewards are normalized to roughly [0, 1] before aggregation.
The aggregator is a pure function — no I/O, no LLM calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# Default τ thresholds (readme §12.7)
DEFAULT_TAU_INFO = 0.5
DEFAULT_TAU_RUBRIC = 0.6

DEFAULT_COEFS = {
    "alpha": 0.20,
    "beta": 0.10,
    "delta": 0.20,
    "gamma": 0.25,
    "lam": 0.05,
    "mu": 0.10,
}


@dataclass
class RewardBreakdown:
    """All sub-reward terms and the final R_total."""
    R_rubric: float = 0.0
    R_verify: float = 0.0
    R_format: float = 0.0
    R_task: float = 0.0
    P_error: float = 0.0
    P_cost: float = 0.0
    P_early_submit: float = 0.0
    # Weighted contributions for transparency
    weighted_verify: float = 0.0
    weighted_format: float = 0.0
    weighted_task: float = 0.0
    weighted_error: float = 0.0
    weighted_cost: float = 0.0
    weighted_early_submit: float = 0.0
    R_total: float = 0.0
    coefs: dict = field(default_factory=lambda: DEFAULT_COEFS.copy())

    def to_dict(self) -> dict:
        return asdict(self)


def aggregate_reward(
    *,
    R_rubric: float,
    R_verify: float = 0.0,
    R_format: float = 0.0,
    R_task: float = 0.0,
    P_error: float = 0.0,
    P_cost: float = 0.0,
    P_early_submit: float = 0.0,
    coefs: Optional[dict] = None,
) -> RewardBreakdown:
    """Aggregate the episode-level rewards into R_total.

    All inputs assumed already normalized (rubric in [0,1], penalties in [0,1]).
    """
    c = DEFAULT_COEFS.copy()
    if coefs:
        c.update(coefs)

    w_verify = c["alpha"] * R_verify
    w_format = c["beta"] * R_format
    w_task = c["delta"] * R_task
    w_error = c["gamma"] * P_error
    w_cost = c["lam"] * P_cost
    w_early = c["mu"] * P_early_submit

    R_total = (
        R_rubric
        + w_verify
        + w_format
        + w_task
        - w_error
        - w_cost
        - w_early
    )

    return RewardBreakdown(
        R_rubric=R_rubric,
        R_verify=R_verify,
        R_format=R_format,
        R_task=R_task,
        P_error=P_error,
        P_cost=P_cost,
        P_early_submit=P_early_submit,
        weighted_verify=w_verify,
        weighted_format=w_format,
        weighted_task=w_task,
        weighted_error=w_error,
        weighted_cost=w_cost,
        weighted_early_submit=w_early,
        R_total=R_total,
        coefs=c,
    )


def compute_rubric_score(criteria_scores: list[dict]) -> dict:
    """Aggregate per-criterion scores → rubric_score_raw/norm + missing_items.

    criteria_scores: list of {id, score, max_score, missing(bool), category?}
    Returns dict matching data_scientist's spec (rubric_design_spec.md A.4).
    """
    if not criteria_scores:
        return {
            "rubric_score_raw": 0.0,
            "rubric_score_norm": 0.0,
            "missing_items": [],
            "by_category": {},
        }
    raw = sum(c["score"] for c in criteria_scores)
    total = sum(c["max_score"] for c in criteria_scores)
    norm = raw / total if total > 0 else 0.0
    missing = [c["id"] for c in criteria_scores if c.get("missing", False)]

    by_cat: dict[str, dict] = {}
    for c in criteria_scores:
        cat = c.get("category", "uncategorized")
        slot = by_cat.setdefault(cat, {"score": 0.0, "max": 0.0, "ids": []})
        slot["score"] += c["score"]
        slot["max"] += c["max_score"]
        slot["ids"].append(c["id"])

    return {
        "rubric_score_raw": raw,
        "rubric_score_norm": norm,
        "missing_items": missing,
        "by_category": by_cat,
    }


def compute_error_penalty(error_count: int, max_steps: int) -> float:
    """P_error = min(error_count / max_steps, 1.0). Normalize by horizon."""
    if max_steps <= 0:
        return 0.0
    return min(max(error_count, 0) / max_steps, 1.0)


def compute_early_submit_penalty(
    *,
    information_coverage: float,
    rubric_coverage: float,
    tau_info: float = DEFAULT_TAU_INFO,
    tau_rubric: float = DEFAULT_TAU_RUBRIC,
) -> float:
    """P_early_submit = 1 if information_coverage < τ_info OR rubric_coverage < τ_rubric else 0."""
    return 1.0 if (information_coverage < tau_info or rubric_coverage < tau_rubric) else 0.0
