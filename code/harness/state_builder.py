"""State builder — readme §10.

Produces a compact state dict for the controller from the current trajectory.
For Base Harness this isn't fed to a learned controller; it's recorded for
inspection / future RL training.
"""
from __future__ import annotations

from typing import Any


def build_state(*, task: dict, trajectory: list, draft_state: dict,
                rubric_status: dict, total_cost: float) -> dict:
    last_action = trajectory[-1]["action"] if trajectory else None
    last_reward = 0.0  # filled later by step-shaped reward computation
    step = len(trajectory)
    return {
        "task_type": task.get("task_type"),
        "step": step,
        "has_draft": draft_state.get("has_draft", False),
        "draft_length": draft_state.get("draft_len_chars", 0),
        "rubric_coverage": rubric_status.get("coverage"),
        "verification_score": None,
        "error_count": sum(
            1 for r in trajectory
            if (r.get("observation") or {}).get("status") == "error"
        ),
        "format_score": None,
        "last_action": last_action,
        "last_reward": last_reward,
        "cost_so_far": total_cost,
        "remaining_steps": task.get("max_steps", 10) - step,
        "task_max_steps": task.get("max_steps", 10),
        "task_id": task.get("task_id"),
    }
