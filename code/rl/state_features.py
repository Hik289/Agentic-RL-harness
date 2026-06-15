"""Numeric state feature extraction for the RL controller (readme §10).

Coding-domain subset:
  step_norm                = step / max_steps                      ∈ [0,1]
  has_draft                = float 0/1
  draft_len_norm           = min(draft_len_chars / 500, 1.0)
  rubric_coverage          ∈ [0,1]  (0 if never checked)
  rubric_missing_norm      = #missing / max(n_criteria, 1)
  error_count_norm         = error_count / max_steps
  cost_so_far_norm         = cost_so_far / cost_budget
  remaining_steps_norm     = (max_steps - step) / max_steps
  last_test_pass_rate      ∈ [0,1]  (-1 if no test yet)
  last_test_fail_count_norm= last_failed / 5 clipped to 1
  + one-hot last_action over CODING_ACTION_SPACE (8 dim)

Total dimension: 10 + 8 = 18.
"""
from __future__ import annotations

from typing import Any

CODING_ACTION_SPACE = [
    "read_problem",
    "inspect_code",
    "write_code",
    "run_tests",
    "debug_error",
    "revise_code",
    "check_rubric",
    "submit",
]

ACTION_TO_IDX = {a: i for i, a in enumerate(CODING_ACTION_SPACE)}
STATE_DIM = 10 + len(CODING_ACTION_SPACE)  # 18


def featurize_state(*, step: int, max_steps: int,
                    cost_so_far: float, cost_budget: float,
                    draft_state: dict, rubric_status: dict,
                    last_action: str | None, error_count: int,
                    last_test_results: dict | None,
                    n_criteria: int) -> list[float]:
    step_norm = min(step / max(max_steps, 1), 1.0)
    has_draft = 1.0 if draft_state.get("has_draft") else 0.0
    draft_len = draft_state.get("draft_len_chars", 0) or 0
    draft_len_norm = min(draft_len / 500.0, 1.0)
    coverage = rubric_status.get("coverage")
    rubric_coverage = float(coverage) if coverage is not None else 0.0
    missing = rubric_status.get("missing_ids") or []
    rubric_missing_norm = len(missing) / max(n_criteria, 1)
    error_count_norm = min(error_count / max(max_steps, 1), 1.0)
    cost_norm = min(cost_so_far / max(cost_budget, 1e-9), 1.0)
    rem_norm = max(0.0, (max_steps - step) / max(max_steps, 1))
    if last_test_results:
        passed = last_test_results.get("passed", 0)
        failed = last_test_results.get("failed", 0)
        tot = passed + failed
        last_pass_rate = passed / tot if tot > 0 else -1.0
        last_fail_norm = min(failed / 5.0, 1.0)
    else:
        last_pass_rate = -1.0
        last_fail_norm = 0.0

    numeric = [step_norm, has_draft, draft_len_norm, rubric_coverage,
               rubric_missing_norm, error_count_norm, cost_norm, rem_norm,
               last_pass_rate, last_fail_norm]
    one_hot = [0.0] * len(CODING_ACTION_SPACE)
    if last_action in ACTION_TO_IDX:
        one_hot[ACTION_TO_IDX[last_action]] = 1.0
    return numeric + one_hot


def trajectory_to_features(records: list, action_space: list,
                            n_criteria: int) -> list[tuple[list[float], int]]:
    """Convert a trajectory into (state, action_idx) pairs.

    The state at index t is what the policy saw BEFORE taking action a_t,
    i.e. computed from records[:t]; action_idx = index of records[t].action.
    """
    pairs = []
    error_count = 0
    last_action = None
    last_test = None
    draft = {"has_draft": False, "draft_len_chars": 0,
             "claims": [], "claims_with_evidence": [],
             "code_blob": None}
    rubric = {"last_checked_step": None, "coverage": None, "missing_ids": None}
    cost_so_far = 0.0
    if not records:
        return pairs
    max_steps = records[0].get("max_steps") or records[0].get("task_max_steps") or 10
    cost_budget = records[0].get("cost_budget") or 1.0

    for t, rec in enumerate(records):
        action = rec.get("action")
        if action not in ACTION_TO_IDX:
            continue
        feats = featurize_state(
            step=t, max_steps=max_steps,
            cost_so_far=cost_so_far, cost_budget=cost_budget,
            draft_state=draft, rubric_status=rubric,
            last_action=last_action, error_count=error_count,
            last_test_results=last_test, n_criteria=n_criteria,
        )
        pairs.append((feats, ACTION_TO_IDX[action]))
        # update running state
        obs = rec.get("observation") or {}
        ds = rec.get("draft_state") or {}
        rs = rec.get("rubric_status") or {}
        draft = ds
        rubric = rs
        cost_so_far += float(obs.get("cost", 0.0) or 0.0)
        if obs.get("status") == "error":
            error_count += 1
        if obs.get("test_results"):
            last_test = obs["test_results"]
        last_action = action
    return pairs
