"""Coding-domain Harness runner: drives the action loop with a policy.

This mirrors harness/agent.py:run_episode but is specialized to the
coding action space and accepts a `policy_callable(state, mask) -> action_name`.

Three policies are supported via factory functions:
  - base_harness_policy()        — fixed scripted policy (no LLM controller)
  - perturbed_base_policy(eps)   — eps-greedy perturbation of Base
  - mlp_policy(net, rng, ...)    — wraps an MLPPolicy + action mask

All emit B.1 trajectory records via TrajectoryLogger.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from harness import actions as A
from harness.actions import Task
from harness.submission import score_trajectory
from harness.trajectory_logger import TrajectoryLogger
from harness.util.llm_client import LLMClient

from .state_features import (CODING_ACTION_SPACE, ACTION_TO_IDX,
                              featurize_state)
from .policy import MLPPolicy, masked_probs, sample_action

log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Action execution (coding subset)
# ────────────────────────────────────────────────────────────────────────────

def _exec_action(action: str, task: Task, logger: TrajectoryLogger,
                 client: LLMClient) -> dict:
    if action == "read_problem":
        return A.act_read_problem(task)
    if action == "inspect_code":
        # alias for re-reading the code blob (cheap, no LLM)
        cb = logger.draft_state.get("code_blob") or ""
        return dict(status="success",
                    summary=f"inspect_code: {len(cb)} chars",
                    cost=0.0)
    if action == "write_code":
        return A.act_write_code(task, llm_client=client)
    if action == "run_tests":
        return A.act_run_tests(task,
                               code_blob=logger.draft_state.get("code_blob") or "")
    if action == "debug_error":
        # cheap reflective op: read last test errors
        errs = []
        for r in reversed(logger.records):
            tr = (r.get("observation") or {}).get("test_results")
            if tr:
                errs = tr.get("errors") or []
                break
        return dict(status="success",
                    summary=f"debug_error inspecting {len(errs)} errors",
                    cost=0.0)
    if action == "revise_code":
        errs = []
        for r in reversed(logger.records):
            tr = (r.get("observation") or {}).get("test_results")
            if tr:
                errs = tr.get("errors") or []
                break
        return A.act_revise_code(task,
                                 code_blob=logger.draft_state.get("code_blob") or "",
                                 test_errors=errs, llm_client=client)
    if action == "check_rubric":
        return A.act_check_rubric(task, draft_state=logger.draft_state)
    if action == "submit":
        return A.act_submit()
    return dict(status="error", summary=f"unknown action {action}", cost=0.0)


# ────────────────────────────────────────────────────────────────────────────
# Action mask: keep policy from picking obviously-invalid actions
# ────────────────────────────────────────────────────────────────────────────

def action_mask(logger: TrajectoryLogger) -> list[bool]:
    """Return a boolean mask over CODING_ACTION_SPACE indicating which
    actions are currently legal."""
    has_code = bool(logger.draft_state.get("code_blob"))
    has_test = any((r.get("observation") or {}).get("test_results")
                   for r in logger.records)
    last_failed = False
    for r in reversed(logger.records):
        tr = (r.get("observation") or {}).get("test_results")
        if tr:
            last_failed = (tr.get("failed", 0) > 0)
            break
    submitted_already = any(r.get("action") == "submit" for r in logger.records)
    if submitted_already:
        return [False] * len(CODING_ACTION_SPACE)
    mask = [True] * len(CODING_ACTION_SPACE)
    if not has_code:
        # cannot run_tests / revise / debug if no code yet
        for a in ("run_tests", "revise_code", "debug_error", "inspect_code"):
            mask[ACTION_TO_IDX[a]] = False
    if not has_test:
        mask[ACTION_TO_IDX["debug_error"]] = False
    if not has_code:
        # discourage submit before any code
        mask[ACTION_TO_IDX["submit"]] = False
    return mask


# ────────────────────────────────────────────────────────────────────────────
# Episode runner that takes a policy callable
# ────────────────────────────────────────────────────────────────────────────

PolicyFn = Callable[[list[float], list[bool], "EnvContext"], str]


@dataclass
class EnvContext:
    """Per-step info handed to policy callable beyond (state, mask)."""
    task_id: str
    step: int
    last_action: str | None
    rng: random.Random


def _state_vec(task: Task, logger: TrajectoryLogger, last_action: str | None,
               error_count: int, last_test: dict | None) -> list[float]:
    return featurize_state(
        step=len(logger.records),
        max_steps=task.max_steps,
        cost_so_far=logger.total_cost,
        cost_budget=task.cost_budget,
        draft_state=logger.draft_state,
        rubric_status=logger.rubric_status,
        last_action=last_action,
        error_count=error_count,
        last_test_results=last_test,
        n_criteria=len(task.rubric.get("criteria", [])),
    )


def run_episode_with_policy(task: Task, policy_fn: PolicyFn, *,
                             client: LLMClient,
                             rng: random.Random,
                             max_steps_override: int | None = None) -> tuple[TrajectoryLogger, dict]:
    max_steps = max_steps_override or task.max_steps
    n_criteria = len(task.rubric.get("criteria", []))
    logger = TrajectoryLogger(
        task_id=task.task_id, task_type=task.task_type,
        available_tools=task.available_tools,
        max_steps=max_steps, cost_budget=task.cost_budget,
        n_criteria=n_criteria,
    )
    last_action = None
    error_count = 0
    last_test = None
    while len(logger.records) < max_steps:
        if logger.total_cost >= task.cost_budget:
            logger.log_step(action="submit", args={}, status="success",
                            summary="forced submit due to budget", cost=0.0)
            logger.finalize(termination_reason="budget_exceeded")
            break
        state = _state_vec(task, logger, last_action, error_count, last_test)
        mask = action_mask(logger)
        if not any(mask):
            # nothing legal → force submit
            logger.log_step(action="submit", args={}, status="success",
                            summary="forced submit (no legal action)", cost=0.0)
            logger.finalize(termination_reason="submit")
            break
        ctx = EnvContext(task_id=task.task_id, step=len(logger.records),
                          last_action=last_action, rng=rng)
        action = policy_fn(state, mask, ctx)
        # safety: ensure picked action passes mask
        if not mask[ACTION_TO_IDX[action]]:
            # fall back to first legal action
            for i, ok in enumerate(mask):
                if ok:
                    action = CODING_ACTION_SPACE[i]
                    break
        obs = _exec_action(action, task, logger, client)
        logger.log_step(
            action=action, args={},
            status=obs.get("status", "success"),
            summary=obs.get("summary", ""),
            extracted_facts=obs.get("extracted_facts"),
            test_results=obs.get("test_results"),
            tool_name=obs.get("tool_name"),
            tool_args_valid=obs.get("tool_args_valid", True),
            cost=obs.get("cost", 0.0),
            draft_update=obs.get("draft_update"),
            rubric_update=obs.get("rubric_update"),
        )
        if obs.get("status") == "error":
            error_count += 1
        if obs.get("test_results"):
            last_test = obs["test_results"]
        last_action = action
        if action == "submit":
            logger.finalize(termination_reason="submit")
            break
    else:
        if not logger.records or not logger.records[-1].get("terminal"):
            # force final submit if hit max_steps
            logger.log_step(action="submit", args={}, status="success",
                            summary="forced submit at max_steps", cost=0.0)
            logger.finalize(termination_reason="max_steps")

    scored = score_trajectory(
        task=task, trajectory=logger.records,
        draft_state=logger.draft_state, rubric_status=logger.rubric_status,
        total_cost=logger.total_cost,
    )
    term = logger.records[-1]
    term["final_rubric_score"] = {
        "rubric_score_norm": scored["rubric_score_norm"],
        "rubric_score_raw": scored["rubric_score_raw"],
        "missing_items": scored["missing_items"],
    }
    return logger, scored


# ────────────────────────────────────────────────────────────────────────────
# Policy factories
# ────────────────────────────────────────────────────────────────────────────

_BASE_SEQ = ["read_problem", "write_code", "run_tests", "submit"]


def base_harness_policy() -> PolicyFn:
    """Fixed scripted policy: read_problem → write_code → run_tests → (revise → run_tests)? → submit."""
    def _picker(state: list[float], mask: list[bool], ctx: EnvContext) -> str:
        last = ctx.last_action
        # if last action was run_tests with failures and we still have steps
        # find in records via state vector last_test fields? Simpler: detect via last_action
        if last is None:
            return "read_problem" if mask[ACTION_TO_IDX["read_problem"]] else _first_legal(mask)
        if last == "read_problem":
            return "write_code" if mask[ACTION_TO_IDX["write_code"]] else _first_legal(mask)
        if last == "write_code":
            return "run_tests" if mask[ACTION_TO_IDX["run_tests"]] else "submit"
        if last == "run_tests":
            # Inspect last_test_pass_rate from state vector (index 8)
            # If failed (pass<1.0 and pass>=0) try revise; else submit
            pr = state[8]
            if 0.0 <= pr < 1.0 and mask[ACTION_TO_IDX["revise_code"]]:
                return "revise_code"
            return "submit"
        if last == "revise_code":
            return "run_tests" if mask[ACTION_TO_IDX["run_tests"]] else "submit"
        return "submit"
    return _picker


def perturbed_base_policy(eps: float = 0.25, seed: int = 0) -> PolicyFn:
    """ε-greedy perturbation of Base Harness for behavioral data collection."""
    base = base_harness_policy()
    def _picker(state: list[float], mask: list[bool], ctx: EnvContext) -> str:
        if ctx.rng.random() < eps:
            # uniform random over legal actions
            legal = [a for a, ok in zip(CODING_ACTION_SPACE, mask) if ok]
            return ctx.rng.choice(legal) if legal else "submit"
        return base(state, mask, ctx)
    return _picker


def mlp_policy(net: MLPPolicy, greedy: bool = False) -> PolicyFn:
    def _picker(state: list[float], mask: list[bool], ctx: EnvContext) -> str:
        logits = net.action_logits(state)
        idx = sample_action(logits, mask, greedy=greedy, rng=ctx.rng)
        return CODING_ACTION_SPACE[idx]
    return _picker


def _first_legal(mask: list[bool]) -> str:
    for i, ok in enumerate(mask):
        if ok:
            return CODING_ACTION_SPACE[i]
    return "submit"
