"""KW (knowledge_work_deliverable) harness: action loop driven by a policy.

KW action space (8 actions): see kw_state_features.KW_ACTION_SPACE.

Base policy template (matches harness/agent.py:_seq_for):
  read_material → draft_deliverable → check_rubric → submit
plus an optional `verify_evidence` and `extract_table` step from perturbed
exploration.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Callable

import torch

from harness import actions as A
from harness.actions import Task
from harness.submission import score_trajectory
from harness.trajectory_logger import TrajectoryLogger
from harness.util.llm_client import LLMClient

from .kw_state_features import (KW_ACTION_SPACE, ACTION_TO_IDX, featurize_state)
from .policy import MLPPolicy, masked_probs, sample_action

log = logging.getLogger(__name__)


def _exec_action(action: str, task: Task, logger: TrajectoryLogger,
                 client: LLMClient) -> dict:
    if action == "read_material":
        return A.act_read_input(task)
    if action == "extract_table":
        # Re-use read_input with a focused summary (cheap, no LLM)
        return A.act_use_tool(task, tool_name="extract_table",
                               tool_args={}, llm_client=client)
    if action == "summarize_material":
        # Light LLM summary of all read facts so far
        facts = []
        for r in logger.records:
            facts.extend((r.get("observation") or {}).get("extracted_facts") or [])
        return dict(status="success",
                    summary=f"summarized {len(facts)} facts",
                    cost=0.0,
                    extracted_facts=facts[:5])
    if action == "compare_evidence":
        facts = []
        for r in logger.records:
            facts.extend((r.get("observation") or {}).get("extracted_facts") or [])
        return dict(status="success",
                    summary=f"compared {len(facts)} facts",
                    cost=0.0)
    if action == "draft_deliverable":
        facts = []
        for r in logger.records:
            facts.extend((r.get("observation") or {}).get("extracted_facts") or [])
        return A.act_draft_solution(task, facts_so_far=facts, llm_client=client)
    if action == "verify_evidence":
        return A.act_verify_solution(task, draft_state=logger.draft_state)
    if action == "check_rubric":
        return A.act_check_rubric(task, draft_state=logger.draft_state)
    if action == "submit":
        return A.act_submit()
    return dict(status="error", summary=f"unknown action {action}", cost=0.0)


def action_mask(logger: TrajectoryLogger) -> list[bool]:
    has_draft = bool(logger.draft_state.get("has_draft"))
    submitted = any(r.get("action") == "submit" for r in logger.records)
    if submitted:
        return [False] * len(KW_ACTION_SPACE)
    mask = [True] * len(KW_ACTION_SPACE)
    if not has_draft:
        # disallow check_rubric / verify_evidence / submit before there's a draft
        for a in ("check_rubric", "verify_evidence", "submit"):
            mask[ACTION_TO_IDX[a]] = False
    return mask


PolicyFn = Callable[[list[float], list[bool], "EnvContext"], str]


@dataclass
class EnvContext:
    task_id: str
    step: int
    last_action: str | None
    rng: random.Random


def _n_facts(logger: TrajectoryLogger) -> int:
    return sum(len((r.get("observation") or {}).get("extracted_facts") or [])
               for r in logger.records)


def _state_vec(task: Task, logger: TrajectoryLogger,
               last_action: str | None, error_count: int) -> list[float]:
    return featurize_state(
        step=len(logger.records),
        max_steps=task.max_steps,
        cost_so_far=logger.total_cost,
        cost_budget=task.cost_budget,
        draft_state=logger.draft_state,
        rubric_status=logger.rubric_status,
        last_action=last_action,
        error_count=error_count,
        n_facts_collected=_n_facts(logger),
        n_criteria=len(task.rubric.get("criteria", [])),
    )


def run_episode_with_policy(task: Task, policy_fn: PolicyFn, *,
                             client: LLMClient, rng: random.Random,
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
    while len(logger.records) < max_steps:
        if logger.total_cost >= task.cost_budget:
            logger.log_step(action="submit", args={}, status="success",
                            summary="forced submit due to budget", cost=0.0)
            logger.finalize(termination_reason="budget_exceeded")
            break
        state = _state_vec(task, logger, last_action, error_count)
        mask = action_mask(logger)
        if not any(mask):
            logger.log_step(action="submit", args={}, status="success",
                            summary="forced submit (no legal action)", cost=0.0)
            logger.finalize(termination_reason="submit")
            break
        ctx = EnvContext(task_id=task.task_id, step=len(logger.records),
                          last_action=last_action, rng=rng)
        action = policy_fn(state, mask, ctx)
        if not mask[ACTION_TO_IDX[action]]:
            for i, ok in enumerate(mask):
                if ok:
                    action = KW_ACTION_SPACE[i]
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
        last_action = action
        if action == "submit":
            logger.finalize(termination_reason="submit")
            break
    else:
        if not logger.records or not logger.records[-1].get("terminal"):
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


# ── Policy factories ──────────────────────────────────────────────────────

def base_harness_policy() -> PolicyFn:
    """Fixed scripted policy: read_material → draft_deliverable → check_rubric → submit."""
    def _picker(state: list[float], mask: list[bool], ctx: EnvContext) -> str:
        last = ctx.last_action
        if last is None:
            return "read_material" if mask[ACTION_TO_IDX["read_material"]] else _first_legal(mask)
        if last == "read_material":
            return "draft_deliverable" if mask[ACTION_TO_IDX["draft_deliverable"]] else _first_legal(mask)
        if last == "draft_deliverable":
            return "check_rubric" if mask[ACTION_TO_IDX["check_rubric"]] else "submit"
        if last == "check_rubric":
            return "submit"
        return "submit"
    return _picker


def perturbed_base_policy(eps: float = 0.25, seed: int = 0) -> PolicyFn:
    base = base_harness_policy()
    def _picker(state: list[float], mask: list[bool], ctx: EnvContext) -> str:
        if ctx.rng.random() < eps:
            legal = [a for a, ok in zip(KW_ACTION_SPACE, mask) if ok]
            return ctx.rng.choice(legal) if legal else "submit"
        return base(state, mask, ctx)
    return _picker


def mlp_policy(net: MLPPolicy, greedy: bool = False) -> PolicyFn:
    def _picker(state: list[float], mask: list[bool], ctx: EnvContext) -> str:
        logits = net.action_logits(state)
        idx = sample_action(logits, mask, greedy=greedy, rng=ctx.rng)
        return KW_ACTION_SPACE[idx]
    return _picker


def _first_legal(mask: list[bool]) -> str:
    for i, ok in enumerate(mask):
        if ok:
            return KW_ACTION_SPACE[i]
    return "submit"
