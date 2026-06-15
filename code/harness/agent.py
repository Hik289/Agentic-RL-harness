"""Base Harness agent — fixed scripted policy that runs to submit on any task.

Per-domain templates (action sequences):
  knowledge_work : read_input → draft_solution → check_rubric → submit
  coding         : read_problem → write_code → run_tests → revise_code? → submit
  research       : search → search → draft_solution → check_rubric → submit
  multi_tool     : observe → use_tool(read_pdf) → use_tool(extract_table)
                   → use_tool(calculator) → draft_solution → check_rubric → submit
  long_memory    : search_memory → search_memory → draft_solution → submit
  planning       : observe → plan → verify_solution → submit

Each step writes a fully populated trajectory record (per spec B.1) so HMS
detector + reward aggregator can run end-to-end.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import actions as A
from .actions import Task
from .state_builder import build_state
from .submission import score_trajectory
from .trajectory_logger import TrajectoryLogger
from .util.llm_client import LLMClient, get_default_client

log = logging.getLogger(__name__)


def _seq_for(task_type: str) -> list[tuple[str, dict]]:
    """Return (action_name, args) sequence."""
    if task_type in ("knowledge_work", "knowledge_work_deliverable"):
        return [
            ("read_input", {}),
            ("draft_solution", {}),
            ("check_rubric", {}),
            ("submit", {}),
        ]
    if task_type == "coding":
        return [
            ("read_problem", {}),
            ("write_code", {}),
            ("run_tests", {}),
            # Optionally a revise step is inserted dynamically (see run_episode)
            ("submit", {}),
        ]
    if task_type == "research":
        return [
            ("search", {"query": ""}),
            ("search", {"query": "compare evidence"}),
            ("draft_solution", {}),
            ("check_rubric", {}),
            ("submit", {}),
        ]
    if task_type == "multi_tool":
        return [
            ("observe", {}),
            ("use_tool", {"tool_name": "read_pdf"}),
            ("use_tool", {"tool_name": "extract_table"}),
            ("use_tool", {"tool_name": "calculator", "tool_args": {"expr": "12*2"}}),
            ("draft_solution", {}),
            ("check_rubric", {}),
            ("submit", {}),
        ]
    if task_type == "long_memory":
        return [
            ("search_memory", {"query": ""}),
            ("search_memory", {"query": "first session"}),
            ("draft_solution", {}),
            ("submit", {}),
        ]
    if task_type == "planning":
        return [
            ("observe", {}),
            ("plan", {}),
            ("verify_solution", {}),
            ("submit", {}),
        ]
    # Fallback
    return [("observe", {}), ("draft_solution", {}), ("submit", {})]


def _exec(action_name: str, task: Task, logger: TrajectoryLogger,
          client: LLMClient, args: dict) -> dict:
    """Execute one action and return obs dict."""
    if action_name == "read_input":
        return A.act_read_input(task, **args)
    if action_name == "read_problem":
        return A.act_read_problem(task)
    if action_name == "search":
        return A.act_search(task, **args)
    if action_name == "search_memory":
        return A.act_search_memory(task, **args)
    if action_name == "use_tool":
        return A.act_use_tool(task, llm_client=client, **args)
    if action_name == "draft_solution":
        facts = []
        for r in logger.records:
            facts.extend((r.get("observation") or {}).get("extracted_facts") or [])
        return A.act_draft_solution(task, facts_so_far=facts, llm_client=client)
    if action_name == "write_code":
        return A.act_write_code(task, llm_client=client)
    if action_name == "run_tests":
        return A.act_run_tests(task, code_blob=logger.draft_state.get("code_blob") or "")
    if action_name == "revise_code":
        errs = []
        for r in reversed(logger.records):
            tr = (r.get("observation") or {}).get("test_results")
            if tr:
                errs = tr.get("errors") or []
                break
        return A.act_revise_code(task,
                                 code_blob=logger.draft_state.get("code_blob") or "",
                                 test_errors=errs, llm_client=client)
    if action_name == "check_rubric":
        return A.act_check_rubric(task, draft_state=logger.draft_state)
    if action_name == "verify_solution":
        return A.act_verify_solution(task, draft_state=logger.draft_state)
    if action_name == "observe":
        return A.act_observe(task)
    if action_name == "plan":
        return A.act_plan(task, llm_client=client)
    if action_name == "submit":
        return A.act_submit()
    return dict(status="error", summary=f"unknown action {action_name}", cost=0.0)


def run_episode(
    task: Task,
    *,
    client: LLMClient | None = None,
    max_steps_override: int | None = None,
) -> tuple[TrajectoryLogger, dict]:
    """Run one episode of Base Harness. Returns (logger, scored_result)."""
    client = client or get_default_client()
    max_steps = max_steps_override or task.max_steps
    n_criteria = len(task.rubric.get("criteria", []))
    logger = TrajectoryLogger(
        task_id=task.task_id, task_type=task.task_type,
        available_tools=task.available_tools,
        max_steps=max_steps, cost_budget=task.cost_budget,
        n_criteria=n_criteria,
    )
    seq = _seq_for(task.task_type)

    seq_idx = 0
    while seq_idx < len(seq) and len(logger.records) < max_steps:
        action_name, args = seq[seq_idx]

        # Coding: if last run_tests had failures and we still have steps,
        # insert revise_code + run_tests before submit (at most twice).
        if action_name == "submit" and task.task_type == "coding":
            last = next((r for r in reversed(logger.records)
                         if (r.get("observation") or {}).get("test_results")),
                        None)
            if last:
                tr = (last.get("observation") or {}).get("test_results") or {}
                if tr.get("failed", 0) > 0 and len(logger.records) <= max_steps - 3:
                    # Splice in revise + rerun
                    seq.insert(seq_idx, ("revise_code", {}))
                    seq.insert(seq_idx + 1, ("run_tests", {}))
                    action_name, args = seq[seq_idx]

        # Cost guard
        if logger.total_cost >= task.cost_budget:
            logger.log_step(action="submit",
                            args={}, status="success",
                            summary="forced submit due to budget", cost=0.0)
            logger.finalize(termination_reason="budget_exceeded")
            break

        obs = _exec(action_name, task, logger, client, args)
        logger.log_step(
            action=action_name, args=args,
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
        if action_name == "submit":
            logger.finalize(termination_reason="submit")
            break
        seq_idx += 1
    else:
        # exited loop without submit
        if not logger.records or not logger.records[-1].get("terminal"):
            logger.finalize(termination_reason="max_steps")

    # Compute post-hoc score
    scored = score_trajectory(
        task=task, trajectory=logger.records,
        draft_state=logger.draft_state, rubric_status=logger.rubric_status,
        total_cost=logger.total_cost,
    )
    # Attach to terminal record
    term = logger.records[-1]
    term["final_rubric_score"] = {
        "rubric_score_norm": scored["rubric_score_norm"],
        "rubric_score_raw": scored["rubric_score_raw"],
        "missing_items": scored["missing_items"],
    }
    return logger, scored
