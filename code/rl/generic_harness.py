"""Generic per-domain harness for research / multi_tool / long_memory / planning.

Each domain has its own action space (read from task.available_tools), but
all domains follow the same skeleton:

  Base policy template (one of):
    - research: search → search → draft_answer → check_rubric → submit
    - multi_tool: observe → use_tool(read_pdf) → use_tool(extract_table)
                  → use_tool(calculator) → draft_solution → check_rubric → submit
    - long_memory: search_memory → search_memory → draft (answer_question) → submit
    - planning: observe → draft_solution → verify_solution → submit

The harness:
  - Takes a domain string and looks up its action space + base policy seq.
  - Executes generic actions (search/draft/check/etc) via harness/actions.py.
  - Emits B.1 trajectory records.
  - Same MLP policy / state featurization as KW.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Callable

import torch

from harness import actions as A
from harness.actions import Task
from harness.submission import score_trajectory
from harness.trajectory_logger import TrajectoryLogger
from harness.util.llm_client import LLMClient

from .policy import MLPPolicy, masked_probs, sample_action

log = logging.getLogger(__name__)


# Domain → action space (must match task.json available_tools order)
DOMAIN_ACTION_SPACE = {
    "research": [
        "search", "open_source", "extract_claims", "compare_sources",
        "draft_answer", "cite_sources", "check_rubric", "submit",
    ],
    "multi_tool": [
        "observe", "use_tool_read_pdf", "use_tool_extract_table",
        "use_tool_calculator", "run_code", "draft_solution",
        "check_rubric", "submit",
    ],
    "long_memory": [
        "search_memory", "retrieve_session", "read_memory_chunk",
        "compare_memory_facts", "answer_question", "check_rubric", "submit",
    ],
    "planning": [
        "observe", "draft_solution", "verify_solution",
        "revise_solution", "check_rubric", "submit",
    ],
}

# State dim is always 10 + len(action_space) (one-hot last action).
# To keep MLPPolicy compatible across domains (it's 18-dim), we pad action
# space to length 8 with no-op slots.
PAD_ACTION_DIM = 8

# Domain → base policy sequence (what Base Harness does)
DOMAIN_BASE_SEQ = {
    "research": ["search", "search", "draft_answer", "check_rubric", "submit"],
    "multi_tool": ["observe", "use_tool_read_pdf", "use_tool_extract_table",
                    "use_tool_calculator", "draft_solution", "check_rubric", "submit"],
    "long_memory": ["search_memory", "search_memory", "answer_question",
                     "check_rubric", "submit"],
    "planning": ["observe", "draft_solution", "verify_solution",
                  "check_rubric", "submit"],
}


def _pad(action_space: list[str]) -> list[str]:
    out = list(action_space)
    while len(out) < PAD_ACTION_DIM:
        out.append(f"__noop_{len(out)}__")
    return out[:PAD_ACTION_DIM]


def domain_action_space(domain: str) -> list[str]:
    return _pad(DOMAIN_ACTION_SPACE[domain])


def domain_action_to_idx(domain: str) -> dict[str, int]:
    return {a: i for i, a in enumerate(domain_action_space(domain))}


# ── Featurizer (10 numeric + 8 one-hot) ────────────────────────────────────

def featurize_state(*, step: int, max_steps: int,
                    cost_so_far: float, cost_budget: float,
                    draft_state: dict, rubric_status: dict,
                    last_action: str | None, error_count: int,
                    n_facts_collected: int, n_criteria: int,
                    domain: str) -> list[float]:
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
    facts_norm = min(n_facts_collected / 10.0, 1.0)
    claims = draft_state.get("claims") or []
    cwe = draft_state.get("claims_with_evidence") or []
    cwe_ratio = (len(cwe) / max(len(claims), 1)) if claims else 0.0
    numeric = [step_norm, has_draft, draft_len_norm, rubric_coverage,
               rubric_missing_norm, error_count_norm, cost_norm, rem_norm,
               facts_norm, cwe_ratio]
    a2i = domain_action_to_idx(domain)
    one_hot = [0.0] * PAD_ACTION_DIM
    if last_action in a2i:
        one_hot[a2i[last_action]] = 1.0
    return numeric + one_hot


def trajectory_to_features(records: list, n_criteria: int,
                           domain: str) -> list[tuple[list[float], int]]:
    a2i = domain_action_to_idx(domain)
    pairs = []
    error_count = 0
    last_action = None
    n_facts = 0
    draft = {"has_draft": False, "draft_len_chars": 0,
             "claims": [], "claims_with_evidence": []}
    rubric = {"last_checked_step": None, "coverage": None, "missing_ids": None}
    cost_so_far = 0.0
    if not records:
        return pairs
    max_steps = records[0].get("max_steps") or records[0].get("task_max_steps") or 10
    cost_budget = records[0].get("cost_budget") or 1.0
    for t, rec in enumerate(records):
        action = rec.get("action")
        if action not in a2i:
            continue
        feats = featurize_state(
            step=t, max_steps=max_steps,
            cost_so_far=cost_so_far, cost_budget=cost_budget,
            draft_state=draft, rubric_status=rubric,
            last_action=last_action, error_count=error_count,
            n_facts_collected=n_facts, n_criteria=n_criteria,
            domain=domain,
        )
        pairs.append((feats, a2i[action]))
        obs = rec.get("observation") or {}
        draft = rec.get("draft_state") or {}
        rubric = rec.get("rubric_status") or {}
        cost_so_far += float(obs.get("cost", 0.0) or 0.0)
        if obs.get("status") == "error":
            error_count += 1
        n_facts += len(obs.get("extracted_facts") or [])
        last_action = action
    return pairs


# ── Action execution ─────────────────────────────────────────────────────

def _exec_action(domain: str, action: str, task: Task,
                 logger: TrajectoryLogger, client: LLMClient) -> dict:
    """Map domain-specific action name → harness/actions.py call."""
    # All domains share: check_rubric, submit
    if action == "check_rubric":
        return A.act_check_rubric(task, draft_state=logger.draft_state)
    if action == "submit":
        return A.act_submit()

    if domain == "research":
        if action == "search":
            return A.act_search(task, query="")
        if action == "open_source":
            return A.act_search(task, query="source")
        if action == "extract_claims":
            facts = [f for r in logger.records
                     for f in ((r.get("observation") or {}).get("extracted_facts") or [])]
            return dict(status="success",
                        summary=f"extracted claims from {len(facts)} facts",
                        cost=0.0, extracted_facts=facts[:5])
        if action == "compare_sources":
            facts = [f for r in logger.records
                     for f in ((r.get("observation") or {}).get("extracted_facts") or [])]
            return dict(status="success",
                        summary=f"compared {len(facts)} sources", cost=0.0)
        if action == "draft_answer":
            facts = [f for r in logger.records
                     for f in ((r.get("observation") or {}).get("extracted_facts") or [])]
            return A.act_draft_solution(task, facts_so_far=facts, llm_client=client)
        if action == "cite_sources":
            return dict(status="success", summary="citations added", cost=0.0)

    if domain == "multi_tool":
        if action == "observe":
            return A.act_observe(task)
        if action == "use_tool_read_pdf":
            return A.act_use_tool(task, tool_name="read_pdf",
                                   tool_args={}, llm_client=client)
        if action == "use_tool_extract_table":
            return A.act_use_tool(task, tool_name="extract_table",
                                   tool_args={}, llm_client=client)
        if action == "use_tool_calculator":
            return A.act_use_tool(task, tool_name="calculator",
                                   tool_args={"expr": "12*2"}, llm_client=client)
        if action == "run_code":
            return dict(status="success", summary="run_code (stub)", cost=0.0)
        if action == "draft_solution":
            facts = [f for r in logger.records
                     for f in ((r.get("observation") or {}).get("extracted_facts") or [])]
            return A.act_draft_solution(task, facts_so_far=facts, llm_client=client)

    if domain == "long_memory":
        if action == "search_memory":
            return A.act_search_memory(task, query="")
        if action == "retrieve_session":
            return A.act_search_memory(task, query="session")
        if action == "read_memory_chunk":
            return A.act_search_memory(task, query="memory")
        if action == "compare_memory_facts":
            facts = [f for r in logger.records
                     for f in ((r.get("observation") or {}).get("extracted_facts") or [])]
            return dict(status="success",
                        summary=f"compared {len(facts)} memory facts",
                        cost=0.0)
        if action == "answer_question":
            facts = [f for r in logger.records
                     for f in ((r.get("observation") or {}).get("extracted_facts") or [])]
            return A.act_draft_solution(task, facts_so_far=facts, llm_client=client)

    if domain == "planning":
        if action == "observe":
            return A.act_observe(task)
        if action == "draft_solution":
            return A.act_plan(task, llm_client=client)
        if action == "verify_solution":
            return A.act_verify_solution(task, draft_state=logger.draft_state)
        if action == "revise_solution":
            return A.act_plan(task, llm_client=client)

    if action.startswith("__noop_"):
        return dict(status="success", summary="noop pad slot", cost=0.0)

    return dict(status="error", summary=f"unknown action {action} for domain {domain}", cost=0.0)


# ── Action mask ──────────────────────────────────────────────────────────

def action_mask(domain: str, logger: TrajectoryLogger) -> list[bool]:
    space = domain_action_space(domain)
    a2i = domain_action_to_idx(domain)
    has_draft = bool(logger.draft_state.get("has_draft"))
    submitted = any(r.get("action") == "submit" for r in logger.records)
    if submitted:
        return [False] * len(space)
    mask = [True] * len(space)
    # Don't allow check_rubric / verify_solution / submit before draft
    for blocked in ("check_rubric", "verify_solution", "submit",
                    "revise_solution", "cite_sources"):
        if not has_draft and blocked in a2i:
            mask[a2i[blocked]] = False
    # Pad slots not selectable
    for k, idx in a2i.items():
        if k.startswith("__noop_"):
            mask[idx] = False
    return mask


# ── Policy factories ────────────────────────────────────────────────────

PolicyFn = Callable[[list[float], list[bool], "EnvContext"], str]


@dataclass
class EnvContext:
    task_id: str
    step: int
    last_action: str | None
    rng: random.Random
    domain: str


def base_harness_policy(domain: str) -> PolicyFn:
    seq = DOMAIN_BASE_SEQ[domain]
    space = domain_action_space(domain)
    a2i = domain_action_to_idx(domain)

    def _picker(state: list[float], mask: list[bool], ctx: EnvContext) -> str:
        # Use the current step count to index into the seq directly.
        # This handles duplicates (e.g. research seq has [search, search,...]).
        i = ctx.step
        if i < len(seq):
            nxt = seq[i]
            if nxt in a2i and mask[a2i[nxt]]:
                return nxt
        if "submit" in a2i and mask[a2i["submit"]]:
            return "submit"
        return _first_legal(mask, space)
    return _picker


def perturbed_base_policy(domain: str, eps: float = 0.25, seed: int = 0) -> PolicyFn:
    base = base_harness_policy(domain)
    space = domain_action_space(domain)
    def _picker(state: list[float], mask: list[bool], ctx: EnvContext) -> str:
        if ctx.rng.random() < eps:
            legal = [a for a, ok in zip(space, mask) if ok]
            return ctx.rng.choice(legal) if legal else "submit"
        return base(state, mask, ctx)
    return _picker


def mlp_policy(domain: str, net: MLPPolicy, greedy: bool = False) -> PolicyFn:
    space = domain_action_space(domain)
    def _picker(state: list[float], mask: list[bool], ctx: EnvContext) -> str:
        logits = net.action_logits(state)
        idx = sample_action(logits, mask, greedy=greedy, rng=ctx.rng)
        return space[idx]
    return _picker


def _first_legal(mask: list[bool], space: list[str]) -> str:
    for a, ok in zip(space, mask):
        if ok and not a.startswith("__noop_"):
            return a
    return "submit"


# ── State + Episode runner ───────────────────────────────────────────────

def _state_vec(task: Task, logger: TrajectoryLogger,
               last_action: str | None, error_count: int,
               domain: str) -> list[float]:
    n_facts = sum(len((r.get("observation") or {}).get("extracted_facts") or [])
                   for r in logger.records)
    return featurize_state(
        step=len(logger.records),
        max_steps=task.max_steps,
        cost_so_far=logger.total_cost,
        cost_budget=task.cost_budget,
        draft_state=logger.draft_state,
        rubric_status=logger.rubric_status,
        last_action=last_action,
        error_count=error_count,
        n_facts_collected=n_facts,
        n_criteria=len(task.rubric.get("criteria", [])),
        domain=domain,
    )


def run_episode_with_policy(domain: str, task: Task, policy_fn: PolicyFn, *,
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
    a2i = domain_action_to_idx(domain)
    space = domain_action_space(domain)
    last_action = None
    error_count = 0
    while len(logger.records) < max_steps:
        if logger.total_cost >= task.cost_budget:
            logger.log_step(action="submit", args={}, status="success",
                            summary="forced submit due to budget", cost=0.0)
            logger.finalize(termination_reason="budget_exceeded")
            break
        state = _state_vec(task, logger, last_action, error_count, domain)
        mask = action_mask(domain, logger)
        if not any(mask):
            logger.log_step(action="submit", args={}, status="success",
                            summary="forced submit (no legal action)", cost=0.0)
            logger.finalize(termination_reason="submit")
            break
        ctx = EnvContext(task_id=task.task_id, step=len(logger.records),
                          last_action=last_action, rng=rng, domain=domain)
        action = policy_fn(state, mask, ctx)
        if action not in a2i or not mask[a2i[action]]:
            action = _first_legal(mask, space)
        obs = _exec_action(domain, action, task, logger, client)
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
