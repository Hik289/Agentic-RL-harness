"""Submission finalizer — computes a post-hoc rubric_score for a trajectory.

For Base Harness we use a deterministic, structural rubric evaluation (no
LLM judge) so that anchor_2 doesn't burn API budget. Real rubric_judge.py
(LLM-based) will plug in here later by overriding `score_criterion`.
"""
from __future__ import annotations

import re
from typing import Any

from reward.format_checker import format_reward
from reward.reward_aggregator import (
    compute_rubric_score, compute_error_penalty,
    compute_early_submit_penalty, aggregate_reward,
)
from reward.cost_penalty import cost_penalty


def _format_schema_for(task_type: str) -> dict:
    if task_type in ("knowledge_work_deliverable", "knowledge_work"):
        return {"schema_type": "markdown_sections",
                "required_sections": ["Summary", "Findings", "Recommendation"]}
    if task_type == "long_memory":
        return {"schema_type": "json", "required_keys": ["answer", "source_session"]}
    if task_type == "planning":
        return {"schema_type": "plain", "min_chars": 30}
    return {"schema_type": "plain", "min_chars": 10}


def _struct_score(criterion: dict, draft_text: str, code_blob: str | None,
                  facts: list, test_results: dict | None,
                  task_type: str) -> float:
    """Cheap deterministic scorer used by Base Harness."""
    max_score = float(criterion.get("max_score", 1.0))
    verifier = criterion.get("verifier", "")
    desc = (criterion.get("description") or "").lower()
    text = (draft_text or code_blob or "").strip()

    if verifier == "format_checker":
        schema = _format_schema_for(task_type)
        return max_score * format_reward(text, schema)
    if verifier == "test_runner":
        if not test_results:
            return 0.0
        passed = test_results.get("passed", 0)
        failed = test_results.get("failed", 0)
        tot = passed + failed
        return max_score * (passed / tot) if tot > 0 else 0.0
    if verifier == "cost_budget":
        # Filled in by caller via aggregate; here use 1.0 if cheap text exists
        return max_score * (1.0 if text else 0.0)
    if verifier == "evidence_support":
        # Reward proportional to number of input facts referenced
        if not facts:
            return 0.0
        hit = sum(1 for f in facts if any(t in text.lower()
                                          for t in re.findall(r"\w+", f.lower())
                                          if len(t) > 4))
        ratio = min(hit / max(len(facts), 1), 1.0)
        return max_score * ratio
    if verifier == "criterion_coverage":
        words = [w for w in re.findall(r"[a-zA-Z']+", desc) if len(w) > 4]
        if not words:
            return max_score * (1.0 if text else 0.0)
        hit = sum(1 for w in words if w in text.lower())
        return max_score * (hit / len(words))
    if verifier == "answer_correctness":
        # No reference solver in Base Harness -> reward presence of an answer
        # roughly aligned with focus question keywords
        words = [w for w in re.findall(r"[a-zA-Z']+", desc) if len(w) > 4]
        if not words:
            return max_score * (0.5 if text else 0.0)
        hit = sum(1 for w in words if w in text.lower())
        return max_score * (hit / len(words)) * 0.6  # cap at 0.6 (placeholder)
    if verifier == "tool_call_valid":
        # Decided by caller from trajectory; default 1.0 if any text
        return max_score * (1.0 if text else 0.0)
    if verifier == "plan_satisfies_constraints":
        return max_score * (1.0 if "steps" in text.lower()
                             or "step 1" in text.lower() else 0.0)
    # default: heuristic — partial credit for non-empty
    return max_score * (0.5 if text else 0.0)


def score_trajectory(
    *,
    task,
    trajectory: list,
    draft_state: dict,
    rubric_status: dict,
    total_cost: float,
) -> dict:
    rubric = task.rubric
    criteria = rubric.get("criteria", [])
    facts_all = []
    last_test = None
    for rec in trajectory:
        obs = rec.get("observation") or {}
        facts_all.extend(obs.get("extracted_facts") or [])
        if obs.get("test_results"):
            last_test = obs["test_results"]
    text = draft_state.get("draft_text") or ""
    code = draft_state.get("code_blob") or ""

    per_crit = []
    for c in criteria:
        s = _struct_score(c, text, code, facts_all, last_test, task.task_type)
        per_crit.append({
            "id": c["id"],
            "score": s,
            "max_score": float(c["max_score"]),
            "missing": (s == 0.0) and c.get("required", True),
            "category": c.get("category"),
        })
    rubric_blob = compute_rubric_score(per_crit)

    # P_error: count error observations
    err_count = sum(1 for r in trajectory
                    if (r.get("observation") or {}).get("status") == "error")
    p_err = compute_error_penalty(err_count, task.max_steps)
    p_cost = cost_penalty(total_cost, task.cost_budget)
    info_cov = min(len(facts_all) / 5.0, 1.0)  # crude info coverage
    rubric_cov = rubric_status.get("coverage") or 0.0
    p_early = compute_early_submit_penalty(
        information_coverage=info_cov, rubric_coverage=rubric_cov,
    )
    R_verify = rubric_blob["rubric_score_norm"]  # placeholder until verifier
    R_format = format_reward(text or code, _format_schema_for(task.task_type))
    R_task = rubric_blob["rubric_score_norm"]
    R_rubric = rubric_blob["rubric_score_norm"]
    total = aggregate_reward(
        R_rubric=R_rubric, R_verify=R_verify, R_format=R_format,
        R_task=R_task, P_error=p_err, P_cost=p_cost,
        P_early_submit=p_early,
    )
    return {
        "rubric_score_raw": rubric_blob["rubric_score_raw"],
        "rubric_score_norm": rubric_blob["rubric_score_norm"],
        "missing_items": rubric_blob["missing_items"],
        "by_category": rubric_blob["by_category"],
        "per_criterion": per_crit,
        "R_total_breakdown": total.to_dict(),
        "P_error": p_err, "P_cost": p_cost, "P_early": p_early,
        "info_coverage": info_cov,
        "rubric_coverage": rubric_cov,
        "total_cost": total_cost,
    }
