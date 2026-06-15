"""HMS detector — 7 behavior events + aux (readme §20, operationalized in
data/rubric_design_spec.md Part C).

Input: a trajectory list[dict] (B.1 schema in rubric_design_spec.md):
  Each step record contains action, args, observation, draft_state,
  rubric_status, timestamp_ms, etc. The final record has terminal=True,
  termination_reason, final_rubric_score (optional).

Output:
  HMSResult(hms_raw, hms_norm, events: dict, aux: dict)

Events (per rubric_design_spec.md Part C):
  C.1 CheckBeforeSubmit   (+w1)
  C.2 EvidenceBeforeClaim (+w2)
  C.3 TestBeforeSubmit    (+w3)
  C.4 RevisionAfterFailure(+w4)
  C.5 ValidToolUse        (+w5)
  C.6 StopWhenSufficient  (+w6)
  C.7 EarlySubmit         (−w7)

Default weights all = 1 (readme §20 / spec Part D).

For each event, returns:
  {fired: bool, applicable: bool, explanation: str}
- applicable=False → event is skipped in aggregation (no bias for tasks
  whose action space doesn't include the action being measured).

Aux events (Part C.8, not in main HMS):
  RepeatedUselessAction → n_useless_runs, n_useless_steps
  CostOverrun           → bool
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

EVIDENCE_ACTIONS = {
    "read_input", "search", "retrieve_context", "retrieve_session",
    "read_memory_chunk", "search_memory", "extract_claims", "open_source",
}
REVISE_ACTIONS = {"revise_solution", "revise_code", "debug_error"}
TEST_ACTIONS = {"run_tests", "run_code"}
TOOL_CALL_ACTIONS = {"use_tool", "search", "run_code", "extract_table"}

DEFAULT_WEIGHTS = {
    "CheckBeforeSubmit": 1.0,
    "EvidenceBeforeClaim": 1.0,
    "TestBeforeSubmit": 1.0,
    "RevisionAfterFailure": 1.0,
    "ValidToolUse": 1.0,
    "StopWhenSufficient": 1.0,
    "EarlySubmit": -1.0,
}


@dataclass
class HMSResult:
    hms_raw: float
    hms_norm: float
    events: dict
    aux: dict
    weights: dict = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Episode helpers
# ─────────────────────────────────────────────────────────────────────────────

def _terminal_step(ep: list[dict]) -> dict:
    for rec in ep:
        if rec.get("terminal"):
            return rec
    return ep[-1]


def _submit_step_index(ep: list[dict]) -> int | None:
    """Return the index of the submit action, or None if never submitted."""
    for i, rec in enumerate(ep):
        if rec.get("action") == "submit":
            return i
    # Else use terminal step index (forced termination)
    return None


def _task_type(ep: list[dict]) -> str:
    return ep[0].get("task_type") or ep[0].get("task", {}).get("task_type", "")


def _task_meta(ep: list[dict]) -> dict:
    return ep[0].get("task_meta", {}) or {}


def _allowed_tools(ep: list[dict]) -> set:
    """Pull task.available_tools from first record (or task_meta)."""
    at = ep[0].get("available_tools") or ep[0].get("task_meta", {}).get("available_tools")
    if at is None:
        return set()
    return set(at)


# ─────────────────────────────────────────────────────────────────────────────
# C.1 CheckBeforeSubmit
# ─────────────────────────────────────────────────────────────────────────────

def check_event_C1(ep: list[dict]) -> dict:
    submit_idx = _submit_step_index(ep)
    available = _allowed_tools(ep)
    # Planning-style task without check_rubric in action set: applicable=False
    if available and "check_rubric" not in available:
        return {"fired": True, "applicable": False,
                "explanation": "check_rubric not in available_tools; skipped"}

    if submit_idx is None:
        return {"fired": False, "applicable": True,
                "explanation": "no submit action found"}

    # Find last check_rubric before submit
    last_check = None
    for i in range(submit_idx):
        if ep[i].get("action") == "check_rubric":
            last_check = i
    if last_check is None:
        return {"fired": False, "applicable": True,
                "explanation": "no check_rubric in trajectory before submit"}

    check_rec = ep[last_check]
    missing_at_check = (check_rec.get("rubric_status") or {}).get("missing_ids") or []
    # If missing was empty at check, allow immediate submit
    if not missing_at_check:
        return {"fired": True, "applicable": True,
                "explanation": f"check_rubric @ step {last_check} found no missing; submit @ {submit_idx}"}
    # Else require at least 1 step gap to fix
    if submit_idx - last_check >= 2:
        return {"fired": True, "applicable": True,
                "explanation": f"check_rubric @ step {last_check} (missing={missing_at_check}) "
                               f"followed by at least 1 step before submit @ {submit_idx}"}
    return {"fired": False, "applicable": True,
            "explanation": f"check_rubric @ step {last_check} found missing={missing_at_check} "
                           f"but submitted next step without fixes"}


# ─────────────────────────────────────────────────────────────────────────────
# C.2 EvidenceBeforeClaim
# ─────────────────────────────────────────────────────────────────────────────

def check_event_C2(ep: list[dict]) -> dict:
    task_type = _task_type(ep)
    if task_type == "coding":
        return {"fired": True, "applicable": False,
                "explanation": "task_type=coding; claims typically empty"}

    submit_idx = _submit_step_index(ep)
    if submit_idx is None:
        submit_idx = len(ep) - 1

    final_draft = ep[submit_idx].get("draft_state") or {}
    claims = final_draft.get("claims") or []
    if not claims:
        return {"fired": True, "applicable": False,
                "explanation": "no factual claims in final draft"}

    claims_with_evidence = final_draft.get("claims_with_evidence") or []
    support_ratio = len(claims_with_evidence) / len(claims)

    # T_first_claim: first step where claims set becomes non-empty
    t_first_claim = None
    prev_n = 0
    for i in range(0, submit_idx + 1):
        cur = (ep[i].get("draft_state") or {}).get("claims") or []
        if len(cur) > 0 and prev_n == 0:
            t_first_claim = i
            break
        prev_n = len(cur)
    if t_first_claim is None:
        return {"fired": False, "applicable": True,
                "explanation": "could not locate first claim step"}

    # Look for evidence_actions in [0, t_first_claim)
    has_prior_evidence = any(
        (ep[i].get("action") in EVIDENCE_ACTIONS
         and (ep[i].get("observation") or {}).get("status") == "success")
        for i in range(0, t_first_claim)
    )
    fired = (support_ratio >= 0.7) and has_prior_evidence
    return {
        "fired": fired,
        "applicable": True,
        "explanation": (
            f"support_ratio={support_ratio:.2f} "
            f"(>=0.7 required), prior_evidence_action={has_prior_evidence}, "
            f"t_first_claim={t_first_claim}"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# C.3 TestBeforeSubmit
# ─────────────────────────────────────────────────────────────────────────────

def check_event_C3(ep: list[dict]) -> dict:
    task_type = _task_type(ep)
    meta = _task_meta(ep)
    applicable = (task_type == "coding") or bool(meta.get("requires_code_execution"))
    if not applicable:
        return {"fired": True, "applicable": False,
                "explanation": "not a coding / code-execution task"}

    submit_idx = _submit_step_index(ep)
    if submit_idx is None:
        return {"fired": False, "applicable": True,
                "explanation": "no submit"}

    # Find last run_tests/run_code with test_results before submit
    last_test_idx = None
    for i in range(submit_idx):
        rec = ep[i]
        if rec.get("action") in TEST_ACTIONS and (rec.get("observation") or {}).get("status") == "success":
            if (rec.get("observation") or {}).get("test_results"):
                last_test_idx = i
    if last_test_idx is None:
        return {"fired": False, "applicable": True,
                "explanation": "no successful run_tests/run_code before submit"}

    test_rec = ep[last_test_idx]
    failed = (test_rec["observation"]["test_results"] or {}).get("failed", 0)
    # If failures observed, require >=1 step gap before submit
    if failed > 0 and submit_idx - last_test_idx < 2:
        return {"fired": False, "applicable": True,
                "explanation": f"last test failed={failed} but submitted next step"}

    # Code hash consistency: harness writes draft_state.code_blob (or .code_hash)
    last_hash = (test_rec.get("draft_state") or {}).get("code_hash") \
                or (test_rec.get("draft_state") or {}).get("code_blob")
    submit_hash = (ep[submit_idx].get("draft_state") or {}).get("code_hash") \
                  or (ep[submit_idx].get("draft_state") or {}).get("code_blob")
    if last_hash is not None and submit_hash is not None and last_hash != submit_hash:
        return {"fired": False, "applicable": True,
                "explanation": "code hash at last test != code hash at submit"}
    return {"fired": True, "applicable": True,
            "explanation": f"run_tests @ step {last_test_idx} (failed={failed}), submit @ {submit_idx}"}


# ─────────────────────────────────────────────────────────────────────────────
# C.4 RevisionAfterFailure
# ─────────────────────────────────────────────────────────────────────────────

def _is_failure(rec: dict) -> bool:
    obs = rec.get("observation") or {}
    if obs.get("status") == "error":
        return True
    tr = obs.get("test_results")
    if tr and tr.get("failed", 0) > 0:
        return True
    if rec.get("action") == "check_rubric":
        if (rec.get("rubric_status") or {}).get("missing_ids"):
            return True
    if rec.get("action") == "verify_solution":
        summ = obs.get("summary") or ""
        if any(k in summ.lower() for k in ("fail", "invalid", "missing")):
            return True
    return False


def check_event_C4(ep: list[dict]) -> dict:
    submit_idx = _submit_step_index(ep)
    end_idx = submit_idx if submit_idx is not None else len(ep) - 1

    failures = [i for i in range(end_idx) if _is_failure(ep[i])]
    if not failures:
        return {"fired": True, "applicable": False,
                "explanation": "no failure signals"}

    # Each failure must be followed by a revise within 3 steps
    unrecovered = []
    for f in failures:
        lo, hi = f + 1, min(end_idx, f + 3)
        recovered = any(ep[t].get("action") in REVISE_ACTIONS for t in range(lo, hi + 1))
        if not recovered:
            unrecovered.append(f)
    fired = len(unrecovered) == 0
    return {
        "fired": fired,
        "applicable": True,
        "explanation": (
            f"{len(failures)} failures; unrecovered@steps={unrecovered}"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# C.5 ValidToolUse
# ─────────────────────────────────────────────────────────────────────────────

def check_event_C5(ep: list[dict]) -> dict:
    submit_idx = _submit_step_index(ep)
    end_idx = submit_idx if submit_idx is not None else len(ep) - 1
    available = _allowed_tools(ep)

    tool_calls = [ep[i] for i in range(end_idx + 1)
                  if ep[i].get("action") in TOOL_CALL_ACTIONS]
    if not tool_calls:
        return {"fired": True, "applicable": False,
                "explanation": "no tool calls in trajectory"}

    valid = 0
    for rec in tool_calls:
        obs = rec.get("observation") or {}
        args_valid = obs.get("tool_args_valid", True)
        ok_status = obs.get("status") != "error"
        # If action=use_tool, also check tool_name ∈ available
        if rec.get("action") == "use_tool":
            tname = obs.get("tool_name") or (rec.get("args") or {}).get("tool_name")
            if available and tname is not None and tname not in available:
                continue
        if args_valid and ok_status:
            valid += 1
    ratio = valid / len(tool_calls)
    return {
        "fired": ratio >= 0.9,
        "applicable": True,
        "explanation": f"valid={valid}/{len(tool_calls)} = {ratio:.2f} (>=0.9 required)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# C.6 StopWhenSufficient
# ─────────────────────────────────────────────────────────────────────────────

def check_event_C6(ep: list[dict]) -> dict:
    available = _allowed_tools(ep)
    if available and "check_rubric" not in available:
        return {"fired": True, "applicable": False,
                "explanation": "check_rubric not in available_tools"}

    term = _terminal_step(ep)
    submit_idx = _submit_step_index(ep)
    reason = term.get("termination_reason")
    if reason != "submit" or submit_idx is None:
        return {"fired": False, "applicable": True,
                "explanation": f"termination_reason={reason} (not active submit)"}

    max_steps = term.get("task_max_steps") or ep[0].get("task_max_steps") or ep[0].get("max_steps")
    # If we don't know max_steps we can't tell if agent left budget on table
    if max_steps is None:
        return {"fired": False, "applicable": True,
                "explanation": "max_steps unknown"}
    if submit_idx >= max_steps - 1:  # bottomed out — no slack to leave
        return {"fired": False, "applicable": True,
                "explanation": f"submit at last available step ({submit_idx}/{max_steps})"}

    # Find last check_rubric
    last_check = None
    for i in range(submit_idx):
        if ep[i].get("action") == "check_rubric":
            last_check = i
    if last_check is None:
        return {"fired": False, "applicable": True,
                "explanation": "no rubric check before submit (cannot assess sufficiency)"}

    cov = (ep[last_check].get("rubric_status") or {}).get("coverage")
    if cov is None:
        return {"fired": False, "applicable": True,
                "explanation": "rubric_status.coverage missing at last check"}
    if cov < 0.85:
        return {"fired": False, "applicable": True,
                "explanation": f"coverage at last check = {cov:.2f} < 0.85"}
    if submit_idx - last_check > 3:
        return {"fired": False, "applicable": True,
                "explanation": f"last check stale ({submit_idx - last_check} steps before submit)"}
    return {"fired": True, "applicable": True,
            "explanation": f"coverage={cov:.2f} @ step {last_check}, submit @ {submit_idx}/{max_steps}"}


# ─────────────────────────────────────────────────────────────────────────────
# C.7 EarlySubmit  (penalty)
# ─────────────────────────────────────────────────────────────────────────────

def check_event_C7(ep: list[dict]) -> dict:
    """C.7 EarlySubmit detector (v4, REVISED 2026-06-11 per reviewer F5).

    v1→v4 change: condition #1's "rushed" threshold tightened from
    0.4*max_steps to 0.25*max_steps. Rationale: many domains have natural
    Base sequences (read→draft→check→submit) consuming 33-42% of the step
    budget; v1 treated these as "rushed" even when rubric deliberation
    occurred. v4 means "rushed" requires consuming less than a quarter of
    the budget, which is genuinely impulsive. All other conditions (2-4)
    unchanged. paper §6 honest documentation of the calibration.
    """
    submit_idx = _submit_step_index(ep)
    if submit_idx is None:
        return {"fired": False, "applicable": True,
                "explanation": "no active submit"}

    term = _terminal_step(ep)
    max_steps = term.get("task_max_steps") or ep[0].get("task_max_steps") or ep[0].get("max_steps")
    if max_steps is None:
        max_steps = 10
    final_score = term.get("final_rubric_score") or {}
    norm = final_score.get("rubric_score_norm")
    missing = final_score.get("missing_items") or []
    n_criteria = term.get("n_criteria") or ep[0].get("n_criteria")

    # 1) low score + early submit (REVISED v4: 0.25 instead of 0.4)
    threshold = float(ep[0].get("_c7_rushed_threshold", 0.25))
    if norm is not None and norm < 0.5 and submit_idx <= threshold * max_steps:
        return {"fired": True, "applicable": True,
                "explanation": f"rubric_norm={norm:.2f}<0.5 and submit@{submit_idx}<=0.25*{max_steps}"}

    # 2) majority of criteria missing AND submit before max_steps
    if n_criteria and len(missing) >= 0.5 * n_criteria and submit_idx < max_steps:
        return {"fired": True, "applicable": True,
                "explanation": f"missing={len(missing)}>=0.5*{n_criteria} and submit@{submit_idx}<{max_steps}"}

    # 3) coding: test pass rate <50% and submit before max_steps
    task_type = _task_type(ep)
    if task_type == "coding" and submit_idx < max_steps:
        # find last run_tests
        last_test = None
        for i in range(submit_idx):
            rec = ep[i]
            if rec.get("action") in TEST_ACTIONS and (rec.get("observation") or {}).get("test_results"):
                last_test = rec
        if last_test:
            tr = last_test["observation"]["test_results"]
            passed = tr.get("passed", 0)
            failed = tr.get("failed", 0)
            tot = passed + failed
            if tot > 0 and (passed / tot) < 0.5:
                return {"fired": True, "applicable": True,
                        "explanation": f"coding test pass={passed}/{tot}<0.5 and submit@{submit_idx}<{max_steps}"}

    # 4) evidence required but never gathered, with non-empty claims
    meta = _task_meta(ep)
    if meta.get("requires_evidence"):
        gathered = any(
            (ep[i].get("action") in EVIDENCE_ACTIONS
             and (ep[i].get("observation") or {}).get("status") == "success")
            for i in range(submit_idx + 1)
        )
        final_claims = (ep[submit_idx].get("draft_state") or {}).get("claims") or []
        if (not gathered) and final_claims:
            return {"fired": True, "applicable": True,
                    "explanation": "requires_evidence=True, no evidence action, claims non-empty"}

    return {"fired": False, "applicable": True,
            "explanation": "no early-submit condition triggered"}


# ─────────────────────────────────────────────────────────────────────────────
# Aux: RepeatedUselessAction + CostOverrun
# ─────────────────────────────────────────────────────────────────────────────

def _canonical(args) -> str:
    """Stable string repr for grouping repeated actions."""
    try:
        import json as _j
        return _j.dumps(args, sort_keys=True, default=str)
    except Exception:
        return str(args)


def _fact_set(rec: dict) -> set:
    obs = rec.get("observation") or {}
    facts = obs.get("extracted_facts") or []
    return set(facts)


def compute_aux(ep: list[dict]) -> dict:
    n_useless_runs = 0
    n_useless_steps = 0
    prev_key: str | None = None
    prev_facts: set | None = None
    run_len = 0
    for rec in ep:
        key = f"{rec.get('action')}::{_canonical(rec.get('args'))}"
        facts = _fact_set(rec)
        if key == prev_key:
            if prev_facts is not None and facts == prev_facts:
                run_len += 1
                n_useless_steps += 1
                if run_len == 1:
                    n_useless_runs += 1
                continue
        prev_key = key
        prev_facts = facts
        run_len = 0

    total_cost = sum((r.get("observation") or {}).get("cost", 0.0) or 0.0 for r in ep)
    budget = ep[0].get("cost_budget") or _terminal_step(ep).get("task_cost_budget")
    overrun = bool(budget and total_cost > budget * 1.5)
    return {
        "n_useless_runs": n_useless_runs,
        "n_useless_steps": n_useless_steps,
        "total_cost": total_cost,
        "cost_budget": budget,
        "cost_overrun": overrun,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregator
# ─────────────────────────────────────────────────────────────────────────────

EVENT_CHECKERS = {
    "CheckBeforeSubmit": check_event_C1,
    "EvidenceBeforeClaim": check_event_C2,
    "TestBeforeSubmit": check_event_C3,
    "RevisionAfterFailure": check_event_C4,
    "ValidToolUse": check_event_C5,
    "StopWhenSufficient": check_event_C6,
    "EarlySubmit": check_event_C7,
}


def compute_hms(ep: list[dict], weights: dict | None = None,
                c7_rushed_threshold: float = 0.25) -> HMSResult:
    w = DEFAULT_WEIGHTS.copy()
    if weights:
        w.update(weights)

    # Inject c7 threshold into the episode metadata so check_event_C7 sees it
    if ep and isinstance(ep[0], dict):
        ep[0]["_c7_rushed_threshold"] = c7_rushed_threshold
    events = {name: fn(ep) for name, fn in EVENT_CHECKERS.items()}
    applicable = {k: v for k, v in events.items() if v["applicable"]}
    hms_raw = sum(w[k] * (1.0 if v["fired"] else 0.0) for k, v in applicable.items())
    denom = sum(abs(w[k]) for k in applicable) or 1.0
    hms_norm = hms_raw / denom
    return HMSResult(
        hms_raw=hms_raw,
        hms_norm=hms_norm,
        events=events,
        aux=compute_aux(ep),
        weights=w,
    )
