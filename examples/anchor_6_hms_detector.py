"""Anchor 6: HMS detector on 5 hand-constructed trajectories.

Invariant: 5 trajectory × 7 metric = 35 individual judgments all correct.

Each trajectory exercises a specific behavior:
  ep_1_clean_submit          — all positive events fire, EarlySubmit not
  ep_2_no_check_no_test      — coding task; never checked rubric, never ran tests
  ep_3_claim_without_evidence — knowledge task; claims but no prior evidence
  ep_4_failure_no_revise     — coding task; saw test failures, no revise
  ep_5_planning_skip_check   — planning task without check_rubric (skipped events)

For each ep we list the EXPECTED (fired, applicable) per event. The script
asserts the detector matches all 35 judgments.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.hms_detector import compute_hms, EVENT_CHECKERS


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory builder helpers
# ──────────────────────────────────────────────────────────────────────────────

def _step(action, *, args=None, status="success", summary="", facts=None,
          test_results=None, draft=None, rubric_status=None,
          tool_name=None, tool_args_valid=True, cost=0.01,
          terminal=False, termination_reason=None,
          final_rubric_score=None, task_max_steps=None,
          n_criteria=None, **extra):
    rec = {
        "action": action,
        "args": args or {},
        "observation": {
            "status": status, "summary": summary,
            "extracted_facts": facts or [],
            "test_results": test_results,
            "tool_name": tool_name,
            "tool_args_valid": tool_args_valid,
            "cost": cost,
            "duration_ms": 100,
        },
        "draft_state": draft or {"has_draft": False, "draft_len_chars": 0,
                                 "claims": [], "claims_with_evidence": []},
        "rubric_status": rubric_status or {"last_checked_step": None,
                                            "coverage": None, "missing_ids": None},
        "timestamp_ms": 0,
    }
    if terminal:
        rec["terminal"] = True
        rec["termination_reason"] = termination_reason
        if final_rubric_score is not None:
            rec["final_rubric_score"] = final_rubric_score
        if task_max_steps is not None:
            rec["task_max_steps"] = task_max_steps
        if n_criteria is not None:
            rec["n_criteria"] = n_criteria
    rec.update(extra)
    return rec


def _set_meta(ep, task_type, available_tools, max_steps=10, cost_budget=1.0,
              requires_evidence=False, requires_code_execution=False,
              n_criteria=None):
    for r in ep:
        r["task_type"] = task_type
        r["available_tools"] = list(available_tools)
        r["task_max_steps"] = max_steps
        r["max_steps"] = max_steps
        r["cost_budget"] = cost_budget
        r["task_meta"] = {
            "requires_evidence": requires_evidence,
            "requires_code_execution": requires_code_execution,
        }
        if n_criteria is not None:
            r["n_criteria"] = n_criteria
    return ep


# ──────────────────────────────────────────────────────────────────────────────
# Episode 1: clean submit (knowledge_work)
#   Events: C.1=True, C.2=True, C.3=N/A, C.4=N/A, C.5=True, C.6=True, C.7=False
# ──────────────────────────────────────────────────────────────────────────────

def build_ep1_clean_submit():
    AVAIL = ["read_input", "search", "draft_solution", "check_rubric",
             "verify_solution", "revise_solution", "submit", "use_tool"]
    ep = [
        # 0: read_input (evidence)
        _step("read_input", facts=["fact_A", "fact_B"], cost=0.02),
        # 1: search (evidence)
        _step("search", facts=["fact_C"], cost=0.03),
        # 2: draft with claims supported
        _step("draft_solution",
              draft={"has_draft": True, "draft_len_chars": 500,
                     "claims": ["claim_A", "claim_B"],
                     "claims_with_evidence": [0, 1]}),
        # 3: check_rubric → coverage 0.9, missing []
        _step("check_rubric",
              rubric_status={"last_checked_step": 3, "coverage": 0.9,
                              "missing_ids": []}),
        # 4: submit
        _step("submit",
              draft={"has_draft": True, "draft_len_chars": 500,
                     "claims": ["claim_A", "claim_B"],
                     "claims_with_evidence": [0, 1]},
              rubric_status={"last_checked_step": 3, "coverage": 0.9,
                              "missing_ids": []},
              terminal=True, termination_reason="submit",
              final_rubric_score={"rubric_score_norm": 0.92,
                                   "missing_items": []},
              task_max_steps=10, n_criteria=4),
    ]
    return _set_meta(ep, "knowledge_work", AVAIL,
                     requires_evidence=True, n_criteria=4)


EP1_EXPECT = {
    "CheckBeforeSubmit":    {"fired": True,  "applicable": True},
    "EvidenceBeforeClaim":  {"fired": True,  "applicable": True},
    "TestBeforeSubmit":     {"fired": True,  "applicable": False},
    "RevisionAfterFailure": {"fired": True,  "applicable": False},
    "ValidToolUse":         {"fired": True,  "applicable": True},   # 1 search tool call, valid
    "StopWhenSufficient":   {"fired": True,  "applicable": True},
    "EarlySubmit":          {"fired": False, "applicable": True},
}


# ──────────────────────────────────────────────────────────────────────────────
# Episode 2: no_check_no_test (coding task, never check_rubric, never run_tests)
#   C.1: fired=False (no check)
#   C.2: applicable=False (coding)
#   C.3: fired=False (no test before submit)
#   C.4: applicable=False (no failure signals)
#   C.5: applicable=False (no tool calls — just write_code/submit)
#   C.6: fired=False (no check)
#   C.7: fired=True (coding test pass <50% trivially; here no test run, but
#        criterion 4 applies: requires_evidence is False so #4 N/A; #1 needs
#        norm<0.5 and t_submit<=0.4*max_steps. We make norm=0.3 and submit@3
#        with max_steps=10 → satisfies)
# ──────────────────────────────────────────────────────────────────────────────

def build_ep2_no_check_no_test():
    AVAIL = ["read_problem", "write_code", "run_tests", "revise_code",
             "check_rubric", "submit"]
    ep = [
        _step("read_problem", cost=0.01,
              draft={"has_draft": False, "draft_len_chars": 0,
                     "claims": [], "claims_with_evidence": [],
                     "code_blob": ""}),
        _step("write_code",
              draft={"has_draft": True, "draft_len_chars": 300,
                     "claims": [], "claims_with_evidence": [],
                     "code_blob": "def f(): return 0"}),
        _step("submit",
              draft={"has_draft": True, "draft_len_chars": 300,
                     "claims": [], "claims_with_evidence": [],
                     "code_blob": "def f(): return 0"},
              terminal=True, termination_reason="submit",
              final_rubric_score={"rubric_score_norm": 0.30,
                                   "missing_items": ["c1", "c3"]},
              task_max_steps=10, n_criteria=3),
    ]
    return _set_meta(ep, "coding", AVAIL,
                     requires_code_execution=True, n_criteria=3)


EP2_EXPECT = {
    "CheckBeforeSubmit":    {"fired": False, "applicable": True},
    "EvidenceBeforeClaim":  {"fired": True,  "applicable": False},
    "TestBeforeSubmit":     {"fired": False, "applicable": True},
    "RevisionAfterFailure": {"fired": True,  "applicable": False},
    "ValidToolUse":         {"fired": True,  "applicable": False},
    "StopWhenSufficient":   {"fired": False, "applicable": True},
    "EarlySubmit":          {"fired": True,  "applicable": True},
}


# ──────────────────────────────────────────────────────────────────────────────
# Episode 3: claim_without_evidence (knowledge_work; draft has claims; no prior
# evidence action successful)
#   C.1: fired=False — no check_rubric
#   C.2: fired=False — claims but no prior evidence
#   C.3: applicable=False (knowledge)
#   C.4: applicable=False (no failures)
#   C.5: applicable=False (no tool calls)
#   C.6: fired=False (no check)
#   C.7: fired=True (#4: requires_evidence + no evidence action + claims non-empty)
# ──────────────────────────────────────────────────────────────────────────────

def build_ep3_claim_without_evidence():
    AVAIL = ["read_input", "search", "draft_solution", "check_rubric",
             "verify_solution", "submit"]
    ep = [
        _step("draft_solution",
              draft={"has_draft": True, "draft_len_chars": 200,
                     "claims": ["X", "Y"],
                     "claims_with_evidence": []}),
        _step("submit",
              draft={"has_draft": True, "draft_len_chars": 200,
                     "claims": ["X", "Y"],
                     "claims_with_evidence": []},
              terminal=True, termination_reason="submit",
              final_rubric_score={"rubric_score_norm": 0.55,
                                   "missing_items": ["c2"]},
              task_max_steps=10, n_criteria=4),
    ]
    return _set_meta(ep, "knowledge_work", AVAIL,
                     requires_evidence=True, n_criteria=4)


EP3_EXPECT = {
    "CheckBeforeSubmit":    {"fired": False, "applicable": True},
    "EvidenceBeforeClaim":  {"fired": False, "applicable": True},
    "TestBeforeSubmit":     {"fired": True,  "applicable": False},
    "RevisionAfterFailure": {"fired": True,  "applicable": False},
    "ValidToolUse":         {"fired": True,  "applicable": False},
    "StopWhenSufficient":   {"fired": False, "applicable": True},
    "EarlySubmit":          {"fired": True,  "applicable": True},
}


# ──────────────────────────────────────────────────────────────────────────────
# Episode 4: failure_no_revise (coding test failures, no revise action)
#   C.1: fired=False (no check)
#   C.2: applicable=False
#   C.3: fired=False (last test failed=2, submitted next step)
#   C.4: fired=False (failure not revised)
#   C.5: applicable=False (no use_tool/search; run_tests we treat as tool call)
#        Actually run_tests IS in TOOL_CALL_ACTIONS via run_code... but ep uses
#        "run_tests" which is NOT in TOOL_CALL_ACTIONS. So applicable=False.
#   C.6: fired=False (no check)
#   C.7: fired=True (coding test pass<50% before submit @ t_submit<max_steps)
# ──────────────────────────────────────────────────────────────────────────────

def build_ep4_failure_no_revise():
    AVAIL = ["read_problem", "write_code", "run_tests", "revise_code",
             "check_rubric", "submit"]
    ep = [
        _step("read_problem", cost=0.01),
        _step("write_code",
              draft={"has_draft": True, "draft_len_chars": 150,
                     "code_blob": "buggy"}),
        _step("run_tests",
              test_results={"passed": 1, "failed": 3, "errors": []},
              draft={"has_draft": True, "draft_len_chars": 150,
                     "code_blob": "buggy"}),
        _step("submit",
              draft={"has_draft": True, "draft_len_chars": 150,
                     "code_blob": "buggy"},
              terminal=True, termination_reason="submit",
              final_rubric_score={"rubric_score_norm": 0.40,
                                   "missing_items": ["c1"]},
              task_max_steps=10, n_criteria=3),
    ]
    return _set_meta(ep, "coding", AVAIL,
                     requires_code_execution=True, n_criteria=3)


EP4_EXPECT = {
    "CheckBeforeSubmit":    {"fired": False, "applicable": True},
    "EvidenceBeforeClaim":  {"fired": True,  "applicable": False},
    "TestBeforeSubmit":     {"fired": False, "applicable": True},
    "RevisionAfterFailure": {"fired": False, "applicable": True},
    "ValidToolUse":         {"fired": True,  "applicable": False},
    "StopWhenSufficient":   {"fired": False, "applicable": True},
    "EarlySubmit":          {"fired": True,  "applicable": True},
}


# ──────────────────────────────────────────────────────────────────────────────
# Episode 5: planning_skip_check (planning task without check_rubric in actions)
#   C.1: applicable=False (no check_rubric in action set)
#   C.2: applicable=False (no claims)
#   C.3: applicable=False
#   C.4: applicable=False
#   C.5: applicable=True (use_tool calls valid)
#   C.6: applicable=False (no check_rubric)
#   C.7: fired=False (good rubric, late submit)
# ──────────────────────────────────────────────────────────────────────────────

def build_ep5_planning():
    AVAIL = ["observe", "plan", "use_tool", "execute_action", "submit"]
    ep = [
        _step("observe", cost=0.01),
        _step("plan", draft={"has_draft": True, "draft_len_chars": 100,
                              "claims": []}),
        _step("use_tool", tool_name="execute_action", cost=0.05,
              args={"tool_name": "execute_action"},
              tool_args_valid=True),
        _step("use_tool", tool_name="execute_action", cost=0.05,
              args={"tool_name": "execute_action"},
              tool_args_valid=True),
        _step("submit",
              terminal=True, termination_reason="submit",
              final_rubric_score={"rubric_score_norm": 0.80,
                                   "missing_items": []},
              task_max_steps=10, n_criteria=3),
    ]
    return _set_meta(ep, "planning", AVAIL, n_criteria=3)


EP5_EXPECT = {
    "CheckBeforeSubmit":    {"fired": True,  "applicable": False},
    "EvidenceBeforeClaim":  {"fired": True,  "applicable": False},
    "TestBeforeSubmit":     {"fired": True,  "applicable": False},
    "RevisionAfterFailure": {"fired": True,  "applicable": False},
    "ValidToolUse":         {"fired": True,  "applicable": True},
    "StopWhenSufficient":   {"fired": True,  "applicable": False},
    "EarlySubmit":          {"fired": False, "applicable": True},
}


CASES = [
    ("ep1_clean_submit", build_ep1_clean_submit, EP1_EXPECT),
    ("ep2_no_check_no_test", build_ep2_no_check_no_test, EP2_EXPECT),
    ("ep3_claim_without_evidence", build_ep3_claim_without_evidence, EP3_EXPECT),
    ("ep4_failure_no_revise", build_ep4_failure_no_revise, EP4_EXPECT),
    ("ep5_planning_skip_check", build_ep5_planning, EP5_EXPECT),
]


def main():
    out_results = []
    n_pass, n_total = 0, 0
    for label, build, expect in CASES:
        ep = build()
        result = compute_hms(ep)
        ev = result.events
        per_event_match = {}
        ep_pass = 0
        ep_total = 0
        for name in EVENT_CHECKERS:
            got = ev[name]
            exp = expect[name]
            match = (got["fired"] == exp["fired"]) and (got["applicable"] == exp["applicable"])
            per_event_match[name] = {
                "expected": exp, "got": {"fired": got["fired"], "applicable": got["applicable"]},
                "explanation": got["explanation"], "match": match,
            }
            ep_total += 1
            ep_pass += int(match)
        out_results.append({
            "label": label,
            "per_event": per_event_match,
            "ep_pass": ep_pass,
            "ep_total": ep_total,
            "hms_raw": result.hms_raw,
            "hms_norm": result.hms_norm,
            "aux": result.aux,
        })
        n_pass += ep_pass
        n_total += ep_total
        print(f"{label}: {ep_pass}/{ep_total} events match  hms_norm={result.hms_norm:.3f}")
        for name, m in per_event_match.items():
            if not m["match"]:
                print(f"  ✗ {name}: expected={m['expected']} got={m['got']} :: {m['explanation']}")

    blob = {
        "timestamp_jst": time.strftime("%Y-%m-%dT%H:%M:%S+09:00", time.localtime()),
        "n_cases": len(CASES),
        "n_pass": n_pass,
        "n_total": n_total,
        "overall_ok": n_pass == n_total,
        "results": out_results,
    }
    out_path = Path(__file__).resolve().parent / "anchor_6_results.json"
    out_path.write_text(json.dumps(blob, indent=2, default=str))
    print(f"\n[anchor_6] {n_pass}/{n_total} judgments match  →  {out_path}")
    return 0 if n_pass == n_total else 2


if __name__ == "__main__":
    sys.exit(main())
