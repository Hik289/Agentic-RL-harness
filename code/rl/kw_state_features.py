"""State feature extraction for the KW (knowledge_work_deliverable) domain.

KW action space (per task.json):
  read_material, extract_table, summarize_material, compare_evidence,
  draft_deliverable, verify_evidence, check_rubric, submit
"""
from __future__ import annotations

KW_ACTION_SPACE = [
    "read_material",
    "extract_table",
    "summarize_material",
    "compare_evidence",
    "draft_deliverable",
    "verify_evidence",
    "check_rubric",
    "submit",
]

ACTION_TO_IDX = {a: i for i, a in enumerate(KW_ACTION_SPACE)}
STATE_DIM = 10 + len(KW_ACTION_SPACE)  # 18, same as coding


def featurize_state(*, step: int, max_steps: int,
                    cost_so_far: float, cost_budget: float,
                    draft_state: dict, rubric_status: dict,
                    last_action: str | None, error_count: int,
                    n_facts_collected: int,
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
    facts_norm = min(n_facts_collected / 10.0, 1.0)
    # one extra dim: claims_with_evidence ratio
    claims = draft_state.get("claims") or []
    cwe = draft_state.get("claims_with_evidence") or []
    cwe_ratio = (len(cwe) / max(len(claims), 1)) if claims else 0.0

    numeric = [step_norm, has_draft, draft_len_norm, rubric_coverage,
               rubric_missing_norm, error_count_norm, cost_norm, rem_norm,
               facts_norm, cwe_ratio]
    one_hot = [0.0] * len(KW_ACTION_SPACE)
    if last_action in ACTION_TO_IDX:
        one_hot[ACTION_TO_IDX[last_action]] = 1.0
    return numeric + one_hot


def trajectory_to_features(records: list, n_criteria: int) -> list[tuple[list[float], int]]:
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
        if action not in ACTION_TO_IDX:
            continue
        feats = featurize_state(
            step=t, max_steps=max_steps,
            cost_so_far=cost_so_far, cost_budget=cost_budget,
            draft_state=draft, rubric_status=rubric,
            last_action=last_action, error_count=error_count,
            n_facts_collected=n_facts, n_criteria=n_criteria,
        )
        pairs.append((feats, ACTION_TO_IDX[action]))
        obs = rec.get("observation") or {}
        ds = rec.get("draft_state") or {}
        rs = rec.get("rubric_status") or {}
        draft = ds
        rubric = rs
        cost_so_far += float(obs.get("cost", 0.0) or 0.0)
        if obs.get("status") == "error":
            error_count += 1
        n_facts += len(obs.get("extracted_facts") or [])
        last_action = action
    return pairs
