"""Trajectory logger — emits records per data/rubric_design_spec.md §B.1.

Each step record:
{
  "task_id": ...,
  "episode_id": ...,
  "step": int,
  "action": str,
  "args": dict,
  "observation": {status, summary, extracted_facts?, test_results?, tool_name?,
                  tool_args_valid?, cost, duration_ms},
  "draft_state": {has_draft, draft_len_chars, claims, claims_with_evidence,
                  code_blob?, code_hash?},
  "rubric_status": {last_checked_step, coverage, missing_ids},
  "timestamp_ms": int
}

Terminal record additionally has:
  terminal: true, termination_reason: submit|max_steps|budget_exceeded|error,
  final_rubric_score (filled by judge post-hoc),
  task_max_steps, n_criteria (convenience for hms_detector).
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _hash(blob: str) -> str:
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


@dataclass
class TrajectoryLogger:
    task_id: str
    task_type: str
    available_tools: list
    max_steps: int
    cost_budget: float
    n_criteria: int = 0
    episode_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    records: list = field(default_factory=list)
    _t0: float = field(default_factory=time.monotonic)

    # Persistent state across steps
    draft_state: dict = field(default_factory=lambda: {
        "has_draft": False, "draft_len_chars": 0,
        "claims": [], "claims_with_evidence": [],
        "code_blob": None, "code_hash": None,
    })
    rubric_status: dict = field(default_factory=lambda: {
        "last_checked_step": None, "coverage": None, "missing_ids": None,
    })

    # Per-episode aggregates
    total_cost: float = 0.0

    def log_step(
        self,
        action: str,
        *,
        args: dict | None = None,
        status: str = "success",
        summary: str = "",
        extracted_facts: list | None = None,
        test_results: dict | None = None,
        tool_name: str | None = None,
        tool_args_valid: bool = True,
        cost: float = 0.0,
        duration_ms: int | None = None,
        draft_update: dict | None = None,
        rubric_update: dict | None = None,
    ) -> dict:
        if draft_update:
            self.draft_state.update(draft_update)
            if "code_blob" in draft_update and draft_update["code_blob"] is not None:
                self.draft_state["code_hash"] = _hash(draft_update["code_blob"])
            self.draft_state["draft_len_chars"] = len(
                self.draft_state.get("code_blob") or ""
            ) + len(self._draft_text(draft_update))
        if rubric_update:
            self.rubric_status.update(rubric_update)
            self.rubric_status["last_checked_step"] = len(self.records)
        self.total_cost += float(cost or 0.0)

        rec = {
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "step": len(self.records),
            "action": action,
            "args": args or {},
            "observation": {
                "status": status,
                "summary": summary,
                "extracted_facts": extracted_facts or [],
                "test_results": test_results,
                "tool_name": tool_name,
                "tool_args_valid": tool_args_valid,
                "cost": float(cost or 0.0),
                "duration_ms": duration_ms or int((time.monotonic() - self._t0) * 1000),
            },
            "draft_state": dict(self.draft_state),
            "rubric_status": dict(self.rubric_status),
            "timestamp_ms": int(time.time() * 1000),
            # Convenience denorm for downstream tools:
            "task_type": self.task_type,
            "available_tools": self.available_tools,
            "task_max_steps": self.max_steps,
            "max_steps": self.max_steps,
            "cost_budget": self.cost_budget,
            "n_criteria": self.n_criteria,
        }
        self.records.append(rec)
        return rec

    def finalize(self, *, termination_reason: str,
                 final_rubric_score: dict | None = None) -> dict:
        # mark last record as terminal
        if not self.records:
            raise RuntimeError("cannot finalize an empty trajectory")
        last = self.records[-1]
        last["terminal"] = True
        last["termination_reason"] = termination_reason
        if final_rubric_score is not None:
            last["final_rubric_score"] = final_rubric_score
        return last

    @staticmethod
    def _draft_text(draft_update: dict) -> str:
        # Heuristic chars for non-code drafts
        text = draft_update.get("draft_text") or ""
        if isinstance(text, str):
            return text
        return ""

    def to_jsonl(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for rec in self.records:
                f.write(json.dumps(rec, default=str) + "\n")
        return path

    def as_list(self) -> list:
        return list(self.records)
