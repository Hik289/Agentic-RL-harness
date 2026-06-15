"""Anchor 2: Base Harness end-to-end on one task per domain.

Invariants:
  * each trajectory has full B.1 fields (step, action, args, observation,
    draft_state, rubric_status, terminal+termination_reason)
  * submit_rate over 6 episodes > 80%  (= at least 5/6 reach submit)
  * trajectories pass into hms_detector + score_trajectory without crashing
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1]))  # code/

from harness.actions import Task
from harness.agent import run_episode
from harness.util.llm_client import LLMClient
from modules.hms_detector import compute_hms


DOMAINS = [
    ("knowledge_work", "knowledge_work_000"),
    ("coding", "coding_000"),
    ("research", "research_000"),
    ("multi_tool", "multi_tool_000"),
    ("long_memory", "long_memory_000"),
    ("planning", "planning_000"),
]


REQUIRED_FIELDS = ["task_id", "episode_id", "step", "action", "args",
                   "observation", "draft_state", "rubric_status", "timestamp_ms"]


def _validate_record(rec: dict) -> list[str]:
    missing = [k for k in REQUIRED_FIELDS if k not in rec]
    obs = rec.get("observation") or {}
    for k in ("status", "summary", "cost", "duration_ms"):
        if k not in obs:
            missing.append(f"observation.{k}")
    ds = rec.get("draft_state") or {}
    for k in ("has_draft", "draft_len_chars", "claims", "claims_with_evidence"):
        if k not in ds:
            missing.append(f"draft_state.{k}")
    return missing


def main():
    base = Path(os.environ.get("AGENTICRLHARNESS_DATA",
                "./data"))
    task_root = base / "synthetic_tasks"
    if not task_root.exists():
        # fall back to hpc path
        task_root = (Path(os.environ.get("AGENTICRLHARNESS_DATA", "./data")) / "/synthetic_tasks".lstrip("/"))
    print(f"[anchor_2] task_root = {task_root}")

    client = LLMClient()
    out_dir = THIS.parent / "trajectories"
    out_dir.mkdir(exist_ok=True)

    summary = []
    n_submit = 0
    field_failures = 0
    t0 = time.monotonic()
    for domain, task_id in DOMAINS:
        task_dir = task_root / domain / task_id
        if not task_dir.exists():
            print(f"  SKIP {domain}: {task_dir} missing")
            continue
        task = Task.load(task_dir)
        print(f"  [{domain}] running {task.task_id} (max_steps={task.max_steps}) ...")
        try:
            logger, scored = run_episode(task, client=client)
        except Exception as e:
            print(f"    ✗ episode crashed: {type(e).__name__}: {e}")
            summary.append({"domain": domain, "task_id": task.task_id,
                            "ok": False, "error": str(e)})
            continue

        # Validate per-record completeness
        missing_fields = []
        for rec in logger.records:
            m = _validate_record(rec)
            if m:
                missing_fields.append((rec.get("step"), m))
        terminal = logger.records[-1] if logger.records else None
        terminated_with_submit = (terminal.get("termination_reason") == "submit"
                                  if terminal else False)
        if terminated_with_submit:
            n_submit += 1

        # Run HMS detector on the trajectory
        try:
            hms = compute_hms(logger.records)
            hms_ok = True
            hms_norm = hms.hms_norm
            events_summary = {k: {"fired": v["fired"],
                                  "applicable": v["applicable"]}
                              for k, v in hms.events.items()}
        except Exception as e:
            hms_ok = False
            hms_norm = None
            events_summary = {"error": str(e)}

        # Save trajectory jsonl
        traj_path = out_dir / f"{domain}_{task_id}.jsonl"
        logger.to_jsonl(traj_path)

        ep_blob = {
            "domain": domain, "task_id": task.task_id,
            "n_steps": len(logger.records),
            "max_steps": task.max_steps,
            "termination_reason": terminal.get("termination_reason"),
            "submitted": terminated_with_submit,
            "total_cost_usd_llm": logger.total_cost,
            "rubric_score_norm": scored["rubric_score_norm"],
            "R_total": scored["R_total_breakdown"]["R_total"],
            "P_error": scored["P_error"],
            "P_cost": scored["P_cost"],
            "P_early": scored["P_early"],
            "missing_record_fields": missing_fields,
            "hms_ok": hms_ok,
            "hms_norm": hms_norm,
            "hms_events": events_summary,
            "traj_jsonl": str(traj_path),
        }
        summary.append(ep_blob)
        if missing_fields:
            field_failures += 1

        status_icon = "✓" if terminated_with_submit and not missing_fields else ("△" if terminated_with_submit else "✗")
        print(f"    {status_icon} steps={ep_blob['n_steps']} reason={ep_blob['termination_reason']}"
              f"  rubric_norm={scored['rubric_score_norm']:.3f}"
              f"  R_total={ep_blob['R_total']:.3f}"
              f"  cost=${logger.total_cost:.4f}"
              f"  hms_norm={hms_norm if hms_norm is None else round(hms_norm,3)}"
              f"  missing_fields={len(missing_fields)}")

    elapsed = time.monotonic() - t0
    submit_rate = n_submit / len(DOMAINS) if DOMAINS else 0.0
    total_cost = sum(s["total_cost_usd_llm"] for s in summary if "total_cost_usd_llm" in s)
    overall = {
        "timestamp_jst": time.strftime("%Y-%m-%dT%H:%M:%S+09:00", time.localtime()),
        "n_episodes": len(summary),
        "n_submitted": n_submit,
        "submit_rate": submit_rate,
        "field_failures": field_failures,
        "total_cost_usd": total_cost,
        "elapsed_s": elapsed,
        "episodes": summary,
        "pass_submit_rate_gt_0p8": submit_rate > 0.8,
        "pass_no_field_failures": field_failures == 0,
        "overall_ok": (submit_rate > 0.8) and field_failures == 0,
    }
    out_path = THIS.parent / "anchor_2_results.json"
    out_path.write_text(json.dumps(overall, indent=2, default=str))
    print(f"\n[anchor_2] submit_rate={submit_rate:.0%} ({n_submit}/{len(DOMAINS)})  "
          f"field_failures={field_failures}  total_cost=${total_cost:.4f}  "
          f"elapsed={elapsed:.1f}s  -> {out_path}")
    return 0 if overall["overall_ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
