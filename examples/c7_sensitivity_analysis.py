"""EarlySubmit (C.7) threshold sensitivity analysis.

For each domain that has eval_records_{base,aw}.jsonl, re-score HMS_norm
at three rushed-thresholds (0.25, 0.30, 0.35). Report:
  - ΔHMS per (domain, threshold)
  - EarlySubmit fired-rate per (domain, threshold, policy)

This is a pure re-scoring pass: no new LLM calls. Tests the C.7 spec
design space, not whether AW "wins" — we report all numbers honestly.
"""
import os
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1]))

from modules.hms_detector import compute_hms

ROOT = Path(os.environ.get("AGENTICRLHARNESS_RESULTS", "./results"))
THRESHOLDS = [0.25, 0.30, 0.35]

# Source per-domain mapping: (subdir_name, output_label)
DOMAIN_SOURCES = [
    ("main_knowledge_work_v2", "knowledge_work"),
    ("main_coding_v2", "coding"),
    ("main_research_v2", "research"),
    ("main_multi_tool_v2", "multi_tool"),
    ("main_long_memory_v2", "long_memory"),
    ("main_planning_v2", "planning"),
]


def score_records(records_jsonl_path: Path, threshold: float) -> tuple[float, float, dict, list[float]]:
    """Return (hms_mean, hms_std, per-event-fired-counts, per-episode hms list)."""
    norms = []
    per_event = {}
    n_episodes = 0
    with records_jsonl_path.open() as f:
        for line in f:
            entry = json.loads(line)
            recs = entry.get("records") or []
            if not recs:
                continue
            n_episodes += 1
            res = compute_hms(recs, c7_rushed_threshold=threshold)
            norms.append(res.hms_norm)
            for name, ev in res.events.items():
                slot = per_event.setdefault(name, {"fired": 0, "applicable": 0, "total": 0})
                slot["total"] += 1
                if ev["applicable"]:
                    slot["applicable"] += 1
                    if ev["fired"]:
                        slot["fired"] += 1
    if not norms:
        return None, None, per_event, []
    mean = statistics.fmean(norms)
    std = statistics.stdev(norms) if len(norms) > 1 else 0.0
    return mean, std, per_event, norms


def main():
    summary = {"thresholds": THRESHOLDS, "by_domain": {}}
    for subdir, label in DOMAIN_SOURCES:
        base_path = ROOT / subdir / "eval_records_base.jsonl"
        aw_path = ROOT / subdir / "eval_records_aw.jsonl"
        if not base_path.exists() or not aw_path.exists():
            print(f"SKIP {label}: missing eval_records_{{base,aw}}.jsonl in {subdir}")
            summary["by_domain"][label] = {"status": "missing"}
            continue
        dom_out = {"source_dir": subdir, "thresholds": {}}
        for thr in THRESHOLDS:
            b_mean, b_std, b_events, _ = score_records(base_path, thr)
            a_mean, a_std, a_events, _ = score_records(aw_path, thr)
            d_hms = (a_mean - b_mean) if (a_mean is not None and b_mean is not None) else None
            # ES fired rates
            b_es = b_events.get("EarlySubmit", {})
            a_es = a_events.get("EarlySubmit", {})
            b_es_rate = b_es["fired"] / b_es["applicable"] if b_es.get("applicable", 0) > 0 else None
            a_es_rate = a_es["fired"] / a_es["applicable"] if a_es.get("applicable", 0) > 0 else None
            dom_out["thresholds"][str(thr)] = {
                "base_hms_mean": b_mean, "base_hms_std": b_std,
                "aw_hms_mean": a_mean, "aw_hms_std": a_std,
                "delta_hms": d_hms,
                "base_es_fired": b_es.get("fired"), "base_es_app": b_es.get("applicable"),
                "base_es_rate": b_es_rate,
                "aw_es_fired": a_es.get("fired"), "aw_es_app": a_es.get("applicable"),
                "aw_es_rate": a_es_rate,
                "base_per_event_summary": {k: {"fired": v["fired"], "applicable": v["applicable"]}
                                           for k, v in b_events.items()},
                "aw_per_event_summary": {k: {"fired": v["fired"], "applicable": v["applicable"]}
                                         for k, v in a_events.items()},
            }
        summary["by_domain"][label] = dom_out
        print(f"\n=== {label} ===")
        for thr in THRESHOLDS:
            t = dom_out["thresholds"][str(thr)]
            print(f"  thr={thr}: Base HMS={t['base_hms_mean']:.4f}  AW HMS={t['aw_hms_mean']:.4f}  "
                  f"ΔHMS={t['delta_hms']:+.4f}  "
                  f"Base ES {t['base_es_fired']}/{t['base_es_app']}  "
                  f"AW ES {t['aw_es_fired']}/{t['aw_es_app']}")
    out_path = ROOT / "c7_sensitivity_results.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()