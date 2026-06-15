"""Anchor 3b — RubricJudge mode="reward" calibration on 60 toy GT.

Director-mandated (2026-06-11 00:26 UTC): before resuming anchor_5, verify
that the reward-mode judge (which does NOT see reference/*.md) is still
calibrated against annotators.

Invariants (relaxed from anchor_3 because reward mode loses reference info):
  - per-domain Spearman ρ ≥ 0.6
  - macro mean ρ ≥ 0.7 (anchor_3 was 0.8)
  - missing_items overlap ≥ 0.6 (anchor_3 was 0.7)

Reuses anchor_3 evaluator with default_mode="reward".
"""
from __future__ import annotations

import json
import sys
import time
import statistics
import os
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1]))
sys.path.insert(0, str(THIS.parent))

import anchor_3_rubric_judge as a3
from reward.rubric_judge import RubricJudge
from harness.util.llm_client import LLMClient


def main():
    gt_rows = [json.loads(l) for l in a3.GT_PATH.read_text().strip().splitlines()]
    print(f"loaded {len(gt_rows)} GT rows (mode=reward)")

    client = LLMClient()
    judge = RubricJudge(client=client, n_repeats=1,
                        max_completion_tokens=900,
                        default_mode="reward")

    per_task = []
    n_parse_fail = 0
    t0 = time.monotonic()
    total_cost = 0.0
    for i, gt in enumerate(gt_rows):
        info = a3.load_task(gt["task_id"])
        if info is None:
            continue
        report = judge.score(
            info["task"], info["rubric"], gt["candidate_output"],
            inputs_summary=info["inputs_summary"],
        )
        total_cost += report.cost_usd
        ok = report.ok
        if not ok:
            n_parse_fail += 1
        gt_norm = float(gt["total"]) / float(gt["max_total"])
        gt_missing = a3.gt_missing_set(gt)
        judge_missing = set(report.missing_items) if ok else set()
        per_task.append({
            "task_id": gt["task_id"],
            "domain": info["domain"],
            "tier": gt["quality_tier"],
            "gt_norm": gt_norm,
            "judge_norm": report.normalized_score if ok else None,
            "gt_total": float(gt["total"]),
            "judge_total": report.total_score if ok else None,
            "max_total": float(gt["max_total"]),
            "gt_missing": sorted(gt_missing),
            "judge_missing": sorted(judge_missing),
            "missing_overlap_size": len(gt_missing & judge_missing),
            "gt_missing_size": len(gt_missing),
            "judge_cost_usd": report.cost_usd,
            "judge_lat_s": report.latency_s,
            "ok": ok,
            "error": report.error,
        })
        if (i + 1) % 10 == 0 or (i + 1) == len(gt_rows):
            elapsed = time.monotonic() - t0
            print(f"  [{i+1}/{len(gt_rows)}]  elapsed={elapsed:.1f}s  cost=${total_cost:.4f}  parse_fail={n_parse_fail}", flush=True)

    per_domain_spearman = {}
    for d in a3.DOMAINS:
        rows = [r for r in per_task if r["domain"] == d and r["ok"]]
        rho = a3.spearman([r["gt_norm"] for r in rows],
                          [r["judge_norm"] for r in rows])
        per_domain_spearman[d] = {"rho": rho, "n": len(rows)}
    macro = [v["rho"] for v in per_domain_spearman.values() if v["rho"] is not None]
    macro_mean = statistics.fmean(macro) if macro else None
    mid_rows = [r for r in per_task if r["tier"] == "mid" and r["ok"]]
    mid_rho = a3.spearman([r["gt_norm"] for r in mid_rows],
                          [r["judge_norm"] for r in mid_rows])
    overlap_rates = [r["missing_overlap_size"] / r["gt_missing_size"]
                     for r in per_task if r["ok"] and r["gt_missing_size"] > 0]
    missing_overlap = statistics.fmean(overlap_rates) if overlap_rates else None
    by_tier_mae = {}
    for tier in ("high", "mid", "low"):
        rs = [abs(r["gt_norm"] - r["judge_norm"]) for r in per_task
              if r["ok"] and r["tier"] == tier]
        by_tier_mae[tier] = statistics.fmean(rs) if rs else None
    coding_high = [r["judge_norm"] for r in per_task
                   if r["ok"] and r["domain"] == "coding" and r["tier"] == "high"]
    coding_high_stats = {
        "n": len(coding_high),
        "mean": statistics.fmean(coding_high) if coding_high else None,
        "min": min(coding_high) if coding_high else None,
        "max": max(coding_high) if coding_high else None,
    }
    pass_per_domain = all((v["rho"] is not None and v["rho"] >= 0.6)
                          for v in per_domain_spearman.values())
    pass_macro = (macro_mean is not None and macro_mean >= 0.7)
    pass_missing = (missing_overlap is not None and missing_overlap >= 0.6)
    overall_ok = pass_per_domain and pass_macro and pass_missing

    summary = {
        "timestamp_jst": time.strftime("%Y-%m-%dT%H:%M:%S+09:00", time.localtime()),
        "mode": "reward",
        "n_total": len(per_task),
        "n_parse_fail": n_parse_fail,
        "total_cost_usd": total_cost,
        "elapsed_s": time.monotonic() - t0,
        "per_domain_spearman": per_domain_spearman,
        "macro_mean_spearman": macro_mean,
        "mid_tier_spearman": mid_rho,
        "missing_overlap_mean": missing_overlap,
        "by_tier_mae_norm": by_tier_mae,
        "coding_high_judge_stats": coding_high_stats,
        "invariants": {
            "per_domain_all_ge_0.6": pass_per_domain,
            "macro_mean_ge_0.7": pass_macro,
            "missing_overlap_ge_0.6": pass_missing,
        },
        "overall_ok": overall_ok,
        "per_task": per_task,
    }
    out_path = THIS.parent / "anchor_3b_results.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))

    print()
    print("per-domain Spearman (mode=reward):")
    for d, v in per_domain_spearman.items():
        rho = v["rho"]
        print(f"  {d:<16} n={v['n']:>2}  rho={'NA' if rho is None else f'{rho:+.3f}'}")
    print(f"\nmacro mean rho      = {macro_mean}")
    print(f"mid-tier rho        = {mid_rho}")
    print(f"missing_overlap     = {missing_overlap}")
    print(f"by-tier MAE norm    = {by_tier_mae}")
    print(f"\ncoding HIGH-tier judge stats: {coding_high_stats}")
    print(f"\ninvariants (mode=reward):")
    for k, v in summary["invariants"].items():
        print(f"  {k:30s}  {'OK' if v else 'FAIL'}")
    print(f"\noverall_ok = {overall_ok}  total_cost=${total_cost:.4f}")
    return 0 if overall_ok else 2


if __name__ == "__main__":
    sys.exit(main())
