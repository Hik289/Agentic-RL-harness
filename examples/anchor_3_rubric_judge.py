"""Anchor 3 — Rubric Judge calibration on 60 toy GT.

PASS standard (Director upgrade 2026-06-10 21:15 UTC):
  - per-domain Spearman ρ array (6 values) ALL ≥ 0.6
  - macro mean ρ ≥ 0.8
  - mid-tier-only macro Spearman ρ ≥ 0.5  (collapsed across domains; 18 mid GT)
  - missing_items overlap with annotator >= 70%

Implementation:
  - For each of 60 GT we run judge ONCE (n_repeats=1, low temp default).
    Single LLM call per task, all criteria scored together.
  - Spearman computed on normalized total = total_score / max_total.
  - missing_items overlap: judge.missing_items vs annotator-derived
    "missing" set (criterion with score==0 in GT). Compute Jaccard-like
    micro overlap = |J ∩ A| / max(|A|, 1) averaged over tasks where |A|>0.

If parse failures > 5%, abort and report [异常].
"""
from __future__ import annotations

import json
import os
import sys
import time
import statistics
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1]))

from reward.rubric_judge import RubricJudge
from harness.util.llm_client import LLMClient


DATA = Path(os.environ.get("AGENTICRLHARNESS_DATA",
            "./data"))
GT_PATH = DATA / "toy_groundtruth_annotations.jsonl"
TASK_ROOT = DATA / "synthetic_tasks"

DOMAINS = ["knowledge_work", "coding", "research",
           "multi_tool", "long_memory", "planning"]


# ─────────────────────────────────────────────────────────────────────────────
# Spearman without scipy (rank + Pearson on ranks; handle ties via average rank)
# ─────────────────────────────────────────────────────────────────────────────

def _rank(xs: list[float]) -> list[float]:
    idx = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and xs[idx[j + 1]] == xs[idx[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based ranks
        for k in range(i, j + 1):
            ranks[idx[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    rx, ry = _rank(xs), _rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx)
    dy = sum((b - my) ** 2 for b in ry)
    if dx == 0 or dy == 0:
        return None
    return num / (dx ** 0.5 * dy ** 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Load helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_task(task_id: str) -> dict | None:
    domain = task_id.rsplit("_", 1)[0]
    td = TASK_ROOT / domain / task_id
    if not td.exists():
        return None
    tj = json.loads((td / "task.json").read_text())
    rj = json.loads((td / "rubric.json").read_text())
    # Build a short inputs summary (first 200 chars of each input file)
    inputs_summary = ""
    in_dir = td / "inputs"
    if in_dir.exists():
        summary_lines = []
        for p in in_dir.rglob("*"):
            if p.is_file():
                try:
                    text = p.read_text(errors="replace")[:400]
                    summary_lines.append(f"--- {p.relative_to(in_dir)} ---\n{text}")
                except Exception:
                    pass
        inputs_summary = "\n".join(summary_lines)[:1800]
    # Also include reference/answer.md if present (gives judge expected answer / constraints)
    ref_dir = td / "reference"
    if ref_dir.exists():
        ref_chunks = []
        for p in ref_dir.glob("*.md"):
            try:
                ref_chunks.append(f"--- REFERENCE {p.name} ---\n{p.read_text(errors='replace')[:800]}")
            except Exception:
                pass
        if ref_chunks:
            inputs_summary = (inputs_summary + "\n\n" + "\n".join(ref_chunks))[:3000]
    return {"task": tj, "rubric": rj, "inputs_summary": inputs_summary,
            "task_dir": td, "domain": domain}


def gt_missing_set(gt: dict) -> set:
    return {c["id"] for c in gt["criterion_scores"] if c["score"] == 0}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    gt_rows = [json.loads(l) for l in GT_PATH.read_text().strip().splitlines()]
    print(f"loaded {len(gt_rows)} GT rows")

    client = LLMClient()
    # CALIBRATION mode: judge may see reference/*.md (annotator did too).
    # All production reward calls elsewhere must use the default reward mode.
    judge = RubricJudge(client=client, n_repeats=1,
                        max_completion_tokens=900,
                        default_mode="calibration")

    per_task = []
    n_parse_fail = 0
    t0 = time.monotonic()
    total_cost = 0.0
    for i, gt in enumerate(gt_rows):
        info = load_task(gt["task_id"])
        if info is None:
            print(f"  [{i+1}/{len(gt_rows)}] SKIP {gt['task_id']} (task dir missing)")
            continue
        report = judge.score(
            info["task"], info["rubric"], gt["candidate_output"],
            inputs_summary=info["inputs_summary"],
        )
        total_cost += report.cost_usd
        ok = report.ok
        if not ok:
            n_parse_fail += 1
            print(f"  [{i+1}/{len(gt_rows)}] PARSE_FAIL {gt['task_id']} err={report.error}")
        gt_norm = float(gt["total"]) / float(gt["max_total"])
        gt_missing = gt_missing_set(gt)
        judge_missing = set(report.missing_items) if ok else set()
        # Per-criterion abs delta
        crit_deltas = {}
        if ok:
            gt_per = {c["id"]: c["score"] for c in gt["criterion_scores"]}
            for cs in report.criteria_scores:
                cid = cs["criterion_id"]
                if cid in gt_per:
                    crit_deltas[cid] = abs(cs["score"] - gt_per[cid])

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
            "crit_abs_deltas": crit_deltas,
            "judge_cost_usd": report.cost_usd,
            "judge_lat_s": report.latency_s,
            "ok": ok,
            "error": report.error,
        })
        if (i + 1) % 10 == 0 or (i + 1) == len(gt_rows):
            elapsed = time.monotonic() - t0
            print(f"  [{i+1}/{len(gt_rows)}]  elapsed={elapsed:.1f}s  cost=${total_cost:.4f}  parse_fail={n_parse_fail}")

    # Aggregate metrics
    parse_fail_rate = n_parse_fail / max(len(per_task), 1)
    if parse_fail_rate > 0.05:
        out_blob = {
            "timestamp_jst": time.strftime("%Y-%m-%dT%H:%M:%S+09:00", time.localtime()),
            "abort_reason": f"parse_fail_rate={parse_fail_rate:.2%} > 5% threshold",
            "n_parse_fail": n_parse_fail, "n_total": len(per_task),
            "per_task": per_task,
        }
        out_path = THIS.parent / "anchor_3_results.json"
        out_path.write_text(json.dumps(out_blob, indent=2, default=str))
        print(f"[anchor_3] ABORT due to parse failures > 5% -> {out_path}")
        return 2

    # per-domain Spearman
    per_domain_spearman = {}
    for d in DOMAINS:
        rows = [r for r in per_task if r["domain"] == d and r["ok"]]
        gt_norms = [r["gt_norm"] for r in rows]
        judge_norms = [r["judge_norm"] for r in rows]
        rho = spearman(gt_norms, judge_norms)
        per_domain_spearman[d] = {"rho": rho, "n": len(rows)}

    macro = [v["rho"] for v in per_domain_spearman.values()
             if v["rho"] is not None]
    macro_mean = statistics.fmean(macro) if macro else None
    macro_all_ge_0p6 = all((v["rho"] is not None and v["rho"] >= 0.6)
                            for v in per_domain_spearman.values())

    # Mid-tier only macro Spearman (collapsed across domains)
    mid_rows = [r for r in per_task if r["tier"] == "mid" and r["ok"]]
    mid_rho = spearman([r["gt_norm"] for r in mid_rows],
                       [r["judge_norm"] for r in mid_rows])

    # Missing items overlap (averaged over tasks where gt has >=1 missing)
    overlap_rates = []
    for r in per_task:
        if r["ok"] and r["gt_missing_size"] > 0:
            overlap_rates.append(r["missing_overlap_size"] / r["gt_missing_size"])
    missing_overlap_mean = statistics.fmean(overlap_rates) if overlap_rates else None

    # per-tier mean abs delta in normalized score (diagnostic)
    by_tier_mae = {}
    for tier in ("high", "mid", "low"):
        rs = [abs(r["gt_norm"] - r["judge_norm"]) for r in per_task
              if r["ok"] and r["tier"] == tier]
        by_tier_mae[tier] = statistics.fmean(rs) if rs else None

    pass_per_domain = macro_all_ge_0p6
    pass_macro_mean = (macro_mean is not None and macro_mean >= 0.8)
    pass_mid_tier = (mid_rho is not None and mid_rho >= 0.5)
    pass_missing = (missing_overlap_mean is not None and missing_overlap_mean >= 0.7)
    overall_ok = pass_per_domain and pass_macro_mean and pass_mid_tier and pass_missing

    summary = {
        "timestamp_jst": time.strftime("%Y-%m-%dT%H:%M:%S+09:00", time.localtime()),
        "n_total": len(per_task),
        "n_parse_fail": n_parse_fail,
        "total_cost_usd": total_cost,
        "elapsed_s": time.monotonic() - t0,
        "per_domain_spearman": per_domain_spearman,
        "macro_mean_spearman": macro_mean,
        "mid_tier_spearman": mid_rho,
        "missing_overlap_mean": missing_overlap_mean,
        "by_tier_mae_norm": by_tier_mae,
        "invariants": {
            "per_domain_all_ge_0.6": pass_per_domain,
            "macro_mean_ge_0.8": pass_macro_mean,
            "mid_tier_ge_0.5": pass_mid_tier,
            "missing_overlap_ge_0.7": pass_missing,
        },
        "overall_ok": overall_ok,
        "per_task": per_task,
    }
    out_path = THIS.parent / "anchor_3_results.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))

    print()
    print(f"per-domain Spearman:")
    for d, v in per_domain_spearman.items():
        rho = v["rho"]
        print(f"  {d:<16} n={v['n']:>2}  rho={'NA' if rho is None else f'{rho:+.3f}'}")
    print(f"\nmacro mean rho      = {macro_mean}")
    print(f"mid-tier rho        = {mid_rho}")
    print(f"missing_overlap     = {missing_overlap_mean}")
    print(f"by-tier MAE norm    = {by_tier_mae}")
    print(f"\ninvariants:")
    for k, v in summary["invariants"].items():
        print(f"  {k:30s}  {'✓' if v else '✗'}")
    print(f"\noverall_ok = {overall_ok}  total_cost=${total_cost:.4f}")
    return 0 if overall_ok else 2


if __name__ == "__main__":
    sys.exit(main())
