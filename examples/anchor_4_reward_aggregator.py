"""Anchor 4: Reward aggregator numeric correctness.

5 synthetic trajectories, each with a hand-computed expected reward, must match
the aggregator output within 1e-6.

The synthetic episodes cover:
  ep_A: clean high-quality submit (no penalties)
  ep_B: moderate submit, mild cost penalty
  ep_C: errors + low rubric (heavy penalties)
  ep_D: early submit (triggers P_early_submit)
  ep_E: perfect rubric but cost overrun (cost penalty saturates)

Each episode hand-computes:
  R_rubric, R_verify, R_format, R_task, P_error, P_cost, P_early_submit
  weighted contributions and R_total.
The script asserts numerical equality with the aggregator output.
Also exercises:
  * compute_rubric_score (per-criterion → norm + missing_items)
  * compute_error_penalty
  * compute_early_submit_penalty
  * format_reward over 3 schema types (markdown / json / csv)
  * cost_penalty saturation
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reward.reward_aggregator import (
    DEFAULT_COEFS,
    aggregate_reward,
    compute_rubric_score,
    compute_error_penalty,
    compute_early_submit_penalty,
)
from reward.format_checker import format_reward
from reward.cost_penalty import cost_penalty

TOL = 1e-6

C = DEFAULT_COEFS  # α=0.20, β=0.10, δ=0.20, γ=0.25, λ=0.05, μ=0.10


def _close(a: float, b: float, tol: float = TOL) -> bool:
    return math.isclose(a, b, rel_tol=0, abs_tol=tol)


# ───────────────────────────────────────────────────────────────────────────────
# 5 hand-computed episodes.
# Each tuple = (label, inputs_dict, expected_R_total)
# ───────────────────────────────────────────────────────────────────────────────

EPISODES = [
    # ep_A: clean high-quality submission
    # R_rubric = 0.90, R_verify = 0.95, R_format = 1.0, R_task = 0.80
    # P_error = 0.0, P_cost = 0.20, P_early_submit = 0.0
    # R_total = 0.90 + 0.20*0.95 + 0.10*1.0 + 0.20*0.80 - 0.25*0 - 0.05*0.20 - 0.10*0
    #        = 0.90 + 0.190 + 0.100 + 0.160 - 0.000 - 0.010 - 0.000
    #        = 1.340
    (
        "ep_A_clean_submit",
        dict(R_rubric=0.90, R_verify=0.95, R_format=1.0, R_task=0.80,
             P_error=0.0, P_cost=0.20, P_early_submit=0.0),
        1.340,
    ),
    # ep_B: moderate, mild cost
    # R_rubric=0.55, R_verify=0.60, R_format=0.5, R_task=0.40
    # P_error=0.10, P_cost=0.60, P_early_submit=0.0
    # R_total = 0.55 + 0.20*0.60 + 0.10*0.5 + 0.20*0.40
    #          - 0.25*0.10 - 0.05*0.60 - 0.10*0
    #        = 0.55 + 0.12 + 0.05 + 0.08 - 0.025 - 0.030 - 0.000
    #        = 0.745
    (
        "ep_B_moderate",
        dict(R_rubric=0.55, R_verify=0.60, R_format=0.5, R_task=0.40,
             P_error=0.10, P_cost=0.60, P_early_submit=0.0),
        0.745,
    ),
    # ep_C: errors + low rubric
    # R_rubric=0.20, R_verify=0.10, R_format=0.0, R_task=0.0
    # P_error=0.80, P_cost=0.40, P_early_submit=0.0
    # R_total = 0.20 + 0.020 + 0.000 + 0.000 - 0.200 - 0.020 - 0.000
    #        = 0.0
    (
        "ep_C_errors",
        dict(R_rubric=0.20, R_verify=0.10, R_format=0.0, R_task=0.0,
             P_error=0.80, P_cost=0.40, P_early_submit=0.0),
        0.0,
    ),
    # ep_D: early submit triggered
    # R_rubric=0.35, R_verify=0.30, R_format=1.0, R_task=0.10
    # P_error=0.0, P_cost=0.10, P_early_submit=1.0
    # R_total = 0.35 + 0.060 + 0.100 + 0.020 - 0 - 0.005 - 0.100
    #        = 0.425
    (
        "ep_D_early_submit",
        dict(R_rubric=0.35, R_verify=0.30, R_format=1.0, R_task=0.10,
             P_error=0.0, P_cost=0.10, P_early_submit=1.0),
        0.425,
    ),
    # ep_E: perfect rubric but cost saturates at 1.0 (over budget)
    # R_rubric=1.0, R_verify=1.0, R_format=1.0, R_task=1.0
    # P_error=0.0, P_cost=1.0, P_early_submit=0.0
    # R_total = 1.0 + 0.20 + 0.10 + 0.20 - 0 - 0.05 - 0
    #        = 1.45
    (
        "ep_E_perfect_overcost",
        dict(R_rubric=1.0, R_verify=1.0, R_format=1.0, R_task=1.0,
             P_error=0.0, P_cost=1.0, P_early_submit=0.0),
        1.45,
    ),
]


def test_aggregate_reward() -> list[dict]:
    """Verify aggregator on the 5 hand-computed episodes."""
    rows = []
    for label, inputs, expected in EPISODES:
        out = aggregate_reward(**inputs)
        ok = _close(out.R_total, expected)

        # Also independently re-derive R_total from the breakdown to double-check
        recompute = (
            inputs["R_rubric"]
            + C["alpha"] * inputs["R_verify"]
            + C["beta"] * inputs["R_format"]
            + C["delta"] * inputs["R_task"]
            - C["gamma"] * inputs["P_error"]
            - C["lam"] * inputs["P_cost"]
            - C["mu"] * inputs["P_early_submit"]
        )
        ok2 = _close(out.R_total, recompute)

        row = {
            "label": label,
            "inputs": inputs,
            "expected_R_total": expected,
            "got_R_total": out.R_total,
            "manual_recompute": recompute,
            "weighted": {
                "verify": out.weighted_verify,
                "format": out.weighted_format,
                "task": out.weighted_task,
                "error": out.weighted_error,
                "cost": out.weighted_cost,
                "early": out.weighted_early_submit,
            },
            "diff_vs_expected": out.R_total - expected,
            "diff_vs_recompute": out.R_total - recompute,
            "ok_vs_expected": ok,
            "ok_vs_recompute": ok2,
        }
        rows.append(row)
        status = "✓" if (ok and ok2) else "✗"
        print(
            f"  {status} {label:<28} got={out.R_total:.6f}  "
            f"expected={expected:.6f}  recompute={recompute:.6f}"
        )
    return rows


def test_rubric_aggregation() -> dict:
    """Hand-checked criterion aggregation."""
    # 4 criteria, total_max=7.5 (matches data_scientist's knowledge_work tasks)
    criteria = [
        dict(id="c1", score=2.4, max_score=3.0, missing=False, category="correctness"),
        dict(id="c2", score=1.6, max_score=2.0, missing=False, category="evidence"),
        dict(id="c3", score=0.0, max_score=1.5, missing=True,  category="completeness"),
        dict(id="c4", score=1.0, max_score=1.0, missing=False, category="format"),
    ]
    # raw = 2.4+1.6+0+1.0 = 5.0
    # total = 7.5
    # norm = 5.0/7.5 = 0.666666...
    res = compute_rubric_score(criteria)
    raw_ok = _close(res["rubric_score_raw"], 5.0)
    norm_ok = _close(res["rubric_score_norm"], 5.0 / 7.5)
    missing_ok = res["missing_items"] == ["c3"]
    cat_ok = (
        _close(res["by_category"]["correctness"]["score"], 2.4)
        and _close(res["by_category"]["evidence"]["score"], 1.6)
        and _close(res["by_category"]["completeness"]["score"], 0.0)
        and _close(res["by_category"]["format"]["score"], 1.0)
    )
    out = {"raw_ok": raw_ok, "norm_ok": norm_ok, "missing_ok": missing_ok, "cat_ok": cat_ok,
           "result": res}
    print(f"  rubric raw={res['rubric_score_raw']:.6f} norm={res['rubric_score_norm']:.6f} "
          f"missing={res['missing_items']} pass={all([raw_ok, norm_ok, missing_ok, cat_ok])}")
    return out


def test_error_and_early() -> dict:
    """Error normalization + early submit threshold."""
    e1 = compute_error_penalty(3, 10)          # 0.3
    e2 = compute_error_penalty(15, 10)         # clamp to 1.0
    e3 = compute_error_penalty(0, 10)          # 0.0
    es1 = compute_early_submit_penalty(information_coverage=0.40, rubric_coverage=0.90)  # triggers info
    es2 = compute_early_submit_penalty(information_coverage=0.80, rubric_coverage=0.55)  # triggers rubric
    es3 = compute_early_submit_penalty(information_coverage=0.80, rubric_coverage=0.80)  # neither
    out = {
        "error_3/10": e1, "error_15/10": e2, "error_0/10": e3,
        "early_low_info": es1, "early_low_rubric": es2, "early_safe": es3,
        "ok": all([
            _close(e1, 0.3), _close(e2, 1.0), _close(e3, 0.0),
            es1 == 1.0, es2 == 1.0, es3 == 0.0,
        ]),
    }
    print(f"  error_penalty: 3/10={e1}  15/10={e2}  0/10={e3}  ok")
    print(f"  early_submit: info_lo={es1}  rubric_lo={es2}  safe={es3}  ok")
    return out


def test_cost_penalty() -> dict:
    out = {
        "p_cost_0.2_1.0": cost_penalty(0.2, 1.0),     # 0.2
        "p_cost_1.5_1.0": cost_penalty(1.5, 1.0),     # 1.0 (saturated)
        "p_cost_neg_1.0": cost_penalty(-0.5, 1.0),    # 0.0 (clamped)
        "p_cost_zero_budget": cost_penalty(0.5, 0.0), # 0.0 (no budget defined)
    }
    out["ok"] = (
        _close(out["p_cost_0.2_1.0"], 0.2)
        and _close(out["p_cost_1.5_1.0"], 1.0)
        and _close(out["p_cost_neg_1.0"], 0.0)
        and _close(out["p_cost_zero_budget"], 0.0)
    )
    print(f"  cost_penalty: {out}")
    return out


def test_format_reward() -> dict:
    md_good = "# Summary\nfoo\n## Findings\nbar\n## Recommendation\nbaz"
    md_partial = "# Summary\nfoo"  # missing Findings + Recommendation
    md_bad = "no headers at all just plain text"
    json_good = '{"answer": "x", "confidence": 0.8}'
    json_partial = '{"answer": "x"}'  # missing confidence
    json_bad = "not json {{{"
    csv_good = "id,name,value\n1,a,3"
    csv_partial = "id,name\n1,a"
    md_schema = {"schema_type": "markdown_sections",
                 "required_sections": ["Summary", "Findings", "Recommendation"]}
    json_schema = {"schema_type": "json", "required_keys": ["answer", "confidence"]}
    csv_schema = {"schema_type": "csv", "required_columns": ["id", "name", "value"]}

    out = {
        "md_good": format_reward(md_good, md_schema),
        "md_partial": format_reward(md_partial, md_schema),
        "md_bad": format_reward(md_bad, md_schema),
        "json_good": format_reward(json_good, json_schema),
        "json_partial": format_reward(json_partial, json_schema),
        "json_bad": format_reward(json_bad, json_schema),
        "csv_good": format_reward(csv_good, csv_schema),
        "csv_partial": format_reward(csv_partial, csv_schema),
    }
    out["ok"] = (
        out["md_good"] == 1.0 and out["md_partial"] == 0.5 and out["md_bad"] == 0.0
        and out["json_good"] == 1.0 and out["json_partial"] == 0.5 and out["json_bad"] == 0.0
        and out["csv_good"] == 1.0 and out["csv_partial"] == 0.5
    )
    print(f"  format_reward: {out}")
    return out


def main():
    print("[anchor_4] aggregate_reward — 5 hand-computed episodes")
    rows = test_aggregate_reward()

    print("\n[anchor_4] rubric aggregation (4 criteria, total_max=7.5)")
    r2 = test_rubric_aggregation()

    print("\n[anchor_4] error penalty + early submit penalty")
    r3 = test_error_and_early()

    print("\n[anchor_4] cost penalty")
    r4 = test_cost_penalty()

    print("\n[anchor_4] format reward (markdown / json / csv)")
    r5 = test_format_reward()

    all_aggregator_ok = all(r["ok_vs_expected"] and r["ok_vs_recompute"] for r in rows)
    overall_ok = (
        all_aggregator_ok
        and r2["raw_ok"] and r2["norm_ok"] and r2["missing_ok"] and r2["cat_ok"]
        and r3["ok"] and r4["ok"] and r5["ok"]
    )

    blob = {
        "timestamp_jst": time.strftime("%Y-%m-%dT%H:%M:%S+09:00", time.localtime()),
        "tolerance": TOL,
        "coefs": dict(C),
        "aggregate_reward_tests": rows,
        "rubric_aggregation_test": r2,
        "error_and_early_test": r3,
        "cost_penalty_test": r4,
        "format_reward_test": r5,
        "all_aggregator_ok": all_aggregator_ok,
        "overall_ok": overall_ok,
    }
    out_path = Path(__file__).resolve().parent / "anchor_4_results.json"
    out_path.write_text(json.dumps(blob, indent=2, default=str))
    print(f"\n[anchor_4] wrote {out_path}  overall_ok={overall_ok}")
    return 0 if overall_ok else 2


if __name__ == "__main__":
    sys.exit(main())
