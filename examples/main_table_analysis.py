"""Main-table analysis bonus pack for data_scientist + theorist.

Inputs: 6 × main_{domain}/results.json + main_{domain}/eval_detail_{base,aw}.jsonl

Outputs (single JSON + markdown for fast read):
  - per-domain Base/AW G mean ± std, ΔG, paired bootstrap p, CI95
  - per-domain Base/AW HMS_norm mean ± std, ΔHMS (95% CI via boostrap on per-episode hms_norm)
  - sign test on ΔG sign-pattern (binomial 6/n)
  - sign test on ΔHMS sign-pattern
  - macro Δ over 6 domains (mean + 95% CI bootstrap over domains)
  - HMS per-event × per-domain fired-rate (base vs AW) → paper §5 Table 2 candidate
"""
import os
from __future__ import annotations

import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(os.environ.get("AGENTICRLHARNESS_RESULTS", "./results"))
DOMAINS = ["knowledge_work", "coding", "research",
           "multi_tool", "long_memory", "planning"]

EVENTS = ["CheckBeforeSubmit", "EvidenceBeforeClaim", "TestBeforeSubmit",
          "RevisionAfterFailure", "ValidToolUse", "StopWhenSufficient",
          "EarlySubmit"]


def _binom_p_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value."""
    pmf = [math.comb(n, i) * (p**i) * ((1-p)**(n-i)) for i in range(n+1)]
    obs = pmf[k]
    return sum(v for v in pmf if v <= obs + 1e-12)


def _bootstrap_ci(xs: list[float], n_resamples: int = 5000,
                  rng_seed: int = 0) -> tuple[float, float, float]:
    if not xs:
        return (None, None, None)
    rng = random.Random(rng_seed)
    n = len(xs)
    means = []
    for _ in range(n_resamples):
        s = sum(xs[rng.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    mu = statistics.fmean(xs)
    lo = means[int(0.025 * n_resamples)]
    hi = means[int(0.975 * n_resamples)]
    return (mu, lo, hi)


def _paired_bootstrap_delta(base: list[float], aw: list[float],
                             n_resamples: int = 5000, seed: int = 0) -> dict:
    assert len(base) == len(aw)
    n = len(base)
    diffs = [aw[i] - base[i] for i in range(n)]
    obs = sum(diffs)/n
    rng = random.Random(seed)
    centered = [d - obs for d in diffs]
    null = [sum(centered[rng.randrange(n)] for _ in range(n))/n for _ in range(n_resamples)]
    p_two = sum(1 for x in null if abs(x) >= abs(obs)) / n_resamples
    boot = sorted([sum(diffs[rng.randrange(n)] for _ in range(n))/n for _ in range(n_resamples)])
    lo = boot[int(0.025*n_resamples)]
    hi = boot[int(0.975*n_resamples)]
    return {"n": n, "delta": obs, "p_two_sided": p_two,
            "ci95_lo": lo, "ci95_hi": hi}


def _read_eval_detail(domain: str, which: str) -> list[dict]:
    path = ROOT / f"main_{domain}" / f"eval_detail_{which}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    per_domain = {}
    for d in DOMAINS:
        res = json.loads((ROOT / f"main_{d}" / "results.json").read_text())
        per_domain[d] = res

    # ── G analysis ──
    G_table = []
    base_means, aw_means = [], []
    for d in DOMAINS:
        r = per_domain[d]
        base_means.append(r["base"]["mean"])
        aw_means.append(r["aw"]["mean"])
        G_table.append({
            "domain": d,
            "base_mean": r["base"]["mean"], "base_std": r["base"]["std"],
            "aw_mean": r["aw"]["mean"], "aw_std": r["aw"]["std"],
            "delta_G": r["delta_mean"],
            "p_two_sided": r["paired_bootstrap"]["p_two_sided"],
            "ci95_lo": r["paired_bootstrap"].get("ci95_lo"),
            "ci95_hi": r["paired_bootstrap"].get("ci95_hi"),
            "n_paired": r["paired_bootstrap"]["n"],
            "buffer_G_mean": r["buffer"]["G_mean"],
            "buffer_G_std": r["buffer"]["G_std"],
            "buffer_unique": r["buffer"]["G_unique"],
            "buffer_sat_at_1": r["buffer"]["G_saturation_at_1.0"],
            "eval_sat_at_1": r["eval_saturation_at_1.0"],
            "n_train_tasks": r["n_train_tasks"],
            "n_eval_tasks": r["n_eval_tasks"],
        })

    delta_Gs = [g["delta_G"] for g in G_table]
    pos_G = sum(1 for d in delta_Gs if d > 0)
    sign_test_G = {
        "n_positive": pos_G, "n_total": len(delta_Gs),
        "p_binomial_two_sided": _binom_p_two_sided(pos_G, len(delta_Gs)),
    }
    macro_G_delta_stats = _bootstrap_ci(delta_Gs)

    # ── HMS analysis ──
    HMS_table = []
    delta_HMSs = []
    for d in DOMAINS:
        r = per_domain[d]
        dh = r["hms_delta_mean"]
        HMS_table.append({
            "domain": d,
            "hms_base_mean": r["hms_base"]["hms_norm_mean"],
            "hms_base_std": r["hms_base"]["hms_norm_std"],
            "hms_aw_mean": r["hms_aw"]["hms_norm_mean"],
            "hms_aw_std": r["hms_aw"]["hms_norm_std"],
            "delta_HMS": dh,
        })
        if dh is not None:
            delta_HMSs.append(dh)

    pos_H = sum(1 for d in delta_HMSs if d > 0)
    sign_test_H = {
        "n_positive": pos_H, "n_total": len(delta_HMSs),
        "p_binomial_two_sided": _binom_p_two_sided(pos_H, len(delta_HMSs)),
    }
    macro_HMS_delta_stats = _bootstrap_ci(delta_HMSs)

    # ── HMS per-event × per-domain Table 2 ──
    event_table = []
    for ev in EVENTS:
        row = {"event": ev}
        for d in DOMAINS:
            r = per_domain[d]
            bee = r["hms_base"]["per_event"].get(ev, {})
            awe = r["hms_aw"]["per_event"].get(ev, {})
            b_fr = bee.get("fired_rate_among_applicable")
            a_fr = awe.get("fired_rate_among_applicable")
            row[d] = {
                "base_fired_rate": b_fr,
                "aw_fired_rate": a_fr,
                "base_applicable_count": bee.get("applicable_count"),
                "aw_applicable_count": awe.get("applicable_count"),
                "delta_fired_rate": ((a_fr - b_fr) if (a_fr is not None and b_fr is not None) else None),
            }
        event_table.append(row)

    # ── Save outputs ──
    out = {
        "per_domain_G": G_table,
        "per_domain_HMS": HMS_table,
        "sign_test_G": sign_test_G,
        "sign_test_HMS": sign_test_H,
        "macro_G_delta": {"mean": macro_G_delta_stats[0],
                           "ci95_lo": macro_G_delta_stats[1],
                           "ci95_hi": macro_G_delta_stats[2]},
        "macro_HMS_delta": {"mean": macro_HMS_delta_stats[0],
                             "ci95_lo": macro_HMS_delta_stats[1],
                             "ci95_hi": macro_HMS_delta_stats[2]},
        "hms_event_x_domain_table": event_table,
        "domains": DOMAINS,
        "events": EVENTS,
        "total_llm_cost_usd": sum(per_domain[d]["total_llm_cost_usd"] for d in DOMAINS),
    }
    out_path = ROOT / "main_table_analysis.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {out_path}")

    # ── Pretty markdown ──
    md = []
    md.append("# Main Table — 6-domain Base vs Offline AW")
    md.append("")
    md.append(f"**Total LLM cost across 6 domains**: ${out['total_llm_cost_usd']:.2f}")
    md.append("")
    md.append("## Final G (Class R reward, structural verifier)")
    md.append("")
    md.append("| Domain | Base mean ± std | AW mean ± std | ΔG | bootstrap p | CI95 |")
    md.append("|---|---|---|---:|---:|---|")
    for g in G_table:
        md.append(f"| {g['domain']:<14} | {g['base_mean']:.4f} ± {g['base_std']:.4f} "
                  f"| {g['aw_mean']:.4f} ± {g['aw_std']:.4f} "
                  f"| {g['delta_G']:+.4f} "
                  f"| {g['p_two_sided']:.3f} "
                  f"| ({g['ci95_lo']:+.4f}, {g['ci95_hi']:+.4f}) |")
    md.append("")
    md.append(f"- ΔG sign test: **{sign_test_G['n_positive']}/{sign_test_G['n_total']} positive**, "
              f"binomial p (two-sided) = **{sign_test_G['p_binomial_two_sided']:.4f}**")
    md.append(f"- macro mean ΔG = {macro_G_delta_stats[0]:+.5f}, 95% CI ({macro_G_delta_stats[1]:+.5f}, {macro_G_delta_stats[2]:+.5f})")
    md.append("")
    md.append("## HMS_norm (process maturity, readme §20)")
    md.append("")
    md.append("| Domain | Base mean ± std | AW mean ± std | ΔHMS |")
    md.append("|---|---|---|---:|")
    for h in HMS_table:
        md.append(f"| {h['domain']:<14} | {h['hms_base_mean']:+.4f} ± {h['hms_base_std']:.4f} "
                  f"| {h['hms_aw_mean']:+.4f} ± {h['hms_aw_std']:.4f} "
                  f"| {h['delta_HMS']:+.4f} |")
    md.append("")
    md.append(f"- ΔHMS sign test: **{sign_test_H['n_positive']}/{sign_test_H['n_total']} positive**, "
              f"binomial p (two-sided) = **{sign_test_H['p_binomial_two_sided']:.4f}**")
    md.append(f"- macro mean ΔHMS = {macro_HMS_delta_stats[0]:+.5f}, 95% CI ({macro_HMS_delta_stats[1]:+.5f}, {macro_HMS_delta_stats[2]:+.5f})")
    md.append("")
    md.append("## HMS per-event × per-domain (paper §5 Table 2 candidate)")
    md.append("")
    md.append("Fired-rate among applicable episodes (Base → AW; Δ in brackets).")
    md.append("")
    header = "| event | " + " | ".join(DOMAINS) + " |"
    md.append(header)
    md.append("|" + "---|" * (len(DOMAINS) + 1))
    for row in event_table:
        cells = []
        for d in DOMAINS:
            cell = row[d]
            b = cell["base_fired_rate"]
            a = cell["aw_fired_rate"]
            delta = cell["delta_fired_rate"]
            if b is None and a is None:
                cells.append("N/A")
            elif b is None:
                cells.append(f"NA→{a:.2f}")
            elif a is None:
                cells.append(f"{b:.2f}→NA")
            else:
                cells.append(f"{b:.2f}→{a:.2f} ({delta:+.2f})")
        md.append(f"| {row['event']} | " + " | ".join(cells) + " |")
    md.append("")
    md.append("## Buffer + eval saturation (fair-test confirmation)")
    md.append("")
    md.append("| Domain | buffer G mean | std | unique | sat@1.0 | eval sat@1.0 |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for g in G_table:
        md.append(f"| {g['domain']:<14} | {g['buffer_G_mean']:.3f} | {g['buffer_G_std']:.3f} "
                  f"| {g['buffer_unique']} | {g['buffer_sat_at_1']:.0%} | {g['eval_sat_at_1']:.0%} |")
    md.append("")
    (ROOT / "main_table_analysis.md").write_text("\n".join(md))
    print(f"wrote {ROOT / 'main_table_analysis.md'}")
    print()
    print("\n".join(md[:50]))


if __name__ == "__main__":
    sys.exit(main() or 0)