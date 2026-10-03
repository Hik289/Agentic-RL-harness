from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1]))

from harness.actions import Task
from harness.util.llm_client import LLMClient
from reward.rubric_judge import RubricJudge
from rl.coding_harness import (run_episode_with_policy,
                                 base_harness_policy,
                                 perturbed_base_policy,
                                 mlp_policy)
from rl.offline_aw import AWConfig, train_aw


COD_DOMAIN = "coding"
TASK_IDS = [f"coding_{i:03d}" for i in range(10)]


def load_coding_tasks(task_root: Path) -> list[Task]:
    tasks = []
    for tid in TASK_IDS:
        td = task_root / COD_DOMAIN / tid
        if td.exists():
            tasks.append(Task.load(td))
    return tasks


def collect_buffer(tasks: list[Task], client: LLMClient,
                   *, n_rollouts: int, eps: float, seed: int) -> tuple[list[dict], float]:
    behavioral = perturbed_base_policy(eps=eps, seed=seed)
    buffer = []
    llm_cost = 0.0
    for tid_idx, task in enumerate(tasks):
        for r in range(n_rollouts):
            rng = random.Random((seed + 1) * 1000 + tid_idx * 100 + r)
            logger, scored = run_episode_with_policy(
                task, behavioral, client=client, rng=rng,
            )
            llm_cost += logger.total_cost
            buffer.append({
                "task_id": task.task_id,
                "rollout": r,
                "records": logger.records,
                "structural_score": scored["rubric_score_norm"],
                "return_G": scored["rubric_score_norm"],
            })
    return buffer, llm_cost


def eval_policy(label: str, tasks: list[Task], policy_fn_factory,
                client: LLMClient,
                *, n_rollouts: int, n_seeds: int) -> tuple[dict, float, list[dict]]:
    per_seed_overall = []
    per_seed_per_task = {tid: [] for tid in TASK_IDS}
    llm_cost = 0.0
    detail = []
    all_records: list[dict] = []
    for s in range(n_seeds):
        seed_scores = []
        seed_per_task = {tid: [] for tid in TASK_IDS}
        policy = policy_fn_factory(s)
        for task in tasks:
            for r in range(n_rollouts):
                rng = random.Random((s + 1) * 9999 + r * 13 + hash(task.task_id) % 97)
                logger, scored = run_episode_with_policy(
                    task, policy, client=client, rng=rng,
                )
                llm_cost += logger.total_cost
                G = scored["rubric_score_norm"]
                seed_scores.append(G)
                seed_per_task[task.task_id].append(G)
                detail.append({"label": label, "seed": s, "task_id": task.task_id,
                                "rollout": r, "G": G,
                                "submit": logger.records[-1].get("termination_reason"),
                                "n_steps": len(logger.records)})
                all_records.append({"label": label, "seed": s,
                                     "task_id": task.task_id, "rollout": r,
                                     "records": logger.records,
                                     "G": G})
        per_seed_overall.append(statistics.fmean(seed_scores))
        for tid, lst in seed_per_task.items():
            if lst:
                per_seed_per_task[tid].append(statistics.fmean(lst))
    return {
        "label": label,
        "per_seed_overall": per_seed_overall,
        "mean_over_seeds": statistics.fmean(per_seed_overall),
        "std_over_seeds": statistics.stdev(per_seed_overall) if len(per_seed_overall) > 1 else 0.0,
        "per_seed_per_task": per_seed_per_task,
        "detail": detail,
    }, llm_cost, all_records


def welch_t_two_sample(a: list[float], b: list[float]) -> dict:
    if len(a) < 2 or len(b) < 2:
        return {"t": None, "df": None, "p_two_sided": None,
                "mean_diff": (statistics.fmean(a) if a else 0) - (statistics.fmean(b) if b else 0)}
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    t = (ma - mb) / max(((va / na) + (vb / nb)) ** 0.5, 1e-12)
    num = ((va / na) + (vb / nb)) ** 2
    den = (va ** 2) / ((na ** 2) * (na - 1)) + (vb ** 2) / ((nb ** 2) * (nb - 1))
    df = num / den if den > 0 else 1.0
    import math
    z = abs(t)
    p = 2 * 0.5 * math.erfc(z / (2 ** 0.5))
    return {"t": t, "df": df, "p_two_sided": p, "mean_diff": ma - mb}


def hms_summary(all_records: list[dict]) -> dict:
    from modules.hms_detector import compute_hms, EVENT_CHECKERS
    norms = []
    per_event = {name: {"fired": 0, "applicable": 0, "total": 0}
                  for name in EVENT_CHECKERS}
    for entry in all_records:
        result = compute_hms(entry["records"])
        norms.append(result.hms_norm)
        for name, ev in result.events.items():
            per_event[name]["total"] += 1
            if ev["applicable"]:
                per_event[name]["applicable"] += 1
                if ev["fired"]:
                    per_event[name]["fired"] += 1
    per_event_out = {}
    for name, c in per_event.items():
        fr = c["fired"] / c["applicable"] if c["applicable"] > 0 else None
        ar = c["applicable"] / c["total"] if c["total"] > 0 else None
        per_event_out[name] = {
            "fired_rate_among_applicable": fr,
            "applicable_rate": ar,
            "fired_count": c["fired"],
            "applicable_count": c["applicable"],
            "total": c["total"],
        }
    return {
        "n_episodes": len(norms),
        "hms_norm_mean": statistics.fmean(norms) if norms else None,
        "hms_norm_std": statistics.stdev(norms) if len(norms) > 1 else 0.0,
        "per_event": per_event_out,
    }


def calibrate_c7_threshold(buffer: list[dict], judge_norms: dict[str, list[float]],
                           ) -> dict:
    from modules.hms_detector import check_event_C7
    rows = []
    all_G = [e["return_G"] for e in buffer]
    if not all_G:
        return {}
    med = statistics.median(all_G)
    for e in buffer:
        ev = check_event_C7(e["records"])
        rows.append({
            "task_id": e["task_id"],
            "G": e["return_G"],
            "fired": ev["fired"] and ev["applicable"],
            "actual_bad": e["return_G"] < med - 0.15,
        })
    tp = sum(1 for r in rows if r["fired"] and r["actual_bad"])
    fp = sum(1 for r in rows if r["fired"] and not r["actual_bad"])
    fn = sum(1 for r in rows if (not r["fired"]) and r["actual_bad"])
    tn = sum(1 for r in rows if (not r["fired"]) and not r["actual_bad"])
    rec = tp / max(tp + fn, 1)
    pre = tp / max(tp + fp, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": rec, "precision": pre,
            "median_G": med, "n_rows": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_root", default=os.environ.get(
        "AGENTICRLHARNESS_DATA",
        "./data") + "/synthetic_tasks")
    ap.add_argument("--n_rollouts_collect", type=int, default=20)
    ap.add_argument("--n_rollouts_eval", type=int, default=3)
    ap.add_argument("--n_eval_seeds", type=int, default=3)
    ap.add_argument("--collect_seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--eps", type=float, default=0.25)
    args = ap.parse_args()

    task_root = Path(args.task_root)
    tasks = load_coding_tasks(task_root)
    print(f"[anchor_5] loaded {len(tasks)} coding tasks from {task_root}")

    client = LLMClient()
    out_dir = THIS.parent
    t0 = time.monotonic()

    print(f"[anchor_5] collecting {args.n_rollouts_collect} rollouts × {len(tasks)} tasks = "
          f"{args.n_rollouts_collect * len(tasks)} traj, eps={args.eps}", flush=True)
    buffer, collect_llm_cost = collect_buffer(
        tasks, client,
        n_rollouts=args.n_rollouts_collect, eps=args.eps, seed=args.collect_seed,
    )
    print(f"  buffer size={len(buffer)}  llm_cost=${collect_llm_cost:.4f}  "
          f"elapsed={time.monotonic()-t0:.1f}s", flush=True)

    with (out_dir / "anchor_5_buffer.jsonl").open("w") as f:
        for ent in buffer:
            f.write(json.dumps({
                "task_id": ent["task_id"],
                "rollout": ent["rollout"],
                "return_G": ent["return_G"],
                "structural_score": ent["structural_score"],
                "n_steps": len(ent["records"]),
                "term_reason": (ent["records"][-1].get("termination_reason")
                                if ent["records"] else None),
            }) + "\n")
    with (out_dir / "anchor_5_buffer_records.jsonl").open("w") as f:
        for ent in buffer:
            f.write(json.dumps({
                "task_id": ent["task_id"], "rollout": ent["rollout"],
                "return_G": ent["return_G"],
                "records": ent["records"],
            }, default=str) + "\n")

    buf_Gs = [e["return_G"] for e in buffer]
    buf_mean = statistics.fmean(buf_Gs)
    buf_std = statistics.stdev(buf_Gs) if len(buf_Gs) > 1 else 0.0
    buf_unique = len(set(round(g, 6) for g in buf_Gs))
    buf_sat_at_1 = sum(1 for g in buf_Gs if g >= 0.999) / len(buf_Gs)
    print(f"  buffer G: mean={buf_mean:.4f} std={buf_std:.4f} unique={buf_unique} "
          f"sat@1.0={buf_sat_at_1:.2%}", flush=True)

    n_criteria_by_task = {t.task_id: len(t.rubric.get("criteria", [])) for t in tasks}
    aw_polices = []
    aw_diags = []
    for seed in range(args.n_eval_seeds):
        cfg = AWConfig(epochs=args.epochs, seed=seed)
        policy, diag = train_aw(buffer, n_criteria_by_task, cfg)
        aw_polices.append(policy)
        aw_diags.append(diag)
        ws = diag["weight_stats"]
        print(f"  AW seed={seed}: final_loss={diag['loss_per_epoch'][-1]:.4f}  "
              f"n_steps={diag['n_steps']}  w_mean={ws['mean']:.3f} "
              f"w_max={ws['max']:.3f} w_std={ws['std']:.3f}", flush=True)

    (out_dir / "anchor_5_training.json").write_text(
        json.dumps({"diagnostics": aw_diags}, indent=2, default=str))

    print(f"[anchor_5] evaluating Base + AW on {len(tasks)} tasks × "
          f"{args.n_rollouts_eval} rollouts × {args.n_eval_seeds} seeds each",
          flush=True)

    base_factory = lambda s: base_harness_policy()
    base_summary, base_llm_cost, base_records = eval_policy(
        "base", tasks, base_factory, client,
        n_rollouts=args.n_rollouts_eval, n_seeds=args.n_eval_seeds,
    )
    aw_factory = lambda s: mlp_policy(aw_polices[s], greedy=False)
    aw_summary, aw_llm_cost, aw_records = eval_policy(
        "aw", tasks, aw_factory, client,
        n_rollouts=args.n_rollouts_eval, n_seeds=args.n_eval_seeds,
    )
    base_mean, base_std = base_summary["mean_over_seeds"], base_summary["std_over_seeds"]
    aw_mean, aw_std = aw_summary["mean_over_seeds"], aw_summary["std_over_seeds"]
    delta = aw_mean - base_mean
    pass_5pp = delta >= 0.05

    welch = welch_t_two_sample(aw_summary["per_seed_overall"],
                                base_summary["per_seed_overall"])

    print(f"[anchor_5] computing HMS for {len(base_records)} Base + "
          f"{len(aw_records)} AW eval rollouts", flush=True)
    base_hms = hms_summary(base_records)
    aw_hms = hms_summary(aw_records)

    c7_calib = calibrate_c7_threshold(buffer, {})

    elapsed = time.monotonic() - t0
    total_llm_cost = collect_llm_cost + base_llm_cost + aw_llm_cost

    summary = {
        "timestamp_jst": time.strftime("%Y-%m-%dT%H:%M:%S+09:00", time.localtime()),
        "fork": "alpha (G = structural rubric_score_norm; readme §14 verifier)",
        "config": vars(args),
        "n_tasks": len(tasks),
        "buffer_size": len(buffer),
        "buffer_G": {
            "mean": buf_mean, "std": buf_std,
            "unique_values": buf_unique,
            "saturation_at_1.0": buf_sat_at_1,
            "min": min(buf_Gs), "max": max(buf_Gs),
        },
        "base": {"mean": base_mean, "std": base_std,
                  "per_seed": base_summary["per_seed_overall"],
                  "per_task_mean_over_seeds": {
                      tid: statistics.fmean(v) if v else None
                      for tid, v in base_summary["per_seed_per_task"].items()
                  }},
        "aw":   {"mean": aw_mean,   "std": aw_std,
                  "per_seed": aw_summary["per_seed_overall"],
                  "per_task_mean_over_seeds": {
                      tid: statistics.fmean(v) if v else None
                      for tid, v in aw_summary["per_seed_per_task"].items()
                  }},
        "delta_mean": delta,
        "welch_t_test": welch,
        "pass_5pp": pass_5pp,
        "hms_base": base_hms,
        "hms_aw": aw_hms,
        "hms_delta_mean": (aw_hms["hms_norm_mean"] - base_hms["hms_norm_mean"]
                            if base_hms["hms_norm_mean"] is not None else None),
        "llm_cost_collect_usd": collect_llm_cost,
        "llm_cost_eval_base_usd": base_llm_cost,
        "llm_cost_eval_aw_usd": aw_llm_cost,
        "total_llm_cost_usd": total_llm_cost,
        "elapsed_s": elapsed,
        "c7_calibration_on_behavioral": c7_calib,
    }
    out_path = out_dir / "anchor_5_results.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n[anchor_5] Base G mean={base_mean:.4f} ± {base_std:.4f}  per_seed={base_summary['per_seed_overall']}")
    print(f"[anchor_5] AW   G mean={aw_mean:.4f} ± {aw_std:.4f}  per_seed={aw_summary['per_seed_overall']}")
    print(f"[anchor_5] ΔG  = {delta:+.4f}  (PASS ≥ 0.05? {pass_5pp})")
    print(f"[anchor_5] Welch t={welch.get('t')!s}  p={welch.get('p_two_sided')!s}")
    print(f"[anchor_5] Base HMS mean={base_hms['hms_norm_mean']!s}  AW HMS mean={aw_hms['hms_norm_mean']!s}  "
          f"ΔHMS={summary['hms_delta_mean']!s}")
    print(f"[anchor_5] C.7 calib on behavioral: {c7_calib}")
    print(f"[anchor_5] total LLM cost = ${total_llm_cost:.4f}  elapsed={elapsed:.1f}s  -> {out_path}", flush=True)
    return 0 if pass_5pp else 2


if __name__ == "__main__":
    sys.exit(main())
