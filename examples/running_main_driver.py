"""RUNNING_MAIN driver — per-domain Base vs AW main-table experiment.

Director-approved (2026-06-11 04:04 UTC) Fork-α':
  - Universal structural verifier (readme §14): G = submission.py
    rubric_score_norm for ALL 6 domains. No LLM judge in train or eval.
  - 100 task / domain (data/synthetic_tasks_main/{domain}/)
  - stratified 80/20 train/eval split by metadata.difficulty (easy/standard/hard)
  - 20 rollouts/task in train buffer (1600 collect/policy)
  - 3 seeds × 3 rollouts/task on eval set (180 eval/policy)
  - Base vs Offline AW (perturbed ε=0.25 behavioral)
  - paired bootstrap p-value on rollout-level (n = n_eval_task × n_rollouts × n_seeds)
  - reports: ΔG mean ± std, ΔHMS, per-event fired-rate, buffer + weight diag

Usage:
  python running_main_driver.py --domain knowledge_work
  python running_main_driver.py --domain coding
  ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1]))

from harness.actions import Task
from harness.util.llm_client import LLMClient

# Per-domain harness modules
from rl import coding_harness as cod
from rl import kw_harness as kw
from rl import generic_harness as gen

from rl.offline_aw import AWConfig, train_aw
from rl.state_features import (
    CODING_ACTION_SPACE, ACTION_TO_IDX as COD_ACTION_TO_IDX,
    trajectory_to_features as cod_traj_feat,
    STATE_DIM as COD_STATE_DIM,
)
from rl.kw_state_features import (
    KW_ACTION_SPACE, ACTION_TO_IDX as KW_ACTION_TO_IDX,
    trajectory_to_features as kw_traj_feat,
    STATE_DIM as KW_STATE_DIM,
)
from modules.hms_detector import compute_hms, EVENT_CHECKERS


def _generic_factory(domain: str):
    """Build per-domain config for the generic harness."""
    return {
        "run_episode": lambda task, policy_fn, *, client, rng,
                              max_steps_override=None: gen.run_episode_with_policy(
            domain, task, policy_fn, client=client, rng=rng,
            max_steps_override=max_steps_override),
        "base_policy": lambda: gen.base_harness_policy(domain),
        "perturbed_policy": lambda eps, seed: gen.perturbed_base_policy(domain, eps=eps, seed=seed),
        "mlp_policy": lambda net, greedy=False: gen.mlp_policy(domain, net, greedy=greedy),
        "action_space": gen.domain_action_space(domain),
        "state_dim": 10 + gen.PAD_ACTION_DIM,
        "traj_featurizer": (lambda d=domain: lambda recs, n_crit:
                            gen.trajectory_to_features(recs, n_crit, d))(),
    }


# ── Per-domain config ────────────────────────────────────────────────────

DOMAIN_CONFIG = {
    "coding": {
        "run_episode": cod.run_episode_with_policy,
        "base_policy": cod.base_harness_policy,
        "perturbed_policy": cod.perturbed_base_policy,
        "mlp_policy": cod.mlp_policy,
        "action_space": CODING_ACTION_SPACE,
        "state_dim": COD_STATE_DIM,
        "traj_featurizer": lambda recs, n_crit: cod_traj_feat(recs, action_space=None, n_criteria=n_crit),
    },
    "knowledge_work": {
        "run_episode": kw.run_episode_with_policy,
        "base_policy": kw.base_harness_policy,
        "perturbed_policy": kw.perturbed_base_policy,
        "mlp_policy": kw.mlp_policy,
        "action_space": KW_ACTION_SPACE,
        "state_dim": KW_STATE_DIM,
        "traj_featurizer": lambda recs, n_crit: kw_traj_feat(recs, n_criteria=n_crit),
    },
    "research": _generic_factory("research"),
    "multi_tool": _generic_factory("multi_tool"),
    "long_memory": _generic_factory("long_memory"),
    "planning": _generic_factory("planning"),
    # τ-bench retail tasks adapted to KW deliverable format
    "tau_bench_retail": {
        "run_episode": kw.run_episode_with_policy,
        "base_policy": kw.base_harness_policy,
        "perturbed_policy": kw.perturbed_base_policy,
        "mlp_policy": kw.mlp_policy,
        "action_space": KW_ACTION_SPACE,
        "state_dim": KW_STATE_DIM,
        "traj_featurizer": lambda recs, n_crit: kw_traj_feat(recs, n_criteria=n_crit),
    },
    "tau_bench_retail_v2_heldout": {
        "run_episode": kw.run_episode_with_policy,
        "base_policy": kw.base_harness_policy,
        "perturbed_policy": kw.perturbed_base_policy,
        "mlp_policy": kw.mlp_policy,
        "action_space": KW_ACTION_SPACE,
        "state_dim": KW_STATE_DIM,
        "traj_featurizer": lambda recs, n_crit: kw_traj_feat(recs, n_criteria=n_crit),
    },
    "agentbench_dbbench": {
        "run_episode": kw.run_episode_with_policy,
        "base_policy": kw.base_harness_policy,
        "perturbed_policy": kw.perturbed_base_policy,
        "mlp_policy": kw.mlp_policy,
        "action_space": KW_ACTION_SPACE,
        "state_dim": KW_STATE_DIM,
        "traj_featurizer": lambda recs, n_crit: kw_traj_feat(recs, n_criteria=n_crit),
    },
}

# Mapping domain → metadata.task_type prefix for loading from
# synthetic_tasks_main/{domain}/{domain}_NNN/.
TASK_ROOT = (Path(os.environ.get("AGENTICRLHARNESS_DATA", "./data")) / "/synthetic_tasks_main".lstrip("/"))


def load_domain_tasks(domain: str) -> list[Task]:
    """Load all tasks for a domain in numerical id order."""
    domain_dir = TASK_ROOT / domain
    if not domain_dir.exists():
        raise FileNotFoundError(domain_dir)
    tids = sorted(p.name for p in domain_dir.iterdir() if p.is_dir())
    tasks = []
    for tid in tids:
        td = domain_dir / tid
        if (td / "task.json").exists():
            tasks.append(Task.load(td))
    return tasks


def stratified_split(tasks: list[Task], train_frac: float = 0.8,
                     seed: int = 42) -> tuple[list[Task], list[Task]]:
    """Stratify by metadata.difficulty (easy/standard/hard)."""
    rng = random.Random(seed)
    strata: dict[str, list[Task]] = {}
    for t in tasks:
        diff = t.metadata.get("difficulty", "standard")
        strata.setdefault(diff, []).append(t)
    train: list[Task] = []
    eval_set: list[Task] = []
    for diff, lst in strata.items():
        rng.shuffle(lst)
        n_train = max(1, int(len(lst) * train_frac))
        train.extend(lst[:n_train])
        eval_set.extend(lst[n_train:])
    return train, eval_set


def collect_buffer(domain: str, tasks: list[Task], client: LLMClient,
                   *, n_rollouts: int, eps: float, seed: int,
                   progress_every: int = 200) -> tuple[list[dict], float]:
    cfg = DOMAIN_CONFIG[domain]
    behavioral = cfg["perturbed_policy"](eps=eps, seed=seed)
    buffer = []
    llm_cost = 0.0
    n_total = len(tasks) * n_rollouts
    t0 = time.monotonic()
    cnt = 0
    for tid_idx, task in enumerate(tasks):
        for r in range(n_rollouts):
            rng = random.Random((seed + 1) * 1000 + tid_idx * 100 + r)
            logger, scored = cfg["run_episode"](
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
            cnt += 1
            if cnt % progress_every == 0:
                el = time.monotonic() - t0
                print(f"    collect {cnt}/{n_total}  elapsed={el:.0f}s  cost=${llm_cost:.4f}",
                      flush=True)
    return buffer, llm_cost


def eval_policy(domain: str, label: str, tasks: list[Task],
                policy_fn_factory: Callable, client: LLMClient,
                *, n_rollouts: int, n_seeds: int,
                progress_every: int = 60) -> tuple[dict, float, list[dict]]:
    cfg = DOMAIN_CONFIG[domain]
    per_seed_overall: list[float] = []
    per_seed_per_task: dict[str, list[float]] = {}
    llm_cost = 0.0
    detail = []
    all_records: list[dict] = []
    n_total = n_seeds * len(tasks) * n_rollouts
    t0 = time.monotonic()
    cnt = 0
    for s in range(n_seeds):
        seed_scores = []
        seed_per_task: dict[str, list[float]] = {}
        policy = policy_fn_factory(s)
        for task in tasks:
            seed_per_task.setdefault(task.task_id, [])
            for r in range(n_rollouts):
                rng = random.Random((s + 1) * 9999 + r * 13 + hash(task.task_id) % 97)
                logger, scored = cfg["run_episode"](
                    task, policy, client=client, rng=rng,
                )
                llm_cost += logger.total_cost
                G = scored["rubric_score_norm"]
                seed_scores.append(G)
                seed_per_task[task.task_id].append(G)
                detail.append({"label": label, "seed": s,
                                "task_id": task.task_id, "rollout": r,
                                "G": G,
                                "submit": logger.records[-1].get("termination_reason"),
                                "n_steps": len(logger.records)})
                all_records.append({"label": label, "seed": s,
                                     "task_id": task.task_id, "rollout": r,
                                     "records": logger.records, "G": G})
                cnt += 1
                if cnt % progress_every == 0:
                    el = time.monotonic() - t0
                    print(f"    eval[{label}] {cnt}/{n_total}  elapsed={el:.0f}s  cost=${llm_cost:.4f}",
                          flush=True)
        per_seed_overall.append(statistics.fmean(seed_scores))
        for tid, lst in seed_per_task.items():
            per_seed_per_task.setdefault(tid, []).append(
                statistics.fmean(lst) if lst else None)
    return {
        "label": label,
        "per_seed_overall": per_seed_overall,
        "mean_over_seeds": statistics.fmean(per_seed_overall),
        "std_over_seeds": statistics.stdev(per_seed_overall) if len(per_seed_overall) > 1 else 0.0,
        "per_seed_per_task": per_seed_per_task,
        "detail": detail,
    }, llm_cost, all_records


def paired_bootstrap(base_Gs: list[float], aw_Gs: list[float],
                      n_resamples: int = 5000,
                      seed: int = 0) -> dict:
    """Rollout-level paired bootstrap on Δ = aw - base.

    Inputs are aligned lists of length n; output:
       mean_delta, p_two_sided (H0: Δ=0), ci95.
    """
    assert len(base_Gs) == len(aw_Gs)
    n = len(base_Gs)
    if n < 2:
        return {"n": n, "mean_delta": None, "p_two_sided": None}
    deltas = [aw_Gs[i] - base_Gs[i] for i in range(n)]
    obs_mean = sum(deltas) / n
    rng = random.Random(seed)
    centered = [d - obs_mean for d in deltas]
    # H0: Δ=0; bootstrap centered deltas
    null_means = []
    for _ in range(n_resamples):
        s = sum(centered[rng.randrange(n)] for _ in range(n)) / n
        null_means.append(s)
    p = sum(1 for m in null_means if abs(m) >= abs(obs_mean)) / n_resamples
    # 95% CI from non-centered bootstrap
    boot_means = []
    for _ in range(n_resamples):
        s = sum(deltas[rng.randrange(n)] for _ in range(n)) / n
        boot_means.append(s)
    boot_means.sort()
    lo = boot_means[int(0.025 * n_resamples)]
    hi = boot_means[int(0.975 * n_resamples)]
    return {"n": n, "mean_delta": obs_mean, "p_two_sided": p,
            "ci95_lo": lo, "ci95_hi": hi,
            "n_resamples": n_resamples}


def hms_summary(all_records: list[dict]) -> dict:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=list(DOMAIN_CONFIG.keys()))
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--n_rollouts_collect", type=int, default=20)
    ap.add_argument("--n_rollouts_eval", type=int, default=3)
    ap.add_argument("--n_eval_seeds", type=int, default=3)
    ap.add_argument("--collect_seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--eps", type=float, default=0.25)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--output_tag", type=str, default="")
    ap.add_argument("--resume_buffer", type=str, default=None,
                    help="Path to buffer_records.jsonl to skip collect step.")
    ap.add_argument("--task_root_override", type=str, default=None,
                    help="Override TASK_ROOT (for non-standard datasets like τ-bench).")
    ap.add_argument("--n_rollouts_collect_per_task", type=int, default=None,
                    help="Alias for --n_rollouts_collect when small dataset.")
    ap.add_argument("--no_split", action="store_true",
                    help="Skip stratified split; use all tasks for both train and eval.")
    ap.add_argument("--eval_task_ids", type=str, default=None,
                    help="Comma-separated explicit task IDs for eval (overrides split). All other tasks go to train.")
    args = ap.parse_args()
    if args.task_root_override:
        global TASK_ROOT
        TASK_ROOT = Path(args.task_root_override)

    cfg = DOMAIN_CONFIG[args.domain]
    client = LLMClient()
    t0 = time.monotonic()

    # ── Step 0: load + stratified split ──
    all_tasks = load_domain_tasks(args.domain)
    if args.eval_task_ids:
        eval_ids = set(args.eval_task_ids.split(","))
        eval_tasks = [t for t in all_tasks if t.task_id in eval_ids]
        train_tasks = [t for t in all_tasks if t.task_id not in eval_ids]
        print(f"[main:{args.domain}] loaded {len(all_tasks)} tasks  "
              f"train={len(train_tasks)} eval={len(eval_tasks)}  "
              f"(explicit eval_task_ids)",
              flush=True)
    elif args.no_split:
        train_tasks = list(all_tasks)
        eval_tasks = list(all_tasks)
        print(f"[main:{args.domain}] loaded {len(all_tasks)} tasks (no_split: all in both train+eval)", flush=True)
    else:
        train_tasks, eval_tasks = stratified_split(all_tasks, args.train_frac,
                                                    seed=args.split_seed)
        print(f"[main:{args.domain}] loaded {len(all_tasks)} tasks  "
              f"train={len(train_tasks)} eval={len(eval_tasks)}  "
              f"(split_seed={args.split_seed}, train_frac={args.train_frac})",
              flush=True)

    out_dir = THIS.parent / f"main_{args.domain}{args.output_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: collect behavioral buffer (or resume) ──
    if args.resume_buffer:
        print(f"[main:{args.domain}] resuming from buffer_records {args.resume_buffer}",
              flush=True)
        buffer = []
        with open(args.resume_buffer) as f:
            for line in f:
                d = json.loads(line)
                buffer.append(d)
        collect_cost = 0.0
        print(f"  loaded {len(buffer)} from disk; skipping collect", flush=True)
    else:
        print(f"[main:{args.domain}] collecting {args.n_rollouts_collect}/task × "
              f"{len(train_tasks)} = {args.n_rollouts_collect*len(train_tasks)} traj, "
              f"eps={args.eps}", flush=True)
        buffer, collect_cost = collect_buffer(
            args.domain, train_tasks, client,
            n_rollouts=args.n_rollouts_collect, eps=args.eps,
            seed=args.collect_seed,
        )
        print(f"  buffer size={len(buffer)}  llm_cost=${collect_cost:.4f}  "
              f"elapsed={time.monotonic()-t0:.1f}s", flush=True)
        # Write buffer summaries
        with (out_dir / "buffer.jsonl").open("w") as f:
            for ent in buffer:
                f.write(json.dumps({
                    "task_id": ent["task_id"], "rollout": ent["rollout"],
                    "return_G": ent["return_G"],
                    "structural_score": ent["structural_score"],
                    "n_steps": len(ent["records"]),
                    "term_reason": (ent["records"][-1].get("termination_reason")
                                    if ent["records"] else None),
                }) + "\n")
        # Write full records buffer for resume
        with (out_dir / "buffer_records.jsonl").open("w") as f:
            for ent in buffer:
                f.write(json.dumps({
                    "task_id": ent["task_id"], "rollout": ent["rollout"],
                    "return_G": ent["return_G"],
                    "structural_score": ent["structural_score"],
                    "records": ent["records"],
                }, default=str) + "\n")

    buf_Gs = [e["return_G"] for e in buffer]
    buf_mean = statistics.fmean(buf_Gs)
    buf_std = statistics.stdev(buf_Gs) if len(buf_Gs) > 1 else 0.0
    buf_unique = len(set(round(g, 6) for g in buf_Gs))
    buf_sat_at_1 = sum(1 for g in buf_Gs if g >= 0.999) / len(buf_Gs)
    print(f"  buffer G: mean={buf_mean:.4f}  std={buf_std:.4f}  "
          f"unique={buf_unique}  sat@1.0={buf_sat_at_1:.2%}", flush=True)

    # ── Step 2: train AW 3 seeds ──
    n_criteria_by_task = {t.task_id: len(t.rubric.get("criteria", []))
                          for t in train_tasks}
    aw_policies = []
    aw_diags = []
    for seed in range(args.n_eval_seeds):
        aw_cfg = AWConfig(epochs=args.epochs, seed=seed)
        policy, diag = train_aw(
            buffer, n_criteria_by_task, aw_cfg,
            traj_featurizer=cfg["traj_featurizer"],
        )
        aw_policies.append(policy)
        aw_diags.append(diag)
        ws = diag["weight_stats"]
        print(f"  AW seed={seed}: final_loss={diag['loss_per_epoch'][-1]:.4f}  "
              f"n_steps={diag['n_steps']}  w_mean={ws['mean']:.3f} "
              f"w_max={ws['max']:.3f} w_std={ws['std']:.3f}", flush=True)
    (out_dir / "training.json").write_text(
        json.dumps({"diagnostics": aw_diags}, indent=2, default=str))

    # ── Step 3: eval Base + AW on EVAL set (stratified) ──
    print(f"[main:{args.domain}] eval on {len(eval_tasks)} tasks × "
          f"{args.n_rollouts_eval} rollouts × {args.n_eval_seeds} seeds = "
          f"{len(eval_tasks)*args.n_rollouts_eval*args.n_eval_seeds}/policy",
          flush=True)
    base_factory = lambda s: cfg["base_policy"]()
    base_summary, base_cost, base_records = eval_policy(
        args.domain, "base", eval_tasks, base_factory, client,
        n_rollouts=args.n_rollouts_eval, n_seeds=args.n_eval_seeds,
    )
    aw_factory = lambda s: cfg["mlp_policy"](aw_policies[s], greedy=False)
    aw_summary, aw_cost, aw_records = eval_policy(
        args.domain, "aw", eval_tasks, aw_factory, client,
        n_rollouts=args.n_rollouts_eval, n_seeds=args.n_eval_seeds,
    )

    base_mean = base_summary["mean_over_seeds"]
    base_std = base_summary["std_over_seeds"]
    aw_mean = aw_summary["mean_over_seeds"]
    aw_std = aw_summary["std_over_seeds"]
    delta = aw_mean - base_mean

    # paired bootstrap (align by (task, rollout, seed))
    base_Gs = sorted(base_summary["detail"],
                     key=lambda r: (r["seed"], r["task_id"], r["rollout"]))
    aw_Gs = sorted(aw_summary["detail"],
                   key=lambda r: (r["seed"], r["task_id"], r["rollout"]))
    boot = paired_bootstrap([r["G"] for r in base_Gs],
                             [r["G"] for r in aw_Gs],
                             n_resamples=5000, seed=0)
    pass_5pp = (delta >= 0.05) and (boot.get("p_two_sided") is not None
                                     and boot["p_two_sided"] < 0.05)

    # eval-set saturation
    eval_all_Gs = [r["G"] for r in base_summary["detail"]] + \
                  [r["G"] for r in aw_summary["detail"]]
    sat_eval = sum(1 for g in eval_all_Gs if g >= 0.999) / max(len(eval_all_Gs), 1)

    # HMS aggregation
    base_hms = hms_summary(base_records)
    aw_hms = hms_summary(aw_records)

    elapsed = time.monotonic() - t0
    total_cost = collect_cost + base_cost + aw_cost

    summary = {
        "timestamp_jst": time.strftime("%Y-%m-%dT%H:%M:%S+09:00", time.localtime()),
        "domain": args.domain,
        "fork": "alpha-prime (universal structural verifier)",
        "config": vars(args),
        "n_train_tasks": len(train_tasks),
        "n_eval_tasks": len(eval_tasks),
        "n_total_tasks": len(all_tasks),
        "stratified_split_diff_counts": {
            "train": dict(_count_diff(train_tasks)),
            "eval": dict(_count_diff(eval_tasks)),
        },
        "buffer": {
            "size": len(buffer),
            "G_mean": buf_mean, "G_std": buf_std,
            "G_unique": buf_unique,
            "G_saturation_at_1.0": buf_sat_at_1,
            "G_min": min(buf_Gs), "G_max": max(buf_Gs),
        },
        "base": {"mean": base_mean, "std": base_std,
                  "per_seed": base_summary["per_seed_overall"]},
        "aw": {"mean": aw_mean, "std": aw_std,
                "per_seed": aw_summary["per_seed_overall"]},
        "delta_mean": delta,
        "paired_bootstrap": boot,
        "eval_saturation_at_1.0": sat_eval,
        "pass_invariant_5pp_p<0.05": pass_5pp,
        "hms_base": base_hms,
        "hms_aw": aw_hms,
        "hms_delta_mean": (aw_hms["hms_norm_mean"] - base_hms["hms_norm_mean"]
                            if base_hms["hms_norm_mean"] is not None
                            and aw_hms["hms_norm_mean"] is not None else None),
        "llm_cost_collect_usd": collect_cost,
        "llm_cost_eval_base_usd": base_cost,
        "llm_cost_eval_aw_usd": aw_cost,
        "total_llm_cost_usd": total_cost,
        "elapsed_s": elapsed,
    }
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))

    # Per-task detail (for paper supp)
    (out_dir / "eval_detail_base.jsonl").write_text(
        "\n".join(json.dumps(d, default=str) for d in base_summary["detail"]))
    (out_dir / "eval_detail_aw.jsonl").write_text(
        "\n".join(json.dumps(d, default=str) for d in aw_summary["detail"]))
    # Full eval records (B.1 trajectories) for HMS re-analysis
    with (out_dir / "eval_records_base.jsonl").open("w") as f:
        for entry in base_records:
            f.write(json.dumps({"label": entry["label"], "seed": entry["seed"],
                                  "task_id": entry["task_id"], "rollout": entry["rollout"],
                                  "G": entry["G"], "records": entry["records"]},
                                default=str) + "\n")
    with (out_dir / "eval_records_aw.jsonl").open("w") as f:
        for entry in aw_records:
            f.write(json.dumps({"label": entry["label"], "seed": entry["seed"],
                                  "task_id": entry["task_id"], "rollout": entry["rollout"],
                                  "G": entry["G"], "records": entry["records"]},
                                default=str) + "\n")

    print(f"\n[main:{args.domain}] ===========================================", flush=True)
    print(f"  Base G mean = {base_mean:.4f} ± {base_std:.4f}  per_seed={base_summary['per_seed_overall']}", flush=True)
    print(f"  AW   G mean = {aw_mean:.4f} ± {aw_std:.4f}  per_seed={aw_summary['per_seed_overall']}", flush=True)
    print(f"  ΔG = {delta:+.4f}  paired bootstrap p={boot.get('p_two_sided')}  CI95=({boot.get('ci95_lo')!s},{boot.get('ci95_hi')!s})  n={boot.get('n')}", flush=True)
    print(f"  pass_invariant_5pp_p<0.05 = {pass_5pp}", flush=True)
    print(f"  eval saturation@1.0 = {sat_eval:.2%}", flush=True)
    print(f"  Base HMS = {base_hms['hms_norm_mean']!s}  AW HMS = {aw_hms['hms_norm_mean']!s}  ΔHMS = {summary['hms_delta_mean']!s}", flush=True)
    print(f"  Total LLM cost = ${total_cost:.4f}  elapsed={elapsed:.1f}s  -> {out_path}", flush=True)
    return 0 if pass_5pp else 2


def _count_diff(tasks: list[Task]) -> dict:
    out = {}
    for t in tasks:
        d = t.metadata.get("difficulty", "standard")
        out[d] = out.get(d, 0) + 1
    return out


if __name__ == "__main__":
    sys.exit(main())
