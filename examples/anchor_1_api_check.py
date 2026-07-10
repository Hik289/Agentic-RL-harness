"""Anchor 1: OpenAI-compatible API stability check.

100 serial calls + 10 concurrent calls. Reports:
  * pass / fail counts
  * p50 / p95 / p99 latency
  * mean cost per call
  * total cost
  * failure modes (if any)

Writes JSON to results.json next to this file.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import statistics
import sys
import time
from pathlib import Path

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.util.llm_client import LLMClient, CallResult


PROMPT_POOL = [
    "Reply with just the single word: ok",
    "What is 2+2? Reply with only the digit.",
    "Reply with the word 'pong'.",
    "Reply with: hello",
    "Say only: ready",
]


def one_call(client: LLMClient, idx: int) -> dict:
    msg = PROMPT_POOL[idx % len(PROMPT_POOL)]
    res = client.chat(
        messages=[{"role": "user", "content": msg}],
        max_completion_tokens=8,
    )
    d = res.to_dict()
    d["idx"] = idx
    d["prompt"] = msg
    return d


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def run_serial(client: LLMClient, n: int) -> list[dict]:
    out = []
    for i in range(n):
        out.append(one_call(client, i))
    return out


def run_concurrent(client: LLMClient, n: int, workers: int) -> list[dict]:
    out: list[dict] = [{} for _ in range(n)]
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one_call, client, i): i for i in range(n)}
        for fut in cf.as_completed(futs):
            i = futs[fut]
            out[i] = fut.result()
    return out


def summarise(label: str, calls: list[dict]) -> dict:
    ok = [c for c in calls if c.get("ok")]
    fail = [c for c in calls if not c.get("ok")]
    lats = [c["latency_s"] for c in ok]
    costs = [c["cost_usd"] for c in ok]
    tok_in = sum(c.get("prompt_tokens", 0) for c in ok)
    tok_out = sum(c.get("completion_tokens", 0) for c in ok)
    summary = {
        "label": label,
        "n": len(calls),
        "ok": len(ok),
        "fail": len(fail),
        "pass_rate": (len(ok) / len(calls)) if calls else 0.0,
        "lat_p50": percentile(lats, 0.50) if lats else None,
        "lat_p95": percentile(lats, 0.95) if lats else None,
        "lat_p99": percentile(lats, 0.99) if lats else None,
        "lat_mean": (sum(lats) / len(lats)) if lats else None,
        "cost_total_usd": sum(costs),
        "cost_mean_per_call": (sum(costs) / len(ok)) if ok else None,
        "tokens_in_total": tok_in,
        "tokens_out_total": tok_out,
        "failure_modes": {},
    }
    if fail:
        modes: dict[str, int] = {}
        for c in fail:
            k = (c.get("error") or "unknown")[:120]
            modes[k] = modes.get(k, 0) + 1
        summary["failure_modes"] = modes
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", type=int, default=100)
    ap.add_argument("--concurrent", type=int, default=10)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parent / "anchor_1_results.json"),
    )
    args = ap.parse_args()

    client = LLMClient()
    t0 = time.monotonic()
    print(f"[anchor_1] endpoint={client.endpoint} deployment={client.deployment}")
    print(f"[anchor_1] running {args.serial} serial calls ...")
    serial = run_serial(client, args.serial)
    s_serial = summarise("serial", serial)
    print(f"[anchor_1] serial ok={s_serial['ok']}/{s_serial['n']} "
          f"p50={s_serial['lat_p50']:.3f}s p95={s_serial['lat_p95']:.3f}s "
          f"mean_cost=${s_serial['cost_mean_per_call']:.6f}")

    print(f"[anchor_1] running {args.concurrent} concurrent calls ({args.workers} workers) ...")
    par = run_concurrent(client, args.concurrent, args.workers)
    s_par = summarise("concurrent", par)
    print(f"[anchor_1] concurrent ok={s_par['ok']}/{s_par['n']} "
          f"p50={s_par['lat_p50']:.3f}s p95={s_par['lat_p95']:.3f}s "
          f"mean_cost=${s_par['cost_mean_per_call']:.6f}")

    total_cost = s_serial["cost_total_usd"] + s_par["cost_total_usd"]
    total_n = s_serial["n"] + s_par["n"]
    total_ok = s_serial["ok"] + s_par["ok"]
    elapsed = time.monotonic() - t0

    blob = {
        "timestamp_jst": time.strftime("%Y-%m-%dT%H:%M:%S+09:00", time.localtime()),
        "endpoint": client.endpoint,
        "deployment": client.deployment,
        "elapsed_s": elapsed,
        "serial": s_serial,
        "concurrent": s_par,
        "total": {
            "n": total_n,
            "ok": total_ok,
            "pass_rate": total_ok / total_n,
            "cost_total_usd": total_cost,
        },
        "calls_serial": serial,
        "calls_concurrent": par,
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(blob, indent=2))
    print(f"[anchor_1] wrote {out_path}  total_cost=${total_cost:.4f} pass_rate={total_ok}/{total_n}")
    return 0 if total_ok == total_n else 2


if __name__ == "__main__":
    sys.exit(main())
