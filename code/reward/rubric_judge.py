"""Criterion-level Rubric Judge (readme §13).

Given (task, rubric, candidate_output, optional inputs/observations), the
judge returns a JSON dict:

  {
    "total_score": float,
    "normalized_score": float in [0,1],
    "criteria_scores": [
      {criterion_id, score, max_score, explanation, missing_items: [...]}
    ],
    "unsupported_or_unverified_units": [...],
    "major_errors": [...]
  }

Implementation: ONE LLM call per task that scores all criteria together
(cheaper than per-criterion, and judge sees full criteria context).

The judge is deterministic-ish (low temperature, n_repeats=1 default;
n_repeats>1 averages criterion scores for higher reliability).

## CRITICAL MODE BOUNDARY (Director 2026-06-10 23:48 UTC) ##

The judge runs in two mutually exclusive modes:

  * mode="reward" (PRODUCTION DEFAULT):
      The judge does NOT see reference/answer.md.  Reward must remain
      rooted in the rubric criteria themselves, not in
      similarity-to-reference.  This preserves the paper's Class R reward
      property (theorist Thm 1 + paper §6 reward-hacking discussion).  All
      RL training / harness scoring / main-results eval uses this mode.

  * mode="calibration":
      Only for annotator-agreement evaluation (anchor_3 style).  The judge
      DOES see reference/answer.md because human annotators saw it too.
      Must never be used to compute reward during training or main-results
      eval.

Pass `mode=` to RubricJudge.score(); otherwise the constructor
`default_mode` is used (default_mode="reward").

For criteria that genuinely require numerical comparison against a
reference value (e.g. multi_tool quantitative answers) the long-term
solution is a task-specific verifier (readme §14), not relaxing this mode
boundary. — see paper §6 limitations.
"""
from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass
from typing import Any, Optional

from harness.util.llm_client import LLMClient, get_default_client

SYSTEM_PROMPT = (
    "You are a strict, numeric rubric judge. Score each criterion "
    "independently. Return a SINGLE JSON object only — no markdown fences, "
    "no commentary. Be calibrated: give 0 to clear misses, mid scores for "
    "partial, full marks only for clearly meeting the criterion."
)


def _build_prompt(task: dict, rubric: dict, candidate_output: str,
                  inputs_summary: str = "") -> str:
    crit_lines = []
    for c in rubric.get("criteria", []):
        crit_lines.append(
            f"- id={c['id']}  max={c.get('max_score')}  desc={c.get('description')}"
        )
    out_keys_example = []
    for c in rubric.get("criteria", []):
        out_keys_example.append({
            "criterion_id": c["id"],
            "score": "<float in [0, max_score]>",
            "max_score": c.get("max_score"),
            "explanation": "<one short sentence>",
            "missing_items": [],
        })
    schema_example = {
        "total_score": "<float>",
        "normalized_score": "<float in [0,1]>",
        "criteria_scores": out_keys_example,
        "unsupported_or_unverified_units": [],
        "major_errors": [],
    }
    user_msg = (
        f"TASK PROMPT:\n{task.get('prompt','')}\n\n"
        f"TASK TYPE: {task.get('task_type','')}\n\n"
        f"AVAILABLE INPUTS (summarized):\n{inputs_summary or '(none provided)'}\n\n"
        f"RUBRIC CRITERIA:\n" + "\n".join(crit_lines) + "\n\n"
        f"CANDIDATE OUTPUT:\n```\n{candidate_output}\n```\n\n"
        "Score every criterion. Output ONLY this JSON (matching this schema):\n"
        f"{json.dumps(schema_example, indent=2)}"
    )
    return user_msg


def _parse_judge_json(text: str) -> dict | None:
    if not text:
        return None
    # find first {...} block, balanced
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s).rstrip("`").strip()
    # naive bracket extract
    depth = 0
    start = None
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                blob = s[start:i + 1]
                try:
                    return json.loads(blob)
                except Exception:
                    pass
    try:
        return json.loads(s)
    except Exception:
        return None


@dataclass
class JudgeReport:
    ok: bool
    total_score: float = 0.0
    normalized_score: float = 0.0
    criteria_scores: list = None
    missing_items: list = None
    unsupported_or_unverified_units: list = None
    major_errors: list = None
    raw_text: str = ""
    cost_usd: float = 0.0
    latency_s: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "total_score": self.total_score,
            "normalized_score": self.normalized_score,
            "criteria_scores": self.criteria_scores or [],
            "missing_items": self.missing_items or [],
            "unsupported_or_unverified_units": self.unsupported_or_unverified_units or [],
            "major_errors": self.major_errors or [],
            "cost_usd": self.cost_usd,
            "latency_s": self.latency_s,
            "error": self.error,
        }


VALID_MODES = ("reward", "calibration")


class RubricJudge:
    def __init__(self, client: LLMClient | None = None,
                 n_repeats: int = 1,
                 max_completion_tokens: int = 700,
                 default_mode: str = "reward"):
        if default_mode not in VALID_MODES:
            raise ValueError(f"default_mode must be in {VALID_MODES}")
        self.client = client or get_default_client()
        self.n_repeats = n_repeats
        self.max_completion_tokens = max_completion_tokens
        self.default_mode = default_mode

    @staticmethod
    def _strip_reference(inputs_summary: str) -> str:
        """Remove any '--- REFERENCE ...' chunks injected by anchor_3 callers."""
        if not inputs_summary:
            return inputs_summary
        out_lines = []
        skip = False
        for line in inputs_summary.splitlines():
            if line.startswith("--- REFERENCE"):
                skip = True
                continue
            if skip and line.startswith("---"):
                skip = False
            if not skip:
                out_lines.append(line)
        return "\n".join(out_lines).strip()

    def _one_call(self, task: dict, rubric: dict, candidate_output: str,
                  inputs_summary: str = "") -> dict | None:
        prompt = _build_prompt(task, rubric, candidate_output, inputs_summary)
        res = self.client.chat(
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}],
            max_completion_tokens=self.max_completion_tokens,
        )
        if not res.ok:
            return {"_error": res.error or "llm fail", "_text": "",
                    "_cost": 0.0, "_lat": res.latency_s}
        parsed = _parse_judge_json(res.text or "")
        if parsed is None:
            return {"_error": "json parse failed",
                    "_text": res.text or "",
                    "_cost": res.cost_usd,
                    "_lat": res.latency_s}
        parsed["_cost"] = res.cost_usd
        parsed["_lat"] = res.latency_s
        parsed["_text"] = res.text or ""
        return parsed

    def score(self, task: dict, rubric: dict, candidate_output: str,
              inputs_summary: str = "",
              mode: str | None = None) -> JudgeReport:
        active_mode = mode or self.default_mode
        if active_mode not in VALID_MODES:
            raise ValueError(f"mode must be in {VALID_MODES}; got {active_mode!r}")
        if active_mode == "reward":
            # PRODUCTION: strip any reference/oracle context that callers
            # may have leaked into inputs_summary. Reward must remain
            # rooted in rubric criteria, not similarity-to-reference.
            inputs_summary = self._strip_reference(inputs_summary)
        responses = []
        total_cost = 0.0
        total_lat = 0.0
        err = None
        for _ in range(max(1, self.n_repeats)):
            r = self._one_call(task, rubric, candidate_output, inputs_summary)
            if r is None:
                continue
            total_cost += r.get("_cost", 0.0)
            total_lat += r.get("_lat", 0.0)
            if "_error" in r:
                err = r["_error"]
                continue
            responses.append(r)
        if not responses:
            return JudgeReport(ok=False, error=err or "no parseable response",
                               cost_usd=total_cost, latency_s=total_lat)

        # Average criterion scores
        crit_id_order = [c["id"] for c in rubric.get("criteria", [])]
        crit_max = {c["id"]: float(c.get("max_score", 1.0)) for c in rubric.get("criteria", [])}
        per_crit_acc: dict = {cid: [] for cid in crit_id_order}
        missing_union: dict = {cid: set() for cid in crit_id_order}
        last_expl: dict = {cid: "" for cid in crit_id_order}
        major_errors: set = set()
        unsupported: set = set()

        for r in responses:
            for cs in r.get("criteria_scores", []) or []:
                cid = cs.get("criterion_id")
                if cid not in per_crit_acc:
                    continue
                try:
                    sc = float(cs.get("score", 0.0))
                except Exception:
                    sc = 0.0
                # clamp
                sc = max(0.0, min(sc, crit_max[cid]))
                per_crit_acc[cid].append(sc)
                last_expl[cid] = cs.get("explanation", "") or last_expl[cid]
                for mi in (cs.get("missing_items") or []):
                    missing_union[cid].add(mi if isinstance(mi, str) else json.dumps(mi, default=str))
            for me in (r.get("major_errors") or []):
                major_errors.add(me if isinstance(me, str) else json.dumps(me, default=str))
            for u in (r.get("unsupported_or_unverified_units") or []):
                unsupported.add(u if isinstance(u, str) else json.dumps(u, default=str))

        crit_out = []
        missing_flat = []
        total_score = 0.0
        max_total = 0.0
        for cid in crit_id_order:
            scs = per_crit_acc[cid]
            if scs:
                mean_sc = statistics.fmean(scs)
            else:
                mean_sc = 0.0
            mx = crit_max[cid]
            entry = {
                "criterion_id": cid,
                "score": mean_sc,
                "max_score": mx,
                "explanation": last_expl[cid],
                "missing_items": list(missing_union[cid]),
            }
            crit_out.append(entry)
            if mean_sc / max(mx, 1e-9) < 0.5:
                missing_flat.append(cid)
            total_score += mean_sc
            max_total += mx
        norm = total_score / max_total if max_total > 0 else 0.0

        return JudgeReport(
            ok=True,
            total_score=total_score,
            normalized_score=norm,
            criteria_scores=crit_out,
            missing_items=missing_flat,
            unsupported_or_unverified_units=list(unsupported),
            major_errors=list(major_errors),
            cost_usd=total_cost,
            latency_s=total_lat,
        )
