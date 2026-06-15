"""Action executors for the Base Harness.

Each action is a pure function (Task, ExecCtx) -> ObservationDict that does
the side effect (read file, query LLM, run code) and returns observation
fields suitable for trajectory_logger.log_step(**obs).

Base Harness uses a fixed scripted policy (no learned controller) -- the
purpose is to verify that every action surface works end-to-end and that the
trajectory is fully populated. The same action set will be used by
RL-controlled harnesses later.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .util.llm_client import LLMClient, get_default_client


@dataclass
class Task:
    task_dir: Path
    task_id: str
    task_type: str
    prompt: str
    available_tools: list
    max_steps: int
    cost_budget: float
    metadata: dict
    rubric: dict
    # Inputs cached on first read
    inputs: dict = field(default_factory=dict)

    @classmethod
    def load(cls, task_dir: str | Path) -> "Task":
        task_dir = Path(task_dir)
        tj = json.loads((task_dir / "task.json").read_text())
        rj = json.loads((task_dir / "rubric.json").read_text())
        mj = json.loads((task_dir / "metadata.json").read_text())
        return cls(
            task_dir=task_dir,
            task_id=tj["task_id"],
            task_type=tj["task_type"],
            prompt=tj["prompt"],
            available_tools=tj.get("available_tools", []),
            max_steps=tj.get("max_steps", 10),
            cost_budget=tj.get("cost_budget", 1.0),
            metadata=mj,
            rubric=rj,
        )

    def load_inputs(self) -> dict:
        """Return a dict {filename: text} for all input files."""
        if self.inputs:
            return self.inputs
        inputs_dir = self.task_dir / "inputs"
        if not inputs_dir.exists():
            return {}
        result = {}
        for p in inputs_dir.rglob("*"):
            if p.is_file():
                try:
                    result[str(p.relative_to(inputs_dir))] = p.read_text(errors="replace")
                except Exception:
                    pass
        self.inputs = result
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Actions
# ─────────────────────────────────────────────────────────────────────────────


def act_read_input(task: Task, *, input_id: str | None = None) -> dict:
    """Read all task inputs (or one named). Returns observation dict."""
    inputs = task.load_inputs()
    if input_id:
        text = inputs.get(input_id, "")
        if not text:
            return dict(status="error", summary=f"input {input_id} not found",
                        extracted_facts=[], cost=0.0)
        facts = _extract_facts(text)
        return dict(status="success",
                    summary=f"read {input_id} ({len(text)} chars)",
                    extracted_facts=facts, cost=0.0)
    # Read all
    facts: list[str] = []
    for name, text in inputs.items():
        facts.extend(_extract_facts(text)[:5])
    return dict(status="success",
                summary=f"read all inputs ({len(inputs)} files)",
                extracted_facts=facts[:20], cost=0.0)


def act_read_problem(task: Task) -> dict:
    """Coding tasks: read problem statement + tests."""
    inputs = task.load_inputs()
    facts = [task.prompt[:200]]
    facts.extend([f"file:{k}" for k in inputs.keys()])
    return dict(status="success",
                summary=f"problem loaded ({len(task.prompt)} chars, {len(inputs)} files)",
                extracted_facts=facts, cost=0.0)


def act_search(task: Task, *, query: str = "") -> dict:
    """Simulated search — pull from inputs (no real web)."""
    inputs = task.load_inputs()
    hits = []
    q = (query or task.prompt[:80]).lower()
    for name, text in inputs.items():
        for line in text.splitlines():
            if any(tok in line.lower() for tok in q.split() if len(tok) > 3):
                hits.append(f"{name}: {line.strip()[:120]}")
                if len(hits) >= 5:
                    break
        if len(hits) >= 5:
            break
    if not hits:
        hits = [f"{name}: {next(iter(text.splitlines()), '')[:120]}"
                for name, text in list(inputs.items())[:3]]
    return dict(status="success",
                summary=f"search returned {len(hits)} hits",
                extracted_facts=hits, cost=0.0)


def act_search_memory(task: Task, *, query: str = "") -> dict:
    """Long-memory simulated retrieval — same as search but tagged."""
    obs = act_search(task, query=query)
    obs["summary"] = "search_memory: " + obs["summary"]
    return obs


def act_use_tool(task: Task, *, tool_name: str | None = None,
                 tool_args: dict | None = None,
                 llm_client: LLMClient | None = None) -> dict:
    """Generic tool invocation. For Base Harness we simulate calculators/PDF
    readers/etc deterministically rather than going to network."""
    if tool_name is None:
        return dict(status="error", summary="no tool_name", cost=0.0,
                    tool_args_valid=False)
    if tool_name not in task.available_tools:
        return dict(status="error", summary=f"tool {tool_name} not allowed",
                    tool_name=tool_name, tool_args_valid=False, cost=0.0)
    if tool_name == "calculator":
        expr = (tool_args or {}).get("expr", "")
        try:
            # very small safe arith eval
            if not re.match(r"^[\d\s+\-*/().,]+$", expr):
                return dict(status="error", summary="unsafe expr",
                            tool_name=tool_name, tool_args_valid=False, cost=0.0)
            val = eval(expr)
            return dict(status="success", summary=f"{expr}={val}",
                        tool_name=tool_name, tool_args_valid=True, cost=0.0,
                        extracted_facts=[f"calc:{expr}={val}"])
        except Exception as e:
            return dict(status="error", summary=str(e),
                        tool_name=tool_name, tool_args_valid=False, cost=0.0)
    if tool_name in ("read_pdf", "extract_table"):
        # Simulate by reading inputs
        inputs = task.load_inputs()
        facts = []
        for name, text in inputs.items():
            facts.append(f"{tool_name}({name}): {text[:120]}")
        return dict(status="success", summary=f"{tool_name} OK",
                    tool_name=tool_name, tool_args_valid=True, cost=0.0,
                    extracted_facts=facts[:5])
    # Generic fallback: return ok
    return dict(status="success", summary=f"{tool_name} OK",
                tool_name=tool_name, tool_args_valid=True, cost=0.0)


def act_run_tests(task: Task, *, code_blob: str) -> dict:
    """Coding: execute user code + run pytest-style assertions.

    Test files live in task.task_dir/inputs/code/test_*.py.
    """
    inputs_dir = task.task_dir / "inputs" / "code"
    test_files = sorted(inputs_dir.glob("test_*.py")) if inputs_dir.exists() else []
    if not test_files:
        return dict(status="success", summary="no tests to run",
                    test_results={"passed": 0, "failed": 0, "errors": []},
                    cost=0.0)
    with tempfile.TemporaryDirectory() as td:
        sol_path = Path(td) / "solution.py"
        sol_path.write_text(code_blob)
        passed, failed, errors = 0, 0, []
        for tf in test_files:
            test_code = tf.read_text()
            # We assume the function name to test is the first identifier mentioned in assert
            try:
                ns: dict = {}
                exec(code_blob, ns)
                exec(test_code, ns)
                # Treat each assertion line as one test
                lines = [l for l in test_code.splitlines() if l.strip().startswith("assert")]
                passed += len(lines) or 1
            except AssertionError as e:
                failed += 1
                errors.append(f"{tf.name}: AssertionError {e}")
            except Exception as e:
                failed += 1
                errors.append(f"{tf.name}: {type(e).__name__} {e}")
        total = passed + failed
        return dict(
            status="success",
            summary=f"tests: passed={passed} failed={failed}",
            test_results={"passed": passed, "failed": failed, "errors": errors},
            cost=0.0,
        )


def act_draft_solution(
    task: Task,
    *,
    facts_so_far: list[str],
    llm_client: LLMClient | None = None,
    max_tokens: int = 400,
) -> dict:
    """Use LLM to write the deliverable."""
    client = llm_client or get_default_client()
    schema_hint = _schema_hint_for(task)
    sys_msg = (
        "You are an agent producing a final answer for a task. "
        "Be concise. Follow the requested format. Cite specific facts when given."
    )
    user_msg = (
        f"TASK:\n{task.prompt}\n\n"
        f"FACTS ALREADY GATHERED ({len(facts_so_far)}):\n"
        + "\n".join(f"- {f}" for f in facts_so_far[:20])
        + f"\n\n{schema_hint}\n"
        "Produce the final answer now."
    )
    res = client.chat(
        messages=[{"role": "system", "content": sys_msg},
                  {"role": "user", "content": user_msg}],
        max_completion_tokens=max_tokens,
    )
    if not res.ok:
        return dict(status="error", summary=f"LLM draft failed: {res.error}",
                    cost=0.0)
    text = res.text or ""
    claims = _extract_claims(text)
    # Heuristic: any claim that overlaps with a fact gets "evidence".
    cwe = []
    facts_join = " ".join(facts_so_far).lower()
    for i, c in enumerate(claims):
        tokens = [t for t in re.findall(r"\w+", c.lower()) if len(t) > 3]
        if tokens and sum(1 for t in tokens if t in facts_join) / max(len(tokens), 1) > 0.2:
            cwe.append(i)
    return dict(
        status="success",
        summary=f"drafted {len(text)} chars, {len(claims)} claims",
        cost=res.cost_usd,
        draft_update={
            "has_draft": True,
            "draft_text": text,
            "claims": claims,
            "claims_with_evidence": cwe,
        },
    )


def act_write_code(
    task: Task,
    *,
    llm_client: LLMClient | None = None,
    max_tokens: int = 400,
) -> dict:
    """Use LLM to write code for a coding task."""
    client = llm_client or get_default_client()
    tests = ""
    code_dir = task.task_dir / "inputs" / "code"
    if code_dir.exists():
        for tf in code_dir.glob("test_*.py"):
            tests += f"\n--- {tf.name} ---\n{tf.read_text()}\n"
    sys_msg = (
        "You are a code-writing agent. Output ONLY the python code, no markdown fences, "
        "no commentary. Use the function name implied by the tests."
    )
    user_msg = f"PROBLEM:\n{task.prompt}\n\nTESTS:\n{tests}\nReturn the python code."
    res = client.chat(
        messages=[{"role": "system", "content": sys_msg},
                  {"role": "user", "content": user_msg}],
        max_completion_tokens=max_tokens,
    )
    if not res.ok:
        return dict(status="error", summary=f"LLM code-write failed: {res.error}",
                    cost=0.0)
    code = _strip_code_fence(res.text or "")
    return dict(
        status="success",
        summary=f"wrote {len(code)} chars of code",
        cost=res.cost_usd,
        draft_update={
            "has_draft": True,
            "code_blob": code,
        },
    )


def act_revise_code(task: Task, *, code_blob: str, test_errors: list,
                    llm_client: LLMClient | None = None,
                    max_tokens: int = 400) -> dict:
    """Ask LLM to fix code given failing tests."""
    client = llm_client or get_default_client()
    sys_msg = "You are fixing a buggy function. Output ONLY the corrected python code, no markdown."
    user_msg = (
        f"PROBLEM:\n{task.prompt}\n\nOLD CODE:\n{code_blob}\n\n"
        f"ERRORS:\n" + "\n".join(test_errors[:5]) + "\nReturn fixed code."
    )
    res = client.chat(
        messages=[{"role": "system", "content": sys_msg},
                  {"role": "user", "content": user_msg}],
        max_completion_tokens=max_tokens,
    )
    if not res.ok:
        return dict(status="error", summary=res.error or "revise failed", cost=0.0)
    new_code = _strip_code_fence(res.text or "")
    return dict(
        status="success",
        summary=f"revised code -> {len(new_code)} chars",
        cost=res.cost_usd,
        draft_update={"has_draft": True, "code_blob": new_code},
    )


def act_check_rubric(task: Task, *, draft_state: dict,
                     llm_client: LLMClient | None = None) -> dict:
    """Local rubric self-check (no LLM call needed for Base Harness).

    Approximates coverage as fraction of criteria descriptions whose
    keywords appear in the draft text. Cheap and deterministic.
    """
    criteria = task.rubric.get("criteria", [])
    text = (draft_state.get("draft_text")
            or draft_state.get("code_blob") or "").lower()
    missing = []
    cov_hits = 0
    for c in criteria:
        desc = (c.get("description") or "").lower()
        # extract content words
        words = [w for w in re.findall(r"[a-zA-Z']+", desc) if len(w) > 4]
        hit = sum(1 for w in words if w in text)
        if words and hit / len(words) >= 0.3:
            cov_hits += 1
        else:
            missing.append(c["id"])
    coverage = cov_hits / len(criteria) if criteria else 0.0
    return dict(
        status="success",
        summary=f"coverage={coverage:.2f}, missing={missing}",
        cost=0.0,
        rubric_update={"coverage": coverage, "missing_ids": missing},
    )


def act_verify_solution(task: Task, *, draft_state: dict) -> dict:
    """Cheap structural verifier — checks format & non-empty draft.
    Real verification is done at end-of-episode via reward_aggregator."""
    text = draft_state.get("draft_text") or draft_state.get("code_blob") or ""
    if not text or len(text) < 30:
        return dict(status="error", summary="draft too short / missing",
                    cost=0.0)
    return dict(status="success", summary="draft non-empty", cost=0.0)


def act_observe(task: Task) -> dict:
    """No-op observation step (planning / generic)."""
    return dict(status="success", summary="observed task state", cost=0.0)


def act_plan(task: Task, *, llm_client: LLMClient | None = None,
             max_tokens: int = 200) -> dict:
    """Planning: ask LLM for a short plan outline."""
    client = llm_client or get_default_client()
    res = client.chat(
        messages=[
            {"role": "system",
             "content": "Produce a short numbered plan to solve the task."},
            {"role": "user", "content": task.prompt},
        ],
        max_completion_tokens=max_tokens,
    )
    if not res.ok:
        return dict(status="error", summary=res.error, cost=0.0)
    return dict(status="success", summary="plan drafted",
                cost=res.cost_usd,
                draft_update={"has_draft": True, "draft_text": res.text or ""})


def act_submit() -> dict:
    return dict(status="success", summary="submit", cost=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _extract_facts(text: str, max_facts: int = 8) -> list:
    """Extract sentence-level facts heuristically."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out = []
    for s in sentences:
        s = s.strip()
        if 10 <= len(s) <= 200:
            out.append(s)
        if len(out) >= max_facts:
            break
    return out


def _extract_claims(text: str) -> list:
    """Extract factual-looking claims from a draft."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    claims = [s.strip() for s in sentences
              if 20 <= len(s.strip()) <= 250
              and any(w in s.lower() for w in (" is ", " are ", " was ", " were ",
                                                " has ", " have ", " did ", " do ",
                                                "percent", "growth", "value",
                                                "result", "answer", "median",
                                                "average", "compared"))]
    return claims[:8]


def _strip_code_fence(text: str) -> str:
    # Strip ``` blocks if present
    m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.S)
    if m:
        return m.group(1).strip()
    return text.strip()


def _schema_hint_for(task: Task) -> str:
    tt = task.task_type
    if tt == "knowledge_work_deliverable" or tt == "knowledge_work":
        return ("Format: markdown with `## Summary`, `## Findings`, "
                "`## Recommendation` sections.")
    if tt == "coding":
        return "Format: code only, no commentary."
    if tt == "research":
        return ("Format: 3-5 sentences with parenthetical citations like "
                "(source: <name>). Cite at least 2 distinct sources.")
    if tt == "multi_tool":
        return "Format: final answer with reasoning and numeric result."
    if tt == "long_memory":
        return ("Format: a JSON object with keys 'answer' and 'source_session'.")
    if tt == "planning":
        return ('Format: numbered plan lines, then final JSON `{"steps":[...],'
                ' "satisfies_constraints": true}`.')
    return "Format: plain text."
