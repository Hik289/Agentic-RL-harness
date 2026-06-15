"""Format reward (readme §12.3).

R_format ∈ {0, 0.5, 1.0}
  1.0: 完全满足
  0.5: 部分满足
  0.0: 完全失败

Supports schema_type:
  - "markdown_sections": output is markdown, contains required headers (## or # level)
  - "json"             : output is valid JSON and matches required top-level keys
  - "csv"              : output is parseable CSV with required columns
  - "plain"            : non-empty trimmed text
"""
from __future__ import annotations

import csv
import io
import json
import re
from typing import Any


def _check_markdown_sections(output: str, schema: dict) -> float:
    required = [s.lower() for s in schema.get("required_sections", [])]
    if not output or not output.strip():
        return 0.0
    headers = [
        m.group(1).strip().lower()
        for m in re.finditer(r"^#{1,3}\s+(.+?)\s*$", output, flags=re.M)
    ]
    if not headers:
        return 0.0
    if not required:
        return 1.0
    hit = sum(1 for r in required if any(r in h for h in headers))
    if hit == len(required):
        return 1.0
    if hit > 0:
        return 0.5
    return 0.0


def _check_json(output: str, schema: dict) -> float:
    required = schema.get("required_keys", [])
    if not output or not output.strip():
        return 0.0
    try:
        obj = json.loads(output.strip())
    except Exception:
        # Try to extract first {...} blob
        m = re.search(r"\{.*\}", output, flags=re.S)
        if not m:
            return 0.0
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return 0.0
    if not isinstance(obj, dict):
        return 0.5
    if not required:
        return 1.0
    hit = sum(1 for k in required if k in obj)
    if hit == len(required):
        return 1.0
    if hit > 0:
        return 0.5
    return 0.0


def _check_csv(output: str, schema: dict) -> float:
    required_cols = [c.lower() for c in schema.get("required_columns", [])]
    if not output or not output.strip():
        return 0.0
    try:
        reader = csv.reader(io.StringIO(output.strip()))
        rows = list(reader)
    except Exception:
        return 0.0
    if len(rows) < 1:
        return 0.0
    header = [c.strip().lower() for c in rows[0]]
    if not required_cols:
        return 1.0 if header else 0.0
    hit = sum(1 for r in required_cols if r in header)
    if hit == len(required_cols):
        return 1.0
    if hit > 0:
        return 0.5
    return 0.0


def _check_plain(output: str, schema: dict) -> float:
    min_chars = schema.get("min_chars", 1)
    if not output or len(output.strip()) < min_chars:
        return 0.0
    return 1.0


_CHECKERS = {
    "markdown_sections": _check_markdown_sections,
    "json": _check_json,
    "csv": _check_csv,
    "plain": _check_plain,
}


def format_reward(output: str, schema: dict | None = None) -> float:
    """Return R_format ∈ {0.0, 0.5, 1.0}.

    schema = {"schema_type": "markdown_sections"|"json"|"csv"|"plain", ...}
    """
    if schema is None:
        schema = {"schema_type": "plain"}
    stype = schema.get("schema_type", "plain")
    checker = _CHECKERS.get(stype, _check_plain)
    return float(checker(output, schema))
