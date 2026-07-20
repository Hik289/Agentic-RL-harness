# Artifact Guide

This guide maps the public `Agentic-RL-harness` repository to a reviewer-friendly artifact workflow for `Learning to Control LLM Agent Harnesses with Offline Reinforcement Learning`. It is meant to make the release easier to inspect in the style of ICML, ICLR, NeurIPS, and similar artifact-review processes.

## What To Inspect First

- `code/`: Project-specific implementation subtree.
- `examples/`: Small runnable examples and smoke-test entry points.
- `figures/`: README and paper-facing figures.

## Environment Files

- `requirements.txt`: Primary Python dependency list.
- `.env.example`: Template for local credentials or backend configuration.

## Minimal Verification

Run these checks in a fresh environment before launching expensive jobs:

```bash
python -m compileall -q .
```

If a smoke command is not tracked, use the README Quick Start with the smallest available seed, sample, or task count.

## Reproduction And Analysis Entry Points

No single reproduction runner is tracked. Use the README experiment commands and the implementation map above; keep first runs small before scaling to full grids.

## Figure Assets

- `figures/g1_offline_aw_pipeline.png`
- `figures/intuition.png`

## Data, Credentials, And Generated Outputs

- API-backed runs should read credentials from environment variables or local `.env` files only; never commit real keys or provider-specific secrets.
- Record provider endpoint, model/deployment name, sampling parameters, and execution date for every API-backed table or figure.
- Treat generated JSONL files, logs, caches, model checkpoints, and benchmark downloads as local artifacts unless explicitly tracked as fixtures.
- For stochastic experiments, record seeds, task counts, dataset splits, and the exact git commit used for the run.

## Reviewer Reporting Checklist

- `git rev-parse HEAD`
- Python version and dependency-install command
- Full command line for every table, figure, or benchmark cell
- Paths to raw outputs and aggregation scripts
- External data, benchmark, or API-backed steps that were intentionally skipped
