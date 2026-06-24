# Learning to Harness

English | [简体中文](README_zh-CN.md)

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/Hik289/Agentic-RL-harness)
[![Results](https://img.shields.io/badge/Results-v2__heldout%20formal-0A66C2)](https://github.com/Hik289/Agentic-RL-harness/blob/main/results/tables/main_table_v2_heldout_formal.md)
[![License](https://img.shields.io/badge/License-MIT-2ea44f)](https://github.com/Hik289/Agentic-RL-harness/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://github.com/Hik289/Agentic-RL-harness/blob/main/pyproject.toml)
[![Docs](https://img.shields.io/badge/Docs-%E4%B8%AD%E6%96%87-d73a49)](https://github.com/Hik289/Agentic-RL-harness/blob/main/README_zh-CN.md)

Quick links: [Repository](https://github.com/Hik289/Agentic-RL-harness) |
[Results table](https://github.com/Hik289/Agentic-RL-harness/blob/main/results/tables/main_table_v2_heldout_formal.md) |
[Chinese README](https://github.com/Hik289/Agentic-RL-harness/blob/main/README_zh-CN.md) |
[Citation](#citation)

Paper status: preprint link coming soon. The current release includes the full
codebase, reproducible local benchmark, and released `v2_heldout` result
tables.

Many LLM-agent evaluations conflate model capability with harness design. In
practice, the same underlying model can look weak or strong depending on how
the outer loop decides when to search, draft, verify, revise, and submit.
Learning to Harness starts from the thesis that harness control is itself an
optimization problem and should be studied directly rather than treated as a
fixed implementation detail.

This repository releases that view as runnable code: keep the LLM executor
fixed, model the outer-loop harness as a reinforcement-learning problem, and
measure not only final task quality but also process maturity under a fixed
interaction budget.

The project focuses on:

- harness optimization rather than model fine-tuning
- verifier- and rubric-driven reward instead of pass/fail-only grading
- process maturity signals and improvement trajectories, not just final answer quality

The released local benchmark currently covers six synthetic domains:
`knowledge_work`, `coding`, `research`, `multi_tool`, `long_memory`, and
`planning`. The repository supports both deterministic mock-mode reproduction
and Azure-backed runs with a fixed LLM executor.

## Docs Navigation

| Topic | Start here | Why it matters |
|---|---|---|
| Overview | [Method at a Glance](#method-at-a-glance) | The core thesis and what is being optimized |
| Benchmark | [Benchmark and Data](#benchmark-and-data) | Domains, heldout split, and mock-data layout |
| Results | [Results Snapshot](#results-snapshot) | Released `v2_heldout` numbers and result artifacts |
| Reproduction | [Running Main Benchmarks](#running-main-benchmarks) | End-to-end `collect -> train -> eval` commands |
| Code map | [Repository Map](#repository-map) | Where the harness, reward, RL, and scripts live |
| Paper metadata | [Citation](#citation) | BibTeX and release metadata |

## Method at a Glance

The repository separates three things that are often collapsed together in
agent evaluations:

- the fixed LLM executor
- the harness policy that decides what to do next
- the reward and process metrics used to score trajectories

The default Base Harness is scripted. On top of it, the repository trains a
lightweight Offline Advantage-Weighted controller that learns a
state-conditioned policy over harness decisions. Reward combines rubric score,
verification, format, and task-level signals with explicit penalties for
errors, cost, and early submission.

This release also treats process quality as a first-class target. In addition
to final return `G`, it tracks Harness Maturity Score (HMS), which measures
whether a policy exhibits better search, checking, revision, and submission
behavior instead of merely getting lucky on final answers.

## Benchmark and Data

The current local release is built around a deterministic synthetic benchmark
that is easy to reproduce without external dependencies.

- Domains: `knowledge_work`, `coding`, `research`, `multi_tool`,
  `long_memory`, `planning`
- Data roots: `data/synthetic_tasks_main_v2_heldout` and related mock-data
  variants under `data/`
- Split design: the `v2_heldout` setup changes evaluation templates relative
  to training templates
- Outputs: run artifacts land under `results/`, with released summary tables in
  `results/tables/`

## 0. Host Requirements

The local release is designed to run on a normal Python workstation. The
default mock benchmark is CPU-friendly and does not require a GPU.

| Requirement | Needed for | Notes |
|---|---|---|
| Python 3.10+ | all setup paths | `pyproject.toml` requires `>=3.10` |
| `venv` + `pip` | local environment creation | `scripts/setup_local.sh` bootstraps `.venv` for you |
| Writable local workspace | generated data and outputs | mock tasks live under `data/`, outputs under `results/` |
| Azure OpenAI endpoint + key | non-mock LLM-backed runs | only needed when `AGENTICRLHARNESS_LLM_MODE=azure` |

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/Hik289/Agentic-RL-harness.git
cd Agentic-RL-harness
```

### 2. Create the local mock environment

```bash
bash scripts/setup_local.sh
```

This script:

- creates `.venv` if needed
- installs the package from `requirements.txt` (`-e .`)
- copies `.env.local.mock` to `.env` if `.env` does not exist
- bootstraps toy data into `data/`
- runs `scripts/check_environment.py`

### 3. Run the smoke test

```bash
source .env.local.mock
bash scripts/run_local_smoke.sh
```

This is the fastest end-to-end validation path. It exercises environment
checks, API-mode wiring, the Base Harness, a small Offline AW run, and a
smoke-sized main-driver invocation.

Some experiment drivers return exit code `2` when a paper-style improvement
invariant is not met. That does not mean the runtime path is broken.

### 4. Switch to Azure-backed mode (optional)

```bash
cp .env.example .env
$EDITOR .env
source .env

.venv/bin/python examples/anchor_1_api_check.py
```

For Azure mode, set:

- `AGENTICRLHARNESS_LLM_MODE=azure`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`

## Repository Map

- `code/harness/` contains the Base Harness action loop, state builder, logger,
  submission scorer, and LLM client wrapper.
- `code/reward/` contains reward aggregation, format checks, cost penalty
  logic, and rubric-judge code.
- `code/modules/hms_detector.py` implements the Harness Maturity Score event
  detector.
- `code/rl/` contains the policy network, Offline AW trainer, state
  featurizers, and per-domain harness controllers.
- `examples/` contains anchor scripts, calibration checks, ablation runs, the
  main-table driver, and analysis utilities.
- `scripts/` contains setup, mock-data generation, environment checks, and
  benchmark aggregation helpers.

## Results Snapshot

The local `v2_heldout` benchmark uses deterministic mock-mode execution and a
heldout-template split (`data/synthetic_tasks_main_v2_heldout`) so evaluation
templates differ from training templates. The formal local run below uses 10
collect rollouts per train task, 3 eval rollouts, 3 seeds, and 15 AW epochs.

| Domain | Base | AW | Delta G | p | HMS Delta |
|---|---:|---:|---:|---:|---:|
| knowledge_work | 0.7100 | 0.8889 | +0.1789 | 0.0000 | +0.3333 |
| coding | 0.5500 | 1.0000 | +0.4500 | 0.0000 | +0.2000 |
| research | 0.6000 | 0.9200 | +0.3200 | 0.0000 | +0.4667 |
| multi_tool | 0.5200 | 0.6900 | +0.1700 | 0.0000 | +0.3333 |
| long_memory | 0.5500 | 1.0000 | +0.4500 | 0.0000 | +0.3000 |
| planning | 0.9333 | 1.0000 | +0.0667 | 0.0000 | +0.0000 |

Macro Base is `0.6439`, macro AW is `0.9165`, and macro Delta G is `+0.2726`.
All 6 domains show positive Delta G and satisfy the 5pp-plus-bootstrap-`p<0.05`
invariant. The `base_clone` ablation, which evaluates the Base policy in the AW
slot, has macro Delta G `+0.0000` across the same six domains.

Generated tables:

- `results/tables/main_table_v2_heldout_formal.md`
- `results/tables/main_table_v2_heldout_formal.csv`
- `results/tables/main_table_v2_heldout_formal.json`
- `results/tables/main_table_v2_heldout_base_clone_formal.md`
- `results/tables/main_table_v2_heldout_base_clone_formal.csv`

Released result links:

- [Main table (Markdown)](https://github.com/Hik289/Agentic-RL-harness/blob/main/results/tables/main_table_v2_heldout_formal.md)
- [Main table (CSV)](https://github.com/Hik289/Agentic-RL-harness/blob/main/results/tables/main_table_v2_heldout_formal.csv)
- [Main table (JSON)](https://github.com/Hik289/Agentic-RL-harness/blob/main/results/tables/main_table_v2_heldout_formal.json)
- [Base-clone ablation (Markdown)](https://github.com/Hik289/Agentic-RL-harness/blob/main/results/tables/main_table_v2_heldout_base_clone_formal.md)

## Running Main Benchmarks

There are two runtime modes in this repository:

- `mock`: deterministic local responses, no external API calls
- `azure`: fixed LLM executor backed by Azure OpenAI

### Reproduce the `v2_heldout` main table

Each domain runs `collect -> train (3 seeds) -> eval` from the
`AGENTICRLHARNESS_DATA` task root. Results are written to
`AGENTICRLHARNESS_RESULTS/main_{domain}_v2_heldout_{tag}/results.json`.

```bash
source .env.local.mock

python scripts/bootstrap_mock_data.py --variant v2_heldout --overwrite

for D in knowledge_work coding research multi_tool long_memory planning; do
  .venv/bin/python examples/running_main_driver.py \
    --domain "$D" \
    --benchmark_version v2_heldout \
    --n_rollouts_collect 10 \
    --n_rollouts_eval 3 \
    --n_eval_seeds 3 \
    --epochs 15 \
    --output_tag _heldoutformal
done

.venv/bin/python scripts/aggregate_benchmark_results.py \
  --benchmark_version v2_heldout \
  --output_tag _heldoutformal \
  --name main_table_v2_heldout_formal
```

### Reproduce the `base_clone` ablation

```bash
for D in knowledge_work coding research multi_tool long_memory planning; do
  .venv/bin/python examples/running_main_driver.py \
    --domain "$D" \
    --benchmark_version v2_heldout \
    --aw_ablation base_clone \
    --n_rollouts_collect 10 \
    --n_rollouts_eval 3 \
    --n_eval_seeds 3 \
    --epochs 15 \
    --output_tag _formalabl
done

.venv/bin/python scripts/aggregate_benchmark_results.py \
  --benchmark_version v2_heldout \
  --aw_ablation base_clone \
  --output_tag _formalabl \
  --name main_table_v2_heldout_base_clone_formal
```

### Smoke-sized driver run

For a fast local run, shrink the rollout counts:

```bash
.venv/bin/python examples/running_main_driver.py \
  --domain knowledge_work \
  --benchmark_version v2_heldout \
  --n_rollouts_collect 2 \
  --n_rollouts_eval 1 \
  --n_eval_seeds 1 \
  --epochs 3
```

Expected wall-clock for the full paper-scale synthetic setting is much longer
than the local mock setting.

### Re-score the EarlySubmit threshold sensitivity

After the main table runs, the sensitivity analysis needs
`eval_records_{base,aw}.jsonl` per domain:

```bash
python examples/c7_sensitivity_analysis.py
```

This is pure re-scoring at thresholds `0.25`, `0.30`, and `0.35`; no extra LLM
calls are required.

## Where To Go Next

| Topic | Read this | Run this | Outcome |
|---|---|---|---|
| Method | `code/harness/agent.py`, `code/rl/offline_aw.py`, `code/reward/reward_aggregator.py` | `examples/anchor_4_reward_aggregator.py` | Understand the Base Harness, learned controller, and reward design |
| Data | `scripts/bootstrap_mock_data.py`, `data/mock_data_summary.json` | `python scripts/bootstrap_mock_data.py --variant v2_heldout --overwrite` | Recreate the local synthetic benchmark inputs |
| Results | `results/tables/main_table_v2_heldout_formal.md`, `examples/main_table_analysis.py` | `python examples/c7_sensitivity_analysis.py` | Inspect released metrics, tables, and threshold re-scoring |
| Reproduction | `examples/running_main_driver.py` | Use the `v2_heldout` command block above | Reproduce `collect -> train -> eval` per domain |
| Smoke test | `scripts/run_local_smoke.sh` | `bash scripts/run_local_smoke.sh` | Validate the full local runtime path quickly |
| API mode | `examples/anchor_1_api_check.py`, `.env.example` | `.venv/bin/python examples/anchor_1_api_check.py` | Confirm Azure-backed mode before longer runs |
| Process metrics | `code/modules/hms_detector.py`, `examples/anchor_6_hms_detector.py` | `.venv/bin/python examples/anchor_6_hms_detector.py` | Inspect HMS events and process-quality checks |

## Hyperparameters

Single source of truth: `code/rl/offline_aw.py`, `AWConfig`.

| Component | Value |
|---|---|
| Policy net | 1-hidden MLP, 64 units, ReLU, softmax over 8 actions |
| State dim | 18 (10 numeric + 8 one-hot last-action) |
| Optimizer | Adam, lr=1e-3, weight_decay=0 |
| Epochs / batch | 20 / 256 |
| AW temperature beta | 0.2 |
| Weight clip | [0.1, 10.0] |
| Entropy coefficient | 0.01 |
| Behavior epsilon | 0.25 |
| Rollouts / train task | 20 |
| Seeds per domain | 3 (0, 1, 2) |

Same hyperparameters are used across all 8 synthetic domains and 2 public
benchmarks; there is no per-domain tuning.

## Citation

```bibtex
@inproceedings{learningtoharness2026,
  title     = {Learning to Harness: Rubric-Guided Outer-Loop RL for Agentic Harness Optimization},
  author    = {Anonymous},
  year      = {2026},
}
```

## License

MIT. See `LICENSE`.
