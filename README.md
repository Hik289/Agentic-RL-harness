# Learning to Harness — Rubric-Guided Outer-Loop RL for Agentic Harness Optimization

Code release accompanying the paper. The harness is modeled as an outer
control policy over a fixed LLM executor; a lightweight RL controller
(Offline Advantage-Weighted policy) is trained on task-rubric reward to
learn *process maturity* without requiring outcome-distribution change.

## Quick start

```bash
git clone <this repo>
cd agenticrlharness
pip install -r requirements.txt

# Configure provider + paths
cp .env.example .env
$EDITOR .env      # fill in AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY
source .env

# Sanity-check API
python examples/anchor_1_api_check.py
```

You should see ~110/110 successful calls in <2 min and `total_cost ≈ $0.002`.

## Layout

```
code/
├── harness/                Base harness execution machinery
│   ├── agent.py            Scripted Base Harness episode runner
│   ├── actions.py          Generic action executors (read/draft/check/...)
│   ├── state_builder.py    Numeric state vector for RL controller
│   ├── trajectory_logger.py  B.1-schema logger (full per-step records)
│   ├── submission.py       Post-hoc structural rubric scorer (universal verifier)
│   └── util/llm_client.py  Azure gpt-* wrapper, env-var driven
├── reward/                 Reward computation (paper §12)
│   ├── reward_aggregator.py  R_total = R_rubric + α·R_verify + β·R_format + δ·R_task − γ·P_error − λ·P_cost − μ·P_early
│   ├── rubric_judge.py     LLM judge (calibration vs reward mode boundary)
│   ├── format_checker.py   Markdown / JSON / CSV / plain schema check
│   └── cost_penalty.py     min(cost/budget, 1)
├── modules/
│   └── hms_detector.py     Harness Maturity Score (7 behavior events, paper §20)
└── rl/                     Offline AW + per-domain harnesses
    ├── policy.py           1-hidden-layer MLP (hidden=64), softmax over 8 padded actions
    ├── offline_aw.py       Train loop (Adam lr=1e-3, β=0.2, weight clip [0.1,10], entropy=0.01)
    ├── state_features.py   18-dim state featurizer (coding)
    ├── kw_state_features.py  KW featurizer
    ├── coding_harness.py   Coding-domain action loop
    ├── kw_harness.py       Knowledge-work-domain action loop
    └── generic_harness.py  Other domains (research/multi_tool/long_memory/planning)

examples/
├── anchor_1_api_check.py        API stability sanity check
├── anchor_2_base_harness.py     Base Harness end-to-end smoke
├── anchor_3_rubric_judge.py     LLM judge calibration (mode=calibration)
├── anchor_3b_judge_reward_mode.py  LLM judge calibration (mode=reward)
├── anchor_4_reward_aggregator.py   Reward aggregator unit tests
├── anchor_5_offline_aw.py       Offline AW vs Base, toy coding
├── anchor_6_hms_detector.py     HMS detector unit tests
├── running_main_driver.py       Per-domain Base vs AW main-table runner
├── main_table_analysis.py       Aggregate 6 domain results into summary table
└── c7_sensitivity_analysis.py   EarlySubmit threshold sensitivity (v4/v5/v6)
```

## Reproducing the main table

Each domain runs `collect → train (3 seeds) → eval` from the
`AGENTICRLHARNESS_DATA` task root. The driver writes results to
`AGENTICRLHARNESS_RESULTS/main_{domain}/results.json`.

```bash
# Synthetic 6 domains (80 train + 20 eval, stratified by difficulty)
for D in knowledge_work coding research multi_tool long_memory planning; do
  python examples/running_main_driver.py --domain $D
done

# Public benchmarks (16 train + 4 held-out, explicit eval task IDs)
python examples/running_main_driver.py \
  --domain tau_bench_retail_v2_heldout \
  --eval_task_ids tau_retail_003,tau_retail_005,tau_retail_011,tau_retail_015 \
  --n_rollouts_collect 20 --n_rollouts_eval 3 --n_eval_seeds 3

python examples/running_main_driver.py \
  --domain agentbench_dbbench \
  --eval_task_ids agentbench_dbbench_003,agentbench_dbbench_007,agentbench_dbbench_012,agentbench_dbbench_017 \
  --n_rollouts_collect 20 --n_rollouts_eval 3 --n_eval_seeds 3

# Aggregate
python examples/main_table_analysis.py
```

Expected wall-clock per synthetic domain: ~90 min (1600 collect + 540 eval rollouts).
Each LLM call is short (~2 s), Azure rate limits permitting.

## Reproducing the EarlySubmit threshold sensitivity (paper §4 Table)

```bash
# After main table runs (need eval_records_{base,aw}.jsonl per domain):
python examples/c7_sensitivity_analysis.py
```

No additional LLM calls; pure re-scoring at thresholds 0.25 / 0.30 / 0.35.

## Hyperparameters (single source of truth: code/rl/offline_aw.py:AWConfig)

| Component | Value |
|---|---|
| Policy net | 1-hidden MLP, 64 units, ReLU, softmax over 8 actions |
| State dim | 18 (10 numeric + 8 one-hot last-action) |
| Optimizer | Adam, lr=1e-3, weight_decay=0 |
| Epochs / batch | 20 / 256 |
| AW temperature β | 0.2 |
| Weight clip | [0.1, 10.0] |
| Entropy coefficient | 0.01 |
| Behavior ε | 0.25 |
| Rollouts / train task | 20 |
| Seeds per domain | 3 (0, 1, 2) |

Same hyperparameters across all 8 synthetic domains + 2 public benchmarks; no per-domain tuning.

## Citation

```bibtex
@inproceedings{learningtoharness2026,
  title     = {Learning to Harness: Rubric-Guided Outer-Loop RL for Agentic Harness Optimization},
  author    = {Anonymous},
  year      = {2026},
}
```

## License

MIT — see `LICENSE`.
