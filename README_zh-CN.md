# Learning to Harness

[English](README.md) | 简体中文

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/Hik289/Agentic-RL-harness)
[![Results](https://img.shields.io/badge/Results-v2__heldout%20formal-0A66C2)](https://github.com/Hik289/Agentic-RL-harness/blob/main/results/tables/main_table_v2_heldout_formal.md)
[![License](https://img.shields.io/badge/License-MIT-2ea44f)](https://github.com/Hik289/Agentic-RL-harness/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://github.com/Hik289/Agentic-RL-harness/blob/main/pyproject.toml)
[![Docs](https://img.shields.io/badge/Docs-English-0366d6)](https://github.com/Hik289/Agentic-RL-harness/blob/main/README.md)

快速入口：[仓库主页](https://github.com/Hik289/Agentic-RL-harness) |
[结果表](https://github.com/Hik289/Agentic-RL-harness/blob/main/results/tables/main_table_v2_heldout_formal.md) |
[English README](https://github.com/Hik289/Agentic-RL-harness/blob/main/README.md) |
[Citation](#citation)

论文状态：preprint 链接即将补充。当前 release 已经包含完整代码、可复现的
本地 benchmark，以及发布版 `v2_heldout` 结果表。

很多 LLM agent 评测会把模型能力和 harness 设计混在一起看。可是在实际运行
里，同一个底层模型会因为外层控制循环如何决定搜索、起草、验证、修订和提交，
呈现出完全不同的质量表现。Learning to Harness 的核心 thesis 是：harness
control 本身就是一个应该被单独研究和优化的问题，而不应被当作固定不变的
实现细节。

这个仓库把这个观点落实成可运行代码：固定 LLM executor，只把 outer-loop
harness 建模为强化学习问题，并且在固定交互预算下同时衡量最终任务质量和
process maturity。

这个项目重点关注：

- 优化 harness，而不是微调底层模型
- 使用 verifier 和 rubric 驱动的奖励，而不是只有 pass/fail 的二值评分
- 关注 process maturity 信号和改进轨迹，而不只看最终答案质量

当前发布的本地 benchmark 覆盖 6 个 synthetic domain：
`knowledge_work`、`coding`、`research`、`multi_tool`、`long_memory`
和 `planning`。仓库同时支持 deterministic mock-mode 复现，以及固定
LLM executor 的 Azure-backed 运行模式。

## 文档导航

| 主题 | 从这里开始 | 为什么重要 |
|---|---|---|
| 总览 | [Method at a Glance](#method-at-a-glance) | 先理解核心 thesis 和优化对象 |
| Benchmark | [Benchmark and Data](#benchmark-and-data) | 了解 domain、heldout split 和 mock 数据布局 |
| 结果 | [Results Snapshot](#results-snapshot) | 查看发布版 `v2_heldout` 数字和结果文件 |
| 复现 | [Running Main Benchmarks](#running-main-benchmarks) | 找到完整的 `collect -> train -> eval` 命令 |
| 代码地图 | [仓库结构](#仓库结构) | 快速定位 harness、reward、RL 和脚本 |
| 论文元数据 | [Citation](#citation) | 获取 BibTeX 和 release 元信息 |

## Method at a Glance

这个仓库把 agent 评测里经常混在一起的三件事拆开来看：

- 固定的 LLM executor
- 决定下一步做什么的 harness policy
- 用来给轨迹打分的 reward 和 process metrics

默认的 Base Harness 是脚本化的。在这个基础上，仓库训练了一个轻量级的
Offline Advantage-Weighted controller，去学习一个依赖状态的 harness
decision policy。奖励由 rubric score、verification、format 和 task-level
signal 组成，同时对错误、成本和过早提交加入显式 penalty。

这个 release 还把 process quality 当成一等目标。除了最终回报 `G`，它还会
跟踪 Harness Maturity Score（HMS），衡量一个策略是否真的表现出更好的搜索、
检查、修订和提交行为，而不只是偶然得到更好的最终答案。

## Benchmark and Data

当前本地 release 围绕一个 deterministic synthetic benchmark 构建，重点是
在不依赖外部服务的情况下实现稳定复现。

- Domains：`knowledge_work`、`coding`、`research`、`multi_tool`、
  `long_memory`、`planning`
- 数据根目录：`data/synthetic_tasks_main_v2_heldout` 以及 `data/` 下的相关
  mock-data variant
- 切分设计：`v2_heldout` 设置让 evaluation template 和 training template
  不同
- 输出位置：运行产物写到 `results/`，发布版汇总表在 `results/tables/`

## 0. 环境要求

本地 release 面向普通 Python 工作站设计。默认的 mock benchmark
可以在 CPU 上运行，不需要 GPU。

| 要求 | 用途 | 说明 |
|---|---|---|
| Python 3.10+ | 所有安装和运行路径 | `pyproject.toml` 要求 `>=3.10` |
| `venv` + `pip` | 创建本地环境 | `scripts/setup_local.sh` 会帮你初始化 `.venv` |
| 可写本地工作目录 | 生成数据和输出结果 | mock tasks 在 `data/`，输出在 `results/` |
| Azure OpenAI endpoint + key | 非 mock 的 LLM 运行 | 只有 `AGENTICRLHARNESS_LLM_MODE=azure` 时才需要 |

## Quickstart

### 1. 克隆仓库

```bash
git clone https://github.com/Hik289/Agentic-RL-harness.git
cd Agentic-RL-harness
```

### 2. 创建本地 mock 环境

```bash
bash scripts/setup_local.sh
```

这个脚本会：

- 在需要时创建 `.venv`
- 根据 `requirements.txt` 安装项目（`-e .`）
- 如果 `.env` 不存在，就从 `.env.local.mock` 复制一份
- 在 `data/` 下生成 toy data
- 运行 `scripts/check_environment.py`

### 3. 运行 smoke test

```bash
source .env.local.mock
bash scripts/run_local_smoke.sh
```

这是最快的端到端验证路径。它会依次检查环境、API 模式接线、Base Harness、
一个小规模 Offline AW 运行，以及 smoke 大小的 main driver 调用。

有些实验 driver 在没有满足论文式 improvement invariant 时会返回退出码
`2`。这不代表运行路径坏了。

### 4. 切换到 Azure-backed 模式（可选）

```bash
cp .env.example .env
$EDITOR .env
source .env

.venv/bin/python examples/anchor_1_api_check.py
```

Azure 模式需要设置：

- `AGENTICRLHARNESS_LLM_MODE=azure`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`

## 仓库结构

- `code/harness/` 包含 Base Harness 的 action loop、state builder、logger、
  submission scorer 和 LLM client wrapper。
- `code/reward/` 包含 reward aggregation、format check、cost penalty 和
  rubric judge 相关逻辑。
- `code/modules/hms_detector.py` 实现了 Harness Maturity Score 的事件检测器。
- `code/rl/` 包含策略网络、Offline AW trainer、state featurizer，以及
  各 domain 的 harness controller。
- `examples/` 包含 anchor 脚本、calibration 检查、ablation 运行、主表驱动
  脚本和分析工具。
- `scripts/` 包含环境初始化、mock 数据生成、环境检查和 benchmark 聚合工具。

## Results Snapshot

本地 `v2_heldout` benchmark 使用 deterministic mock-mode 执行，并采用
heldout-template split（`data/synthetic_tasks_main_v2_heldout`），因此评估
模板和训练模板不同。下面这组正式本地运行使用了每个 train task 10 次
collect rollout、3 次 eval rollout、3 个 seed，以及 15 个 AW epoch。

| Domain | Base | AW | Delta G | p | HMS Delta |
|---|---:|---:|---:|---:|---:|
| knowledge_work | 0.7100 | 0.8889 | +0.1789 | 0.0000 | +0.3333 |
| coding | 0.5500 | 1.0000 | +0.4500 | 0.0000 | +0.2000 |
| research | 0.6000 | 0.9200 | +0.3200 | 0.0000 | +0.4667 |
| multi_tool | 0.5200 | 0.6900 | +0.1700 | 0.0000 | +0.3333 |
| long_memory | 0.5500 | 1.0000 | +0.4500 | 0.0000 | +0.3000 |
| planning | 0.9333 | 1.0000 | +0.0667 | 0.0000 | +0.0000 |

Macro Base 为 `0.6439`，macro AW 为 `0.9165`，macro Delta G 为
`+0.2726`。6 个 domain 的 Delta G 都是正的，并且满足
5pp-plus-bootstrap-`p<0.05` invariant。`base_clone` ablation 会在 AW 槽位
上评估 Base policy，在这 6 个 domain 上的 macro Delta G 为 `+0.0000`。

生成的结果表：

- `results/tables/main_table_v2_heldout_formal.md`
- `results/tables/main_table_v2_heldout_formal.csv`
- `results/tables/main_table_v2_heldout_formal.json`
- `results/tables/main_table_v2_heldout_base_clone_formal.md`
- `results/tables/main_table_v2_heldout_base_clone_formal.csv`

发布版结果链接：

- [主表（Markdown）](https://github.com/Hik289/Agentic-RL-harness/blob/main/results/tables/main_table_v2_heldout_formal.md)
- [主表（CSV）](https://github.com/Hik289/Agentic-RL-harness/blob/main/results/tables/main_table_v2_heldout_formal.csv)
- [主表（JSON）](https://github.com/Hik289/Agentic-RL-harness/blob/main/results/tables/main_table_v2_heldout_formal.json)
- [base-clone ablation（Markdown）](https://github.com/Hik289/Agentic-RL-harness/blob/main/results/tables/main_table_v2_heldout_base_clone_formal.md)

## Running Main Benchmarks

这个仓库有两种运行模式：

- `mock`：本地 deterministic response，不调用外部 API
- `azure`：使用 Azure OpenAI 作为固定的 LLM executor

### 复现 `v2_heldout` 主表

每个 domain 都会从 `AGENTICRLHARNESS_DATA` task root 执行
`collect -> train (3 seeds) -> eval`。结果会写到
`AGENTICRLHARNESS_RESULTS/main_{domain}_v2_heldout_{tag}/results.json`。

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

### 复现 `base_clone` ablation

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

### 运行一个更小的 smoke 版 driver

如果只是想快速本地验证，可以把 rollout 数缩小：

```bash
.venv/bin/python examples/running_main_driver.py \
  --domain knowledge_work \
  --benchmark_version v2_heldout \
  --n_rollouts_collect 2 \
  --n_rollouts_eval 1 \
  --n_eval_seeds 1 \
  --epochs 3
```

完整的 paper-scale synthetic 运行时间会明显长于本地 mock 设置。

### 重新打分 EarlySubmit threshold sensitivity

在主表运行完成之后，sensitivity analysis 需要每个 domain 的
`eval_records_{base,aw}.jsonl`：

```bash
python examples/c7_sensitivity_analysis.py
```

这个步骤只是对 `0.25`、`0.30`、`0.35` 三个阈值做重新打分，不需要新增
LLM 调用。

## Where To Go Next

| 主题 | 建议先读 | 建议先跑 | 你会得到什么 |
|---|---|---|---|
| 方法 | `code/harness/agent.py`、`code/rl/offline_aw.py`、`code/reward/reward_aggregator.py` | `examples/anchor_4_reward_aggregator.py` | 理解 Base Harness、学习到的 controller 和 reward 设计 |
| 数据 | `scripts/bootstrap_mock_data.py`、`data/mock_data_summary.json` | `python scripts/bootstrap_mock_data.py --variant v2_heldout --overwrite` | 重新生成本地 synthetic benchmark 输入 |
| 结果 | `results/tables/main_table_v2_heldout_formal.md`、`examples/main_table_analysis.py` | `python examples/c7_sensitivity_analysis.py` | 查看发布指标、结果表和阈值重打分 |
| 复现 | `examples/running_main_driver.py` | 直接使用上面的 `v2_heldout` 命令块 | 按 domain 复现 `collect -> train -> eval` |
| Smoke test | `scripts/run_local_smoke.sh` | `bash scripts/run_local_smoke.sh` | 快速验证本地完整运行链路 |
| API 模式 | `examples/anchor_1_api_check.py`、`.env.example` | `.venv/bin/python examples/anchor_1_api_check.py` | 在长跑实验前确认 Azure-backed 模式配置正确 |
| Process metrics | `code/modules/hms_detector.py`、`examples/anchor_6_hms_detector.py` | `.venv/bin/python examples/anchor_6_hms_detector.py` | 查看 HMS 事件和 process-quality 检查逻辑 |

## 超参数

单一事实来源：`code/rl/offline_aw.py` 里的 `AWConfig`。

| 组件 | 值 |
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

相同的超参数被用于全部 8 个 synthetic domain 和 2 个 public benchmark，
没有做 per-domain tuning。

## Citation

```bibtex
@inproceedings{learningtoharness2026,
  title     = {Learning to Harness: Rubric-Guided Outer-Loop RL for Agentic Harness Optimization},
  author    = {Anonymous},
  year      = {2026},
}
```

## License

MIT. 见 `LICENSE`。
