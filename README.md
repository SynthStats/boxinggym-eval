# boxinggym-eval

This fork runs BoxingGym experiments on multiple language models and compares their prediction errors.

## Install

```bash
git clone https://github.com/SynthStats/boxinggym-eval.git
cd boxinggym-eval
uv sync --extra dev
```

## Example

```bash
uv run python run_experiment.py seeds=null seed=1 exp=oed envs=hyperbolic_direct llms=gpt-4o
```

## Contents

| Resource | Contents |
| --- | --- |
| [Paper](https://arxiv.org/abs/2501.01540) | BoxingGym: Benchmarking Progress in Automated Experimental Design and Model Discovery |
| [Upstream](https://github.com/kanishkg/boxing-gym), [documentation](README_upstream.md) | Gandhi, Li, Sadigh and Goodman, Stanford University |
| [Dashboard](https://huggingface.co/spaces/youkad/boxing-gym-dashboard) | Model rankings, environment comparisons and PyMC examples |
| [Data](scripts/streamlit_app/demo_data/canonical_runs.parquet), [metrics](scripts/streamlit_app/demo_data/canonical_metrics.json), [metadata](scripts/streamlit_app/demo_data/canonical_metadata.json) | 2026-10-24 snapshot; 1,948 runs, 7 models, 10 environments; budget ≥ 10; outliers excluded |
| Snapshot scores | Mean standardized prediction error; lower is better; 95% bootstrap confidence intervals |
| Snapshot comparisons | Welch's t-test; Benjamini-Hochberg adjustment for false discovery rate |
| `box sync --local results/` | Cache local JSON runs as Parquet |
| `box query --list` | List analyses: rankings, experiment settings and environment difficulty |
| `box results --tui` | Terminal rankings, heatmaps, budget trends, parameter importance and seed stability |
| `uv run box results --web` | Streamlit rankings, paper comparisons, sweep analysis, PyMC diagnostics and call costs |
| [Experiment runner](run_experiment.py) | API keys in `.env` for selected providers |
| [Environment configs](conf/envs/), [model configs](conf/llms/), [experiment configs](conf/exp/), [sweeps](sweeps/) | Hydra settings and Weights & Biases sweeps |
| [Snapshot refresh](scripts/refresh_canonical_results.py) | Rebuild data and metrics from `.boxing-gym-cache/runs.parquet` |
| [Makefile](Makefile) | `make aggregate` caches results; `make plot` prints analyses |
| Tools | [Python](https://www.python.org/downloads/) 3.11+, [uv](https://github.com/astral-sh/uv), [PyMC](https://www.pymc.io/), [LiteLLM](https://github.com/BerriAI/litellm), [Streamlit](https://streamlit.io), [Weights & Biases](https://wandb.ai/site) |
| [Tests](https://github.com/SynthStats/boxinggym-eval/actions/workflows/ci.yml) | `uv run pytest tests/ -v` |

## Results

<!-- CANONICAL_METADATA:START -->
| Snapshot | Value |
| --- | --- |
| Snapshot version | `2026-10-24` |
| Source cache | `.boxing-gym-cache/runs.parquet` |
| Filters | `budget >= 10`, `exclude_outliers=true` |
| Coverage | 1,948 runs across 7 models and 10 environments |
<!-- CANONICAL_METADATA:END -->

<!-- CANONICAL_RESULTS:START -->
### Overall Model Rankings

| Rank | Model | Mean z | 95% CI | vs #1 (FDR p) | Runs |
| --- | --- | --- | --- | --- | --- |
| 1 | MiniMax-M2.1 | +0.185 | [+0.071, +0.315] | Reference | 311 |
| 2 | GPT-4o | +0.256 | [+0.138, +0.388] | 0.424 | 226 |
| 3 | Kimi-K2 | +0.262 | [+0.130, +0.408] | 0.424 | 302 |
| 4 | Qwen3-32B | +0.275 | [+0.115, +0.464] | 0.424 | 188 |
| 5 | DeepSeek-V3.2 | +0.296 | [+0.122, +0.532] | 0.424 | 369 |
| 6 | GLM-4.7 | +0.546 | [+0.257, +0.904] | 0.120 | 314 |
| 7 | GPT-5.1-Codex-Mini | +0.586 | [+0.310, +0.966] | 0.120 | 238 |

### Per-Environment Champions (Best Model Only)

| Environment | Best Model | Mean z | Status |
| --- | --- | --- | --- |
| death_process | GPT-4o | -0.903 | ✓ Beats baseline |
| lotka_volterra | DeepSeek-V3.2 | -0.301 | ✓ Beats baseline |
| hyperbolic_temporal_discount | DeepSeek-V3.2 | -0.152 | ✓ Beats baseline |
| moral_machines | GPT-5.1-Codex-Mini | -0.140 | ✓ Beats baseline |
| irt | GPT-5.1-Codex-Mini | -0.050 | ✓ Beats baseline |
| dugongs | Qwen3-32B | -0.040 | ✓ Beats baseline |
| peregrines | DeepSeek-V3.2 | +0.094 | Above baseline |
| survival | GLM-4.7 | +0.434 | Above baseline |
| emotion | Qwen3-32B | +1.286 | Above baseline |
| location_finding | GPT-4o | +1.396 | Above baseline |

### Per-Environment Champions (Best Model + PPL)

| Environment | Best Model | use_ppl | Mean z | Status |
| --- | --- | --- | --- | --- |
| death_process | Qwen3-32B | true | -1.170 | ✓ Beats baseline |
| lotka_volterra | DeepSeek-V3.2 | false | -0.301 | ✓ Beats baseline |
| hyperbolic_temporal_discount | DeepSeek-V3.2 | false | -0.159 | ✓ Beats baseline |
| moral_machines | GPT-5.1-Codex-Mini | false | -0.140 | ✓ Beats baseline |
| irt | GPT-4o | true | -0.120 | ✓ Beats baseline |
| dugongs | GPT-4o | true | -0.047 | ✓ Beats baseline |
| location_finding | GPT-4o | true | -0.041 | ✓ Beats baseline |
| peregrines | DeepSeek-V3.2 | false | +0.094 | Above baseline |
| survival | GLM-4.7 | false | +0.429 | Above baseline |
| emotion | Qwen3-32B | false | +1.286 | Above baseline |

### Key Findings

| Finding | Value |
| --- | --- |
| Overall leader | MiniMax-M2.1 |
| Leader mean z | +0.185 |
| Environments beating baseline (model only) | 6/10 |
| Models significantly worse than #1 (FDR < 0.05) | None |

### Definitions

| Result | Definition |
| --- | --- |
| Overall ranking | Mean z-score across all filtered runs; lower is better |
| Model-only champion | Mean over all runs for each model within environment; ignores `use_ppl` |
| Model + PPL champion | Mean over each `(model, use_ppl)` combination within environment |

### Statistical Methods

| Method | Definition |
| --- | --- |
| Welch's t-test | Group comparison with unequal variances |
| Bootstrap CI | 95% confidence intervals from resampling |
| Benjamini-Hochberg | FDR correction for multiple comparisons |

| Snapshot | Value |
| --- | --- |
| Snapshot version | `2026-10-24` |
<!-- CANONICAL_RESULTS:END -->

[SynthStats](https://synthstats.org)

Licence: [MIT](LICENSE), as upstream.
