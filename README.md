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
| [Data](scripts/streamlit_app/demo_data/canonical_runs.parquet), [metrics](scripts/streamlit_app/demo_data/canonical_metrics.json), [metadata](scripts/streamlit_app/demo_data/canonical_metadata.json) | 2026-10-26 snapshot; 1,948 runs, 7 models, 10 environments; budget ≥ 10; outliers excluded |
| Snapshot scores | Mean standardised prediction error; lower is better; 95% bootstrap confidence intervals |
| `box sync --local results/` | Cache local JSON runs as Parquet |
| `box query --list` | List analyses: rankings, experiment settings and environment difficulty |
| `box results --tui` | Terminal rankings, heatmaps, budget trends, parameter importance and seed stability |
| `uv run box results --web` | Streamlit rankings, paper comparisons, sweep analysis, PyMC diagnostics and call costs |
| [Experiment runner](run_experiment.py) | API keys in `.env` for selected providers |
| [Environment configs](conf/envs/), [model configs](conf/llms/), [experiment configs](conf/exp/), [sweeps](sweeps/) | Hydra settings and Weights & Biases sweeps |
| [Snapshot refresh](scripts/refresh_canonical_results.py) | Regenerate the bundled snapshot and its result tables |
| [Makefile](Makefile) | `make aggregate` caches results; `make plot` prints analyses |
| Tools | [Python](https://www.python.org/downloads/) 3.11+, [uv](https://github.com/astral-sh/uv), [PyMC](https://www.pymc.io/), [LiteLLM](https://github.com/BerriAI/litellm), [Streamlit](https://streamlit.io), [Weights & Biases](https://wandb.ai/site) |
| [Tests](https://github.com/SynthStats/boxinggym-eval/actions/workflows/ci.yml) | `uv run pytest tests/ -v` |

## Results

<!-- CANONICAL_METADATA:START -->
| Snapshot | Value |
| --- | --- |
| Snapshot version | `2026-10-26` |
| Bundled snapshot | `scripts/streamlit_app/demo_data/canonical_runs.parquet` |
| Filters | `budget >= 10`, `exclude_outliers=true` |
| Coverage | 1,948 runs across 7 models and 10 environments |
<!-- CANONICAL_METADATA:END -->

<!-- CANONICAL_RESULTS:START -->
### Matched and pooled scores in this snapshot

| Model | Matched mean z | 95% environment-cluster interval | Matched rows | Pooled mean z | Pooled rows |
| --- | --- | --- | --- | --- | --- |
| DeepSeek-V3.2 | +0.245 | [-0.194, +0.760] | 295 | +0.296 | 369 |
| GPT-4o | +0.258 | [-0.126, +0.648] | 157 | +0.256 | 226 |
| MiniMax-M2.1 | +0.262 | [-0.128, +0.694] | 273 | +0.185 | 311 |
| GPT-5.1-Codex-Mini | +0.279 | [-0.158, +0.771] | 174 | +0.586 | 238 |
| Kimi-K2 | +0.352 | [-0.105, +0.863] | 271 | +0.262 | 302 |
| GLM-4.7 | +0.405 | [-0.127, +1.061] | 260 | +0.546 | 314 |
| Qwen3-32B | +0.506 | [+0.040, +0.993] | 165 | +0.275 | 188 |

### Lowest mean score in each environment

| Environment | Model | Mean z | Status |
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

### Lowest mean score by model and PyMC setting

| Environment | Model | PyMC refinement | Mean z | Status |
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

### Snapshot comparisons

| Finding | Value |
| --- | --- |
| Environments beating baseline (model only) | 6/10 |

### Definitions

| Result | Definition |
| --- | --- |
| Matched comparison | 20 shared cells across 10 environments; budget 10 |
| Pooled mean score | Mean standardised prediction error across each model's filtered runs; lower is better |
| Matched cell | Same environment, mode, budget, PyMC setting, prior, goal, evaluation size, simulator parameters, communication limit, information-gain setting, budget schedule, original goal, normalisation, temperature, configured and observed seed lists, aggregation mode and seed count; finite scores from all seven models |
| Matched mean score | Mean over records within each cell, cells within each environment, then environments with equal weight |
| Model within environment | Mean over all runs for each model within an environment, pooling PyMC settings |
| Model and PyMC setting | Mean over each `(model, use_ppl)` combination within an environment; `use_ppl=true` adds PyMC model refinement (Box's Loop) |

### Statistical methods

| Method | Definition |
| --- | --- |
| Environment-cluster bootstrap | 95% percentile intervals from 200,000 resamples of exchangeable environments; seed 20261026 |

| Snapshot | Value |
| --- | --- |
| Snapshot version | `2026-10-26` |
<!-- CANONICAL_RESULTS:END -->

[SynthStats](https://synthstats.org)

Licence: [MIT](LICENSE), as upstream.
