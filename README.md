# SynthStats multi-model evaluation of BoxingGym

<p align="center">
  <a href="https://arxiv.org/abs/2501.01540"><img src="https://img.shields.io/badge/arXiv-2501.01540-b31b1b.svg?style=flat-square" alt="arXiv"></a>&nbsp;
  <a href="https://huggingface.co/spaces/youkad/boxing-gym-dashboard"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HF%20Space-Live%20Demo-yellow?style=flat-square" alt="HF Space"></a>&nbsp;
  <a href="https://github.com/SynthStats/boxinggym-eval/actions/workflows/ci.yml"><img src="https://github.com/SynthStats/boxinggym-eval/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

<p align="center">
  <em>The SynthStats copy of a fork of <a href="https://github.com/kanishkg/boxing-gym">Stanford's BoxingGym</a> that runs its environments on models from several providers and compares the runs</em>
</p>

In the temporal discounting environment of [BoxingGym](https://github.com/kanishkg/boxing-gym), a language model chooses the offers put to a simulated person: some dollars today or a larger sum after a delay, with both amounts and the number of days picked by the model, until its budget of offers runs out and it has to predict the person's choices on new ones. Gandhi et al. wrote ten such environments, described in [README_upstream.md](README_upstream.md), to measure how well language-model agents design experiments and discover models of what they observe.

## Results

<!-- CANONICAL_METADATA:START -->
> Canonical snapshot generated: **2026-02-13T21:14:31Z**
> Source: `.boxing-gym-cache/runs.parquet` @ `a0211d7c6e83300cb1216d4789b084a910523083`
> Filters: `budget >= 10`, `exclude_outliers=true`
> Valid runs: **1,948** across **7 models** and **10 environments**
<!-- CANONICAL_METADATA:END -->

Each run is scored by the standardized error z of the BoxingGym paper: the run's prediction error, minus the mean error of a reference predictor (the baseline in the tables below), divided by the standard deviation of that predictor's error; for the discounting choices above, the reference predictor gives the same answer to every offer, the delayed or the immediate reward, whichever simulated people take more often. In the third table, `use_ppl = true` marks runs in which the model also wrote [PyMC](https://www.pymc.io/) models of its data and revised them (Box's Loop, see README_upstream.md).

<!-- CANONICAL_RESULTS:START -->
### Overall Model Rankings

Mean z-score across all filtered runs (lower is better):

| Rank | Model | Mean z | 95% CI | vs #1 (FDR p) | Runs |
| --- | --- | --- | --- | --- | --- |
| 1 | MiniMax-M2.1 | +0.185 | [+0.071, +0.315] | — | 311 |
| 2 | GPT-4o | +0.256 | [+0.138, +0.388] | 0.424 | 226 |
| 3 | Kimi-K2 | +0.262 | [+0.130, +0.408] | 0.424 | 302 |
| 4 | Qwen3-32B | +0.275 | [+0.115, +0.464] | 0.424 | 188 |
| 5 | DeepSeek-V3.2 | +0.296 | [+0.122, +0.532] | 0.424 | 369 |
| 6 | GLM-4.7 | +0.546 | [+0.257, +0.904] | 0.120 | 314 |
| 7 | GPT-5.1-Codex-Mini | +0.586 | [+0.310, +0.966] | 0.120 | 238 |

### Per-Environment Champions (Best Model Only)

_Definition: mean over all runs for each model within environment (ignores `use_ppl`)._

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

_Definition: mean over each `(model, use_ppl)` combination within environment._

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

- **MiniMax-M2.1** leads overall with z=+0.185
- **6/10** environments beat baseline under model-only definition
- Models significantly worse than #1 (FDR < 0.05): None

### Statistical Methods

- **Welch's t-test**: Group comparison with unequal variances
- **Bootstrap CI**: 95% confidence intervals from resampling
- **Benjamini-Hochberg**: FDR correction for multiple comparisons

_Snapshot version: `2026-02-13`_
<!-- CANONICAL_RESULTS:END -->

## Running an experiment

The code needs [Python](https://www.python.org/downloads/) 3.11 or later. Install it with [uv](https://github.com/astral-sh/uv), and put an API key in `.env` for each provider whose models you run (OpenAI and DeepSeek below):

```bash
git clone https://github.com/SynthStats/boxinggym-eval.git
cd boxinggym-eval

# Install with uv
uv sync
uv run python -c "import boxing_gym; print('✓ Import OK')"

# Set API keys (create .env with your keys)
echo "OPENAI_API_KEY=sk-..." >> .env
echo "DEEPSEEK_API_KEY=sk-..." >> .env

# Run single experiment
uv run python run_experiment.py seeds=null seed=1 exp=oed envs=hyperbolic_direct llms=gpt-4o

# Run full test suite
uv run pytest tests/ -v
```

Models are called through [LiteLLM](https://github.com/BerriAI/litellm), so changing `llms=gpt-4o` to `llms=deepseek-v3.2` in the `run_experiment.py` line runs the same experiment on a DeepSeek model.

## Analysing runs

Each run is saved as a JSON file under `results/`. `box sync` parses those files into a Parquet cache, which every `box query` then reads, so sync again when new runs finish:

```
┌─────────────────┐    ┌──────────────────────────┐    ┌─────────────────┐
│  Local JSON     │───▶│  box sync --local        │───▶│  Parquet Cache  │
│  results/*.json │    │  (aggregates & caches)   │    │  .boxing-gym-   │
└─────────────────┘    └──────────────────────────┘    │   cache/        │
                                                       └────────┬────────┘
                                                                │
┌─────────────────┐    ┌──────────────────────────┐             │
│  Analysis       │◀───│  box query <name>        │◀────────────┘
│  (tables, stats)│    │  (runs pre-built queries)│
└─────────────────┘    └──────────────────────────┘
```

```bash
uv run box sync --local results/    # parse JSON → .boxing-gym-cache/runs.parquet
uv run box sync --status            # check current cache counts
```

Then query it by name:

```bash
uv run box query --list             # list available queries
uv run box query leaderboard        # model rankings with significance tests
uv run box query oed-discovery      # OED vs Discovery 2×2 comparison
uv run box query ppl-impact         # PPL effect analysis
uv run box query env-difficulty     # environment difficulty rankings
uv run box query best-configs       # best config per environment
uv run box query paper-comparison   # comparison with paper baselines
uv run box query all                # full report (all queries)
```

Each one can be filtered and printed as Markdown or JSON:

```bash
uv run box query leaderboard --min-budget 10     # filter by minimum budget
uv run box query leaderboard --env dugongs       # filter by environment
uv run box query leaderboard --format md         # markdown output
uv run box query leaderboard --format json       # JSON output
uv run box query all --include-outliers          # include flagged outliers
```

`box results --web` opens a [Streamlit](https://streamlit.io) dashboard, which can read the JSON files in `results/` or runs stored on W&B, with the pages listed below (in the [Hugging Face demo](https://huggingface.co/spaces/youkad/boxing-gym-dashboard), the Leaderboard page defaults to the snapshot above):

```bash
uv run box results --web
```

| Page | Description |
|------|-------------|
| Leaderboard | Model rankings with bootstrap 95% CIs and significance testing |
| Benchmark Dashboard | Paper baselines (GPT-4o, BOX), filtering, delta coloring |
| Sweep Analysis | Parameter importance, model rankings, heatmaps, best configs |
| Paper Comparison | Gandhi et al. baseline comparison |
| PPL Examples | Generated PyMC models with sampling diagnostics |
| LLM Call Logs | Cost breakdown by model, latency histograms, token counts |

In a terminal, `box results --tui` shows the cache in these views instead:

```bash
uv run box results --tui
```

| View | Description |
|------|-------------|
| model-rankings | Leaderboard: z-scores, bootstrap CIs, significance |
| heatmap | Environment × Model z-mean matrix |
| best-configs | Top configuration per environment |
| budget-progression | Performance trends across budget values |
| parameter-importance | Permutation importance analysis |
| seed-stability | Cross-seed variance, unstable configs |
| local-summary | Quick summary of local results |

The Makefile has shortcuts for the commonest commands:

```bash
make aggregate   # box sync --local results/
make plot        # box query all
make clean       # rm outputs/ .boxing-gym-cache/
```

### Refreshing the results above

One script regenerates the two blocks of results above, and the summary in `HF_SPACE_README.md`, from the cache:

```bash
uv run box sync --local results/
uv run python scripts/refresh_canonical_results.py
```

## Configuration

Settings are Hydra configs, and any of them can be overridden on the command line (`+wandb=true` also logs the run to [Weights & Biases](https://wandb.ai/site)):

```bash
# Override any config parameter
uv run python run_experiment.py \
  envs=hyperbolic_direct \
  exp=oed \
  llms=deepseek-v3.2 \
  seeds=[1,2,3,4,5] \
  +wandb=true
```

Environment configs live in `conf/envs/*.yaml`, two goal types for each of the 10 environments. The 19 models (from 9 providers) have theirs in `conf/llms/*.yaml`, and `conf/exp/{oed,oed_box,discovery}.yaml` holds the experiment types: `oed` is the setting described at the top, and in `discovery` the model also writes an explanation for a second model, which then predicts without running any experiments of its own. Weights & Biases sweeps are defined in `sweeps/*.yaml`, and several sweep agents can work through one sweep in parallel.

## Credits

BoxingGym is the work of Kanishk Gandhi, Michael Y. Li, Dorsa Sadigh and Noah D. Goodman at Stanford University, described in [BoxingGym: Benchmarking Progress in Automated Experimental Design and Model Discovery](https://arxiv.org/abs/2501.01540), with the original code at [github.com/kanishkg/boxing-gym](https://github.com/kanishkg/boxing-gym). This fork keeps their MIT licence. Cite the original paper:

```bibtex
@article{gandhi2025boxinggym,
  title={BoxingGym: Benchmarking Progress in Automated Experimental Design and Model Discovery},
  author={Gandhi, Kanishk and Li, Michael Y and Sadigh, Dorsa and Goodman, Noah D},
  journal={arXiv preprint arXiv:2501.01540},
  year={2025}
}
```
