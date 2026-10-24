---
title: BoxingGym Dashboard
emoji: 🥊
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.39.0"
app_file: scripts/streamlit_app/app.py
pinned: false
license: mit
---

# BoxingGym dashboard

BoxingGym agents choose experiments, such as offering a simulated person money now or later, then predict choices on new offers. Compare their prediction errors here.

<!-- CANONICAL_HF_SNAPSHOT:START -->
## Snapshot

| Snapshot | Value |
| --- | --- |
| Generated | 2026-10-24T04:57:05Z |
| Bundled snapshot | `scripts/streamlit_app/demo_data/canonical_runs.parquet` |
| Filters | `budget >= 10`, `exclude_outliers=true` |
| Coverage | 1,948 runs across 7 models and 10 environments |
| Lowest pooled mean | MiniMax-M2.1 (`z=+0.185`) |
| Leaderboard | Bundled snapshot |
| Sweep analysis | Selected Weights & Biases sweeps |
<!-- CANONICAL_HF_SNAPSHOT:END -->

For live Weights & Biases sweeps, add `WANDB_API_KEY` in Space settings.

[Evaluation code](https://github.com/SynthStats/boxinggym-eval) · [BoxingGym](https://github.com/kanishkg/boxing-gym) · [Paper](https://arxiv.org/abs/2501.01540) · [SynthStats](https://synthstats.org)
