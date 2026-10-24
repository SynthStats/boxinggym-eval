# BoxingGym sweeps

Create a Weights & Biases sweep, then run its agent with the printed ID:

```bash
uv run wandb sweep sweeps/paper_replication_oed.yaml
uv run wandb agent ENTITY/boxing-gym/SWEEP_ID
```

`include_prior` supplies domain knowledge; `use_ppl` adds PyMC model refinement. OED means optimal experimental design. Discovery tasks pass one agent's explanation to another.

## Settings

| Configuration | Search | Seeds per configuration |
| --- | --- | --- |
| [paper_replication_oed.yaml](paper_replication_oed.yaml) | Random | 1, 2, 3, 4, 5 |
| [paper_replication_oed_no_prior.yaml](paper_replication_oed_no_prior.yaml) | Random | 1, 2, 3, 4, 5 |
| [paper_replication_oed_ppl.yaml](paper_replication_oed_ppl.yaml) | Random | 1, 2, 3, 4, 5 |
| [paper_replication_oed_ppl_no_prior.yaml](paper_replication_oed_ppl_no_prior.yaml) | Random | 1, 2, 3, 4, 5 |
| [paper_replication_discovery.yaml](paper_replication_discovery.yaml) | Random | 1, 2, 3, 4, 5 |
| [paper_replication_lite_oed.yaml](paper_replication_lite_oed.yaml) | Grid | 1, 2 |
| [paper_replication_lite_discovery.yaml](paper_replication_lite_discovery.yaml) | Grid | 1, 2 |

## Task configurations

| Task | Goal | Configuration |
| --- | --- | --- |
| dugongs_direct | Length prediction | [dugongs_direct.yaml](../conf/envs/dugongs_direct.yaml) |
| peregrines_direct | Population prediction | [peregrines_direct.yaml](../conf/envs/peregrines_direct.yaml) |
| lotka_volterra_direct | Population dynamics | [lotka_volterra_direct.yaml](../conf/envs/lotka_volterra_direct.yaml) |
| hyperbolic_direct | Choice prediction | [hyperbolic_direct.yaml](../conf/envs/hyperbolic_direct.yaml) |
| hyperbolic_discount | Discount rate | [hyperbolic_discount.yaml](../conf/envs/hyperbolic_discount.yaml) |
| location_finding_direct | Signal prediction | [location_finding_direct.yaml](../conf/envs/location_finding_direct.yaml) |
| location_finding_source | Source location | [location_finding_source.yaml](../conf/envs/location_finding_source.yaml) |
| death_process_direct | Infection count | [death_process_direct.yaml](../conf/envs/death_process_direct.yaml) |
| death_process_infection | Infection rate | [death_process_infection.yaml](../conf/envs/death_process_infection.yaml) |
| irt_direct | Correctness prediction | [irt_direct.yaml](../conf/envs/irt_direct.yaml) |
| survival_direct | Survival prediction | [survival_direct.yaml](../conf/envs/survival_direct.yaml) |
| emotion_direct | Emotion prediction | [emotion_direct.yaml](../conf/envs/emotion_direct.yaml) |
| morals_direct | Moral judgement | [morals_direct.yaml](../conf/envs/morals_direct.yaml) |

[Commands](../README.md) · [SynthStats](https://synthstats.org)
