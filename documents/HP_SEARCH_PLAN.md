# FedVPA-GP HP Search — N=10 and N=100 (2026-05-17)

Simplified HP search to find best hyperparameters for FedVPA-GP on Qwen-2 0.5B + HH-RLHF. **Single seed each** (multi-seed comes after the winners are picked).

## Why this and not the old HP search

The old doc swept 5+ knobs across ~40 configs per model. Most of that work has been done — Table 3 hyperparameters in `CLAUDE.md` are the result. The current investigation only needs to verify whether the 3 **most-likely-to-matter-now** knobs are at their right values for two new regimes (N=10 and N=100) that weren't separately tuned before.

## Knobs swept (3 only)

| Knob | Default | Why it matters | Sweep |
|---|---|---|---|
| `vpl_kl_weight` (β) | 0.01 | Posterior-collapse vs over-regularization. Most fundamental knob; the prior's effective weight changes with N. | {0.001, **0.01**, 0.1} |
| `vpl_orthogonal_weight` (λ) | 1.0 | Cluster separation strength. With only 2 prototypes and few clients, may be over- or under-pulling. | {0.5, **1.0**, 5.0} |
| `reward_coeff` (Stage 2) | 0.1 | How strongly z-rewarded responses pull the policy. Controls whether z-conditioning actually changes the policy. | {0.1, 0.5} |

Everything else stays at the current `fedvpagp_z_hybrid_adrop_kmeans_10212.yaml` / `hrl_z_hybrid_adrop_kmeans_11212.yaml` values.

## Config matrix (6 per N — 12 runs total per N including RL)

One-at-a-time variations from baseline + one combined-best. Selector and Stage-2 RL pair share the same name.

| Name | β | λ | reward_coeff | Notes |
|---|---|---|---|---|
| `baseline` | 0.01 | 1.0 | 0.1 | Current Table-3 defaults |
| `kl_low` | **0.001** | 1.0 | 0.1 | Weaker prior; lets posterior follow data |
| `kl_high` | **0.1** | 1.0 | 0.1 | Stronger prior; combats posterior collapse |
| `ortho_high` | 0.01 | **5.0** | 0.1 | Sharper category separation |
| `reward_high` | 0.01 | 1.0 | **0.5** | Stronger Stage-2 z-conditioning |
| `combined_best` | TBD | TBD | TBD | Combine winners of the above after they finish |

## TID assignment

| Server | N | Selector TIDs | RL TIDs |
|---|---|---|---|
| **This server (here)** | 100 | 90100..90105 | 91100..91105 |
| **Other server** | 10 | 90200..90205 | 91200..91205 |

`{0,1,2,3,4}` map to `{baseline, kl_low, kl_high, ortho_high, reward_high}`; `{5}` is `combined_best`.

## Compute estimate (1 seed, 5 configs + 1 combined)

Assuming the same per-run times we observed in the singlecell sanity check:

| | Selector (per run) | RL (per run) | Total wall-clock (5 GPUs in parallel) |
|---|---|---|---|
| **N=100** | ~5h | ~15h (5 evals × ~3h each) | ~25h (5 selectors → 5 RLs → 1 combined sel+RL) |
| **N=10** | ~5h | ~6h | ~17h |

If only one server is available for both N values, sequential ≈ 42h.

## Files to create (templates only — not yet generated)

For each config `{name}` and each N value:
- `cfg/main_table/qwen_hhrlhf/hp_search/sel_n{N}_{name}.yaml` — copy of `fedvpagp_z_hybrid_adrop_kmeans_10212.yaml` with the swept knobs overridden.
- `cfg/main_table/qwen_hhrlhf/hp_search/rl_n{N}_{name}.yaml` — copy of `hrl_z_hybrid_adrop_kmeans_11212.yaml` with the swept knobs overridden. Points to the matching selector ckpt.

Launch scripts:
- `scripts/server/hp_search/run_n100_selectors.sh` — launch all 5 selectors in parallel on GPUs 0-4.
- `scripts/server/hp_search/run_n100_rl.sh` — launch all 5 RLs in parallel after selectors finish (different GPU allocation for n=100 because of eval-time memory).
- `scripts/server/hp_search/run_n10_selectors.sh` — same idea for n=10 (other server).
- `scripts/server/hp_search/run_n10_rl.sh` — same.

## Decision rule

Pick the config that maximizes **sum(H + HH) on the Pareto-best round**, not on round 50. (Round 50 is rarely Pareto-optimal — see the single-cell sanity-check trajectories.)

If two configs tie on sum, prefer the one with the more balanced (H, HH) — i.e., closer to the 45° line.

## Combined-best (Tier 2)

After the 5 single-knob runs finish:
1. Identify the 2 configs that beat baseline.
2. If their winning knobs are compatible (e.g., `kl_high` and `ortho_high`), build a combined config that stacks them.
3. Train that as `combined_best` (TID `...05`).

If none beat baseline, the baseline IS the answer for that N; skip `combined_best`.

## After the HP search

- Multi-seed the winning config (≥3 seeds) before any paper claim.
- Then launch the full Main Table sweep at the winning hyperparameters.

## Bug-fix dependency

This HP search runs **after** the eval-time `client_average_z_dict` fix (category-aware fallback) is committed — without it, n=100 evals would suffer the same ~57% mis-z fallback that produced the bogus round-49 helpfulness collapse in 81312.
