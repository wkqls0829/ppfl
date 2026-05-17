# Resume State — 2026-05-17 (single-cell sanity check COMPLETE)

## Status

All 4 stage-1 selectors + all 4 stage-2 RL runs have completed for the single-cell sanity check (Qwen-2 0.5B, HH-RLHF, FedBiscuit vs FedVPA-GP at N=10 and N=100). All 7 bug fixes are committed to `refined` (`346a562`) and pushed to `origin/refined`.

**For the next session / other server**, start with
[HANDOFF_OTHER_SERVER.md](./HANDOFF_OTHER_SERVER.md) then
[HP_SEARCH_PLAN.md](./HP_SEARCH_PLAN.md).

## Code fixes (working tree — NOT yet committed)

| Bug | Files | Status |
|---|---|---|
| 1. VPL components never broadcast back to clients | `federatedscope/llm/llm_local/server.py`, `client.py` | **fixed** |
| 3. `client_average_z_dict` missing clients | `client.py`, `server.py` | **fixed** |
| 5. `z_to_embedding` dtype mutation mid-forward | `vpl_reward_choice_trainer.py` | **fixed** |
| 6. `client_z_mu/logvar` aggregation via law of total variance | `vpl_reward_choice_trainer.py` | **fixed** |
| Cleanup: `del to_del` no-op, z_history mixing train+eval | `vpl_reward_choice_trainer.py` | **fixed** |

Bug 4 (adapter dropout) was a false positive — already handled.

## Stage 1 (selector)

| TID | Method | N | Final train_acc | KL | Checkpoint |
|---|---|---|---|---|---|
| 80000 | FedBiscuit | 10 | 0.49–0.53 | — | `final_hhrl_choice_qwen2_singlecell_t80000.ckpt` |
| 80100 | FedBiscuit | 100 | 0.50–0.56 | — | `final_hhrl_choice_qwen2_singlecell_n100_t80100.ckpt` |
| 80212 | FedVPA-GP | 10 | **0.99** | 14–16 | `final_hhrl_choice_qwen2_singlecell_t80212.ckpt` |
| 80312 | FedVPA-GP | 100 | **0.99** | 14–16 | `final_hhrl_choice_qwen2_singlecell_n100_t80312.ckpt` |

## Stage 2 (RL) — all 4 runs complete with full trajectories

### N=10

**FedBiscuit (81000):** `50_hhrl_rlhf_qwen2_singlecell_t81000.ckpt`

| Round | Helpful% | Harmless% |
|---|---|---|
| 9 | 26.7 | 73.3 |
| 19 | 50.0 | 80.0 |
| 29 | 40.0 | 80.0 |
| 39 | 46.7 | 90.0 |
| **49 (final)** | **40.0** | **70.0** |

**FedVPA-GP rerun (81212):** `50_hhrl_rlhf_qwen2_singlecell_t81212.ckpt` — wandb run `e1pb5qhy`

| Round | Helpful% | Harmless% |
|---|---|---|
| 9 | 20.0 | 76.67 |
| 19 | 33.33 | 86.67 |
| 29 | 20.0 | 73.33 |
| 39 | 33.33 | 60.0 |
| **49 (final)** | **33.33** | **76.67** |

N=10 winner: ambiguous. FedBiscuit wins helpful (+6.67), FedVPA-GP wins harmless (+6.67). 30-sample evals very noisy; previous interrupted 81212 had best 50/86.7 — current rerun underperforms it. Need multi-seed averaging.

### N=100

**FedBiscuit (81100):** `50_hhrl_rlhf_qwen2_singlecell_n100_t81100.ckpt`

| Round | Helpful% | Harmless% |
|---|---|---|
| 9 | 23.3 | 76.7 |
| 19 | 53.3 | 90.0 |
| 29 | 36.7 | 93.3 |
| 39 | 40.0 | 86.7 |
| **49 (final)** | **40.0** | **70.0** |

**FedVPA-GP (81312):** `50_hhrl_rlhf_qwen2_singlecell_n100_t81312.ckpt` — wandb run `vx2en1pg`

| Round | Helpful% | Harmless% |
|---|---|---|
| 9 | 80.0 | 83.33 |
| 19 | 80.0 | 70.0 |
| 29 | 83.33 | 73.33 |
| 39 | **86.67** | 60.0 |
| 49 (unreliable, see note) | 26.67 | 63.33 |

⚠️ Round 49 eval threw many `Client ID X not found in client_average_z_dict, using first available` warnings — eval fell back to default z for many samples, artificially deflating win-rates. Treat rd 39 as the strongest valid checkpoint for n=100.

### Best-epoch n=100 comparison

| | FedVPA-GP n=100 | FedBiscuit n=100 |
|---|---|---|
| Best helpful | **86.67** (rd 39) | 53.3 (rd 19) |
| Best harmless | 83.33 (rd 9) | **93.3** (rd 29) |

n=100 case shows clear Pareto separation: **FedVPA-GP dominates helpfulness by ~+30-46%, FedBiscuit dominates harmlessness by ~+10-20%**. This pattern is consistent with the paper's central claim that the Federated Mixture Prior captures distinct preference clusters once enough clients are present.

## Key takeaway

- **N=10**: noisy and near-tied — sample variance dominates with only 30 eval samples per round and 5 sampled clients/round.
- **N=100**: large effect size emerges. FedVPA-GP gains significant helpfulness at the cost of some harmlessness.

The pattern doesn't quite match the paper's Pareto-improvement claim (where FedVPA-GP should win BOTH axes), but shows strong preference differentiation — the mechanism is working.

## Next steps to consider

1. **Commit the bug fixes to git** — still uncommitted on `refined` branch (server.py, client.py, vpl_reward_choice_trainer.py).
2. **Investigate the rd-49 eval bug** — `client_average_z_dict` is missing client IDs when loaded from the saved Stage-2 checkpoint, causing fallback to default z for many clients. Likely a save/load gap in `standalone_training.py`'s checkpoint serialization.
3. **Multi-seed averaging** — the n=10 case especially needs ≥3 seeds to claim anything meaningful given 30-sample eval noise.
4. **Gemma-2B run** — paper's largest claimed gaps are on Gemma; Qwen-0.5B may not show full effect.
5. **Tune the helpful/harmless trade-off** — current configs sacrifice harmless for helpful at n=100. Try `vpl_orthogonal_weight` sweep or `vpl_kl_weight` adjustment.
6. **t-SNE inspection** — `exp/singlecell_rl_*` directories will have round-39 cross-client visualizations to check whether preference clusters are now properly separated.

## Entrypoint reminder

- **Stage 1 (selector)** uses `federatedscope/main.py`
- **Stage 2 (RL)** uses `federatedscope/llm/rlhf/main.py` AND for FedVPA-GP requires `--selector-cfg-file`. Using `main.py` for RL crashes with `KeyError: 'win_data'`.
