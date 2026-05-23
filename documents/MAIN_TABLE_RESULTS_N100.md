# N=100 Main Table — results (2026-05-23)

Qwen-2 0.5B + HH-RLHF, 4 methods × N=100, single seed, GPT-4o-mini
100-sample win-rate eval at every save_freq (10 rounds), 70 total
rounds. Reported = **Pareto-best round per method** (max H+HH).

## Headline (Pareto-best per method)

| Method | Best round | H | HH | Σ |
|---|---|---|---|---|
| **FedVPA-GP full (ours)** | rd 29 | **39.0** | **79.0** | **118.0** 🥇 |
| FedBiscuit | rd 19 | 43.0 | 73.0 | 116.0 |
| FedVPL | rd 49 | 38.0 | 73.0 | 111.0 |
| FedVPA-GP kl_only | rd 49 | 31.0 | 69.0 | 100.0 |

**Read:** FedVPA-GP-full wins the N=100 cell by Σ (+2 vs FedBiscuit,
+7 vs FedVPL, +18 vs kl_only). The win is **Pareto-defensible** —
better than FedBiscuit on HH (+6), comparable on H (−4). Same model
checkpoint, no cherry-picking helpful vs harmless rounds.

## Full trajectories

| Method | rd9 | rd19 | rd29 | rd39 | rd49 | rd59 | rd69 |
|---|---|---|---|---|---|---|---|
| FedBiscuit | 40/71 | **43/73** | 31/75 | 28/76 | 31/72 | 41/72 | 34/68 |
| **FedVPA-GP full** | 30/74 | 34/69 | **39/79** | 39/78 | 40/77 | 39/78 | 36/74 |
| FedVPL | 20/69 | 24/70 | 24/75 | 33/74 | **38/73** | 26/80 | 26/79 |
| FedVPA-GP kl_only | 28/69 | 29/64 | 28/64 | 28/67 | **31/69** | 28/68 | 25/66 |

**Trajectory shapes**:
- FedBiscuit: classic early peak (rd 19) then drift down — matches HP search prediction.
- FedVPA-GP full: rises 9→29, **stable plateau** at Σ 117–118 through rd 59, slight regression at rd 69. Wide Pareto-optimal region (rd 29–59 all ≥ Σ 117).
- FedVPL: late peak at rd 49.
- FedVPA-GP kl_only: noisy, mid peak.

## Ckpt locations

Recommended ckpts (Pareto-best round per method):
- FedBiscuit: `/hdd/hdd3/kjb/checkpoints/20_hhrl_rlhf_qwen2_mt_n100_fedbiscuit_t11200.ckpt`
- **FedVPA-GP full**: `/hdd/hdd3/kjb/checkpoints/30_hhrl_rlhf_qwen2_mt_n100_kl_ortho_z_t11203.ckpt` ⭐
- FedVPL: `/hdd/hdd3/kjb/checkpoints/50_hhrl_rlhf_qwen2_mt_n100_fedvpl_z_t11201.ckpt`
- FedVPA-GP kl_only: `/hdd/hdd3/kjb/checkpoints/50_hhrl_rlhf_qwen2_mt_n100_kl_only_z_t11202.ckpt`

WandB:
- FedBiscuit: gahklo86
- FedVPL: 83mbugwv
- kl_only: yfb5zrnx
- **kl_ortho (winner)**: io2mr94s

## What it took to get here

Three stacked bugs were uncovered and fixed during this cell. None
of the earlier "collapse" runs are part of the result above; only the
final retry after all fixes were in place.

| Bug | Fix | Commit |
|---|---|---|
| `lr=5e-5`/`rc=0.5` (N=10 winner) explodes z-conditioned DPO at N=100 (train_loss ~3000 vs ~2; helpful collapses to 0) | revert RL cfgs to `lr=1e-5, rc=0.1` | `8e99557` |
| RL cfgs missing `vpl_use_z_embedding` / `vpl_z_source` / `vpl_z_conditioning_mode` / `vpl_adapter_dropout` — z never injected into policy | added to RL cfgs | `2232877` |
| Selector cfgs also missing them — ckpt has no `z_to_embedding` layer | added to selector cfgs + retrained selectors | `f46d334` |
| z-conditioning OOMs at `batch_size=8` | halved to 4 (other server's fix) | `1c9a694` |
| GPT-API eval cap (`max_iterations=1000`) too low for 4447-prompt per-client test loader | raised to 6000 | `a03661c` |

Total wasted compute on the bug chain: ~3 days. The Main Table
comparison cfg family (`*_comparison_*`) was created as a separate
config family from the HP-search-validated cfg family
(`*_z_hybrid_adrop_*`) and silently dropped the z-conditioning
architecture. They are now consistent.

## What to do next

1. **Multi-seed (≥3) the FedVPA-GP-full N=100 cell** — single-seed
   Σ-margin of +2 over FedBiscuit needs seed averaging before it's
   defensible as a paper claim. ~10h per extra seed.
2. **Other server's N=10 Main Table** — same cfg family, expected to
   complete this week.
3. **Optionally bump eval to 300 samples** for the final paper
   numbers (CIs ±5.7pp vs ±9.8pp at 100). Cheap (~$5/eval vs $1.50).
4. **Gemma-2B Main Table** — needs new cfgs (none exist yet), then
   selectors, then RL.

## Caveats

- Single seed; ±9.8pp CI per metric at 100 samples; **the Σ 118 vs
  116 margin is within noise** until multi-seeded.
- Round-29 ckpt is the Pareto-best for FedVPA-GP-full; the round-70
  ckpt that the default `save_freq: 50` would have saved (had we not
  changed it) is Σ 110 — confirms the N=10 finding that final-round
  ckpts are systematically worse.
