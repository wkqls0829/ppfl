# N=10 HP search — tier-1 results & tier-2 plan (2026-05-19)

This server (local) ran the N=10 HP search per
[HP_SEARCH_PLAN.md](./HP_SEARCH_PLAN.md). All five selectors finished
cleanly; all five Stage-2 RL runs reached round 49 and saved final
checkpoints. The eval numbers are honest GPT-API (the 2026-05-19
silent-fallback bug only triggers at N=100 — see
[BUGFIX_2026_05_19.md](./BUGFIX_2026_05_19.md) — at N=10 the cyclic
client_id assignment naturally gives 5 harmless + 5 helpful, so the
GPT-API filter found valid samples in every eval).

## Tier-1 winrate trajectory (single seed)

| TID | Config | rd 9 (H/HH) | rd 19 | rd 29 | rd 39 | **rd 49** | Σ@49 |
|---|---|---|---|---|---|---|---|
| 91200 | baseline (β=0.01, λ=1.0, rc=0.1) | 20.0/73.3 | 33.3/76.7 | 23.3/80.0 | 30.0/80.0 | 76.7/60.0 | 136.7 |
| 91201 | kl_low (β=0.001) | 23.3/73.3 | 16.7/73.3 | 26.7/70.0 | 20.0/70.0 | 83.3/56.7 | 140.0 |
| 91202 | kl_high (β=0.1) | 33.3/66.7 | 20.0/63.3 | 26.7/60.0 | 30.0/56.7 | 83.3/66.7 | 150.0 |
| 91203 | ortho_high (λ=5.0) | 26.7/80.0 | 26.7/73.3 | 23.3/70.0 | 40.0/76.7 | 83.3/63.3 | 146.7 |
| **91204** | **reward_high (rc=0.5)** | 30.0/73.3 | 33.3/80.0 | 23.3/73.3 | 30.0/76.7 | **80.0/76.7** | **156.7** ⭐ |

rd 9-39 use a 30-sample GPT eval (±15pp std error). rd 49 uses the
larger per-client-test-data path — substantially more reliable single
snapshot. Pareto-best per config is always rd 49 here.

## Why the tier-1 result is unreliable

Looking at the actual RL training loss/acc trajectory:

| Config | rd 38 loss | rd 49 loss | Δ (10 rds) | rd 49 train_acc | Converged? |
|---|---|---|---|---|---|
| baseline | 374 | 273 | **−37%** | 0.96 | **No** — loss still falling fast |
| kl_low | 403 | 332 | **−21%** | 0.87 | **No** — slowest learner |
| kl_high | 381 | 279 | **−37%** | 0.96 | **No** |
| ortho_high | 396 | 299 | **−33%** | 0.93 | **No** |
| reward_high | 140 | **65** | **−54%** | **0.99** | **Yes** (possibly overfit) |

Three independent problems:

1. **4/5 configs are underfit at rd 49.** Losses are dropping 21-37%
   in the last 10 rounds. The rd 49 comparison is between
   partially-trained models, not their converged performance.

2. **reward_high (rc=0.5) is converging 5× faster than the others.**
   `reward_coeff` literally scales the DPO gradient signal, so a
   higher rc converges in fewer rounds. The "win" might just be
   faster convergence under a fixed round budget — not a better
   final solution.

3. **reward_high may already be overfit.** train_acc=0.99 vs
   test_winrate=80 is a 19pp gap. Worth checking with held-out eval.

## Tier-2 plan (8-GPU launch, 2026-05-19)

All 8 reuse the existing selector
`final_hhrl_choice_qwen2_hpsearch_n10_reward_high_t90204.ckpt`. Only
the RL stage runs (no new selector training needed).

LR raised to **5e-5** (5× the current 1e-5) for runs that test the
"convergence speed vs final quality" question. 1e-4 was considered but
left as a follow-up — too aggressive for DPO without more guardrails.

### Group A — Multi-seed the leader (3 GPUs)

Pins down the ±15pp single-seed noise on the tier-1 winner.

| TID | GPU | Config | Why |
|---|---|---|---|
| 92204 | 0 | reward_high, LR=5e-5, seed=1 | seed |
| 92205 | 1 | reward_high, LR=5e-5, seed=2 | seed |
| 92206 | 2 | reward_high, LR=5e-5, seed=3 | seed |

### Group B — Fairness checks + sweep extensions (5 GPUs)

| TID | GPU | Config | Tests |
|---|---|---|---|
| 93000 | 3 | baseline (rc=0.1), LR=5e-5, 50 rounds | **Fairness**: if baseline at higher LR matches reward_high, the tier-1 win was just convergence speed |
| 93001 | 4 | reward_high, rc=**1.0**, LR=5e-5 | **Push the winning knob further** |
| 93002 | 5 | reward_high, LR=5e-5, **100 rounds** | **Does the winner keep improving with more time?** |
| 93003 | 6 | baseline (rc=0.1), LR=1e-5, **100 rounds** | **Does baseline catch up if given convergence time?** |
| 93004 | 7 | reward_high, rc=**0.3**, LR=5e-5 | **Middle of the rc range** (sweep is 0.1 / 0.3 / 0.5 / 1.0) |

Launch script: `scripts/server/hp_search/run_n10_rl_tier2.sh`.

### Wall-clock

- 50-round runs: ~12-15h each
- 100-round runs (93002, 93003): ~24h each
- All in parallel on 8 GPUs → ~24h end-to-end

### Decision rule

Same as tier 1: max(H+HH) on the Pareto-best round, tiebreak by
balance (closer to 45° line). With multi-seed available for
reward_high, the Group A average is the strongest reference point.

## After tier-2

1. Pick the winning config from tier-1 + tier-2.
2. If the winner is consistent across seeds AND beats baseline at
   matched LR, declare it the N=10 answer.
3. Multi-seed (≥3) the chosen baseline-comparison config for the
   Main Table sweep.
4. If tier-2 reveals reward_high was just a convergence-speed
   artifact, the real winner is whichever of {baseline, kl_high,
   ortho_high} comes out on top at LR=5e-5 / 100 rounds.

## Files written by this session

- `documents/HP_SEARCH_RESULTS_N10.md` (this file)
- `scripts/server/hp_search/run_n10_rl_tier2.sh` (tier-2 launcher)
- `outputs/{92204..92206,93000..93004}.log` (training logs)
- `/hdd/hdd3/kjb/checkpoints/50_hhrl_rlhf_qwen2_hpsearch_n10_tier2_*.ckpt`
