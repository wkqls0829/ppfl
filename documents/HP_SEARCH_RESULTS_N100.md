# N=100 HP search — tier-1 results & tier-2 plan (2026-05-19)

This server ran the N=100 HP search per
[HP_SEARCH_PLAN.md](./HP_SEARCH_PLAN.md). 5 selectors finished cleanly,
and 4 of 5 Stage-2 RL runs reached round 49 with honest GPT-API
win-rates at every eval round (the 2026-05-19 fix —
[BUGFIX_2026_05_19.md](./BUGFIX_2026_05_19.md) — was in place).
**kl_low (91101) was killed at round ~25 because another tenant needed
the GPU**; its last datapoint is rd 19.

## Tier-1 winrate trajectory (single seed)

All numbers are GPT-API (gpt-4o-mini), 30-sample evals, ±15pp std error.

| TID | Config | rd 9 (H/HH) | rd 19 | rd 29 | rd 39 | **rd 49** | **Pareto-best Σ (round)** |
|---|---|---|---|---|---|---|---|
| **91100** | **baseline (β=0.01, λ=1.0, rc=0.1)** | 33.3/70.0 | 40.0/66.7 | 40.0/60.0 | 20.0/73.3 | **46.7/73.3** | **120.0 (rd 49)** ⭐ |
| 91102 | kl_high (β=0.1) | 33.3/86.7 | 20.0/86.7 | 23.3/73.3 | 26.7/73.3 | 23.3/63.3 | 120.0 (rd 9) |
| 91103 | ortho_high (λ=5.0) | 66.7/60.0 | 46.7/70.0 | 30.0/86.7 | 26.7/86.7 | 23.3/63.3 | **126.7 (rd 9)** |
| 91104 | reward_high (rc=0.5) | 33.3/76.7 | 26.7/73.3 | 33.3/66.7 | 13.3/93.3 | 23.3/80.0 | 110.0 (rd 9) |
| ~~91101~~ | ~~kl_low (β=0.001)~~ | 20.0/73.3 | 26.7/80.0 | killed | — | — | (106.7 at rd 19) |

## Two conflicting "winners"

| Selection rule | Winner | Σ |
|---|---|---|
| rd 49 (the only ckpt we'd actually persist) | **baseline** | **120.0** |
| Pareto-best across any round | **ortho_high (rd 9)** | **126.7** |

Either is plausible but neither is decisive given the 30-sample noise.

## Why this comparison is unreliable (same hazards as N=10)

1. **Most configs trend downward across rds 9→39.** ortho_high, kl_high,
   and reward_high all peak at rd 9 and decline through rd 39 — they're
   **overshooting** the helpful target as DPO trains harder, then snap
   back at rd 49. Baseline is the only config that **climbs late**
   (peak at rd 49). This is the opposite of N=10's underfit pattern,
   but the root cause is the same — the 50-round budget doesn't catch
   each config at its real peak.
2. **reward_high (rc=0.5) is the only config that visibly converges
   harmlessness fast**, reaching 93.3 at rd 39. Same convergence-speed
   confound as N=10: rc=0.5 = 5× DPO gradient, so it learns faster.
   We can't tell whether its "real" performance is at rd 49 (103.3) or
   somewhere in between.
3. **Single seed.** 30-sample evals have ±15pp standard error per
   metric; the gap between the top 3 configs (120 vs 113 vs 110) is
   within noise.

## Tier-2 plan (8-GPU launch)

Same structure as the other server's N=10 tier-2 — script already
written: `scripts/server/hp_search/run_n100_rl_tier2.sh`. All 8 reuse
the existing `reward_high` tier-1 selector
(`final_hhrl_choice_qwen2_hpsearch_n100_reward_high_t90104.ckpt`); only
the RL stage runs.

### Group A — Multi-seed the apparent leader (3 GPUs)

Pins down ±15pp noise on `reward_high` (the convergence-speed-faster
candidate, more interesting than baseline despite baseline's rd 49 win).

| TID | GPU | Config |
|---|---|---|
| 92104 | 0 | reward_high, LR=5e-5, seed=1 |
| 92105 | 1 | reward_high, LR=5e-5, seed=2 |
| 92106 | 2 | reward_high, LR=5e-5, seed=3 |

### Group B — Fairness checks + sweep extensions (5 GPUs)

| TID | GPU | Config | Tests |
|---|---|---|---|
| 93100 | 3 | baseline, LR=5e-5 | **Fairness**: does baseline at higher LR match reward_high at higher LR? If yes, reward_high's win is just gradient scale. |
| 93101 | 4 | rc=1.0, LR=5e-5 | Push the winning knob further. |
| 93102 | 5 | reward_high, LR=5e-5, **100 rounds** | Does the winner keep improving? Or did it peak by rd 49? |
| 93103 | 6 | baseline (rc=0.1, LR=1e-5), **100 rounds** | Does baseline catch up if given more time? Most directly tests the convergence-speed hypothesis at the existing baseline LR. |
| 93104 | 7 | rc=0.3, LR=5e-5 | Middle of rc sweep (0.1 / 0.3 / 0.5 / 1.0). |

### GPU mapping note

When another tenant is using GPU 0 or 1, launch with
`HP_TIER2_GPUS="2,3,4,5,6,..."` so the first N runs get the available
GPUs. With only 5 GPUs available the script launches all 8 — the
remaining 3 will fail fast to free up. Or trim the script to a 5-run
subset (recommended: Group A's 92104+92105+92106 + 93100 + 93103 to
cover both fairness questions).

### Wall-clock (post-eval-fix)

- 50-round runs: ~5–7h
- 100-round runs (93102, 93103): ~10–14h
- All in parallel on 8 GPUs → ~14h end-to-end

### Decision rule

Same as tier 1: max(H+HH) on the Pareto-best round (or rd 49 if you
prefer "what we'd save"). With multi-seed available for reward_high,
the Group A average is the strongest reference point.

## After tier-2

1. If `baseline@LR=5e-5` matches `reward_high@LR=5e-5` (within
   ±15pp), the tier-1 "reward_high win" was just convergence speed.
   The real N=100 answer is whichever of {baseline, kl_high, ortho_high}
   converges best given a fair LR / round budget.
2. If `reward_high@LR=5e-5` cleanly beats `baseline@LR=5e-5`,
   `reward_coeff=0.5` is doing something more than gradient scaling and
   is the legit N=100 winner.
3. Multi-seed (≥3) the winning config before any paper claim, and only
   then launch the full Main Table sweep.

## Files written by this session

- `documents/HP_SEARCH_RESULTS_N100.md` (this file)
- `scripts/server/hp_search/run_n100_rl_tier2.sh` (tier-2 launcher)
- `outputs/{91100,91102,91103,91104}.log` (RL logs;
  91101 partial up through round ~25)
- `/hdd/hdd3/kjb/checkpoints/50_hhrl_rlhf_qwen2_hpsearch_n100_{baseline,kl_high,ortho_high,reward_high}_t9110{0,2,3,4}.ckpt`
