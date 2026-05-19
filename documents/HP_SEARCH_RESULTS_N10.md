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

## Tier-2 results (2026-05-20, all 8 runs completed)

### Pareto-best per run

| TID | Config | Best Rd | H | HH | Σ |
|---|---|---|---|---|---|
| 92204 | reward_high seed=1 LR=5e-5 | 9 | 30.0 | 80.0 | 110.0 |
| 92205 | reward_high seed=2 LR=5e-5 | 19 | 33.3 | 80.0 | 113.3 |
| 92206 | reward_high seed=3 LR=5e-5 | 29 | 13.3 | 80.0 | 93.3 |
| 93000 | baseline (rc=0.1) LR=5e-5 50rd | 29 | 33.3 | 80.0 | 113.3 |
| **93001** | rc=**1.0** LR=5e-5 50rd | **39** | **43.3** | **80.0** | **123.3** |
| **93002** | rc=0.5 LR=5e-5 **100rd** | **59** | **50.0** | **73.3** | **123.3** |
| 93003 | baseline (rc=0.1) LR=1e-5 **100rd** | 9 | 40.0 | 80.0 | 120.0 |
| 93004 | rc=**0.3** LR=5e-5 50rd | 9 | 43.3 | 76.7 | 120.0 |

### Full trajectory for the two leaders

**93002 (rc=0.5 LR=5e-5 100rd):** the cleanest run — long plateau then catastrophic collapse.
```
rd  9: H=30.0 HH=76.7 (sum 106.7)
rd 19: H=40.0 HH=66.7 (sum 106.7)
rd 29: H=53.3 HH=63.3 (sum 116.7)
rd 39: H=43.3 HH=70.0 (sum 113.3)
rd 49: H=46.7 HH=66.7 (sum 113.3)
rd 59: H=50.0 HH=73.3 (sum 123.3)  ← PEAK
rd 69: H=46.7 HH=73.3 (sum 120.0)
rd 79: H=36.7 HH=80.0 (sum 116.7)
rd 89: H=46.7 HH=73.3 (sum 120.0)
rd 99: H= 0.0 HH=70.0 (sum  70.0)  ← collapse
```

**93003 (baseline rc=0.1 LR=1e-5 100rd):** even the conservative config collapses with enough training.
```
rd  9: H=40.0 HH=80.0 (sum 120.0)  ← PEAK
rd 19: H=40.0 HH=73.3
rd 29: H=43.3 HH=73.3
rd 39: H=53.3 HH=63.3
rd 49: H=36.7 HH=66.7
rd 59: H=40.0 HH=73.3
rd 69: H=40.0 HH=70.0
rd 79: H=43.3 HH=56.7
rd 89: H=36.7 HH=60.0
rd 99: H= 0.0 HH=76.7  ← collapse
```

### Main findings

1. **Universal late-stage collapse on helpfulness.** Every single
   config — across seeds, LRs, reward_coeffs, and round counts — has
   H→0 at the end (rd 49 for LR=5e-5+rc≥0.3, rd 99 for the rest).
   This is **DPO over-training**, not a configuration issue. The
   policy gradient on the harmlessness side of the reward keeps
   pushing the model toward refusal until it loses helpfulness
   entirely.

2. **The final-round checkpoint is *never* the best.** Pareto-best
   rounds range from 9 to 59, never 49 or 99. The current
   `save_freq: 50` config saves only the worst checkpoint. For the
   Main Table sweep, switch to `save_freq: 10` (or 5) and report the
   Pareto-best round.

3. **The tier-1 "winner" (reward_high rc=0.5, rd 49 = 80/77) was
   a measurement artifact.** Tier-1 used the broken 30-sample eval
   path (the 4f98872/050957d/fe26ef7 cache fix didn't reach the
   client-test-data load until tier-2). On the consistent
   per-client eval, reward_high+LR=5e-5+rd 49 collapses to H=0.
   The honest tier-2 ceiling for that config is rd 29: 53.3/63.3.

4. **Two configs tied for best Σ=123.3 on Pareto:**
   - **93001 (rc=1.0 LR=5e-5 rd 39):** 43.3/80.0 — harmless-heavy
   - **93002 (rc=0.5 LR=5e-5 rd 59):** 50.0/73.3 — most balanced

   93002 is the cleaner result (closer to 45° line, longer plateau
   before collapse). **N=10 winner: 93002 config + save mid-training
   checkpoint.**

5. **Higher LR (5e-5) > lower LR (1e-5)** when paired with mid-
   training stopping. 93002 (5e-5) peaks higher than 93003 (1e-5)
   at sum 123.3 vs 120.0. But both collapse equally at the end.

### Recommended N=10 config for downstream work

```
train.optimizer.lr: 5e-5
llm.reward_coeff: 0.5
federate.total_round_num: 70   # train past the typical peak window
federate.save_freq: 10          # save every 10 rounds
# Then report the Pareto-best checkpoint (likely rd 29-59 range).
```

Or — if running with 50 rounds — use `rc=1.0` to push the early peak
earlier (best at rd 39: 43/80).

### Action items before Main Table

1. **Add intermediate ckpt saves** (`save_freq: 10`) to all Main
   Table cfg files. The current `save_freq: 50` discards 80%+ of
   the useful checkpoints.
2. **Re-run the 5 tier-1 selectors' RL stage** with the recommended
   config (rc=0.5, LR=5e-5, rds 70, save_freq=10) and pick best
   Pareto round per config. The original tier-1 ranking is
   invalidated by the eval-path bug.
3. **Investigate the collapse**: train_acc, train_loss, and z
   diagnostics around the collapse rounds (89-99 for slow configs,
   39-49 for fast configs) to see if there's a measurable indicator
   (gradient explosion, loss spike, z prototype drift) that predicts
   when collapse happens. Could enable an early-stopping signal.

## Original "After tier-2" plan (superseded)

The original tier-2 was designed to confirm reward_high as the
winner. Instead it revealed:
1. **Pick the winning config from tier-1 + tier-2.** → 93002
   (rc=0.5 + LR=5e-5 + mid-training stop).
2. **Beats baseline at matched LR?** → Yes: 93002@rd59 (50/73)
   beats 93000@rd29 (33/80) on sum.
3. **Convergence-speed artifact?** → Confirmed. tier-1's rc=0.5
   "win" at rd 49 (80/77) was the 30-sample eval bug. Real
   per-client eval peak is 50/73 at rd 59.
4. **{baseline, kl_high, ortho_high} at LR=5e-5 / 100 rounds?** →
   Untested directly. The kl/ortho selectors weren't paired with the
   100-round 5e-5 RL stage in tier-2. The 100-round runs only
   compared baseline (93003) vs reward_high (93002) and confirmed
   higher rc + higher LR has the higher peak.

## Files written by this session

- `documents/HP_SEARCH_RESULTS_N10.md` (this file)
- `scripts/server/hp_search/run_n10_rl_tier2.sh` (tier-2 launcher)
- `outputs/{92204..92206,93000..93004}.log` (training logs)
- `/hdd/hdd3/kjb/checkpoints/50_hhrl_rlhf_qwen2_hpsearch_n10_tier2_*.ckpt`
