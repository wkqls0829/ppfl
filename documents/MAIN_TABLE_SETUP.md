# Main Table setup — derived from HP search (2026-05-20)

This doc explains the hyperparameters and eval settings for the
paper's Main Table sweep, with every choice grounded in the N=10 and
N=100 HP search results. It supersedes the original Table-3
hyperparameters in `CLAUDE.md` for Stage-2 RL.

## Settings applied to all 4 Main Table RL cfgs

The four flagship RL cfgs at `cfg/main_table/qwen_hhrlhf/` have been
edited in place:

- `hrl_comparison_fedbiscuit_11200.yaml` — FedBiscuit baseline
- `hrl_comparison_fedvpl_11201.yaml` — FedVPL baseline (naive VPL)
- `hrl_comparison_kl_only_11202.yaml` — FedVPA-GP without orthogonal loss
- `hrl_comparison_kl_ortho_11203.yaml` — FedVPA-GP full

All four now share this Stage-2 RL recipe:

| Setting | Old | New | Why |
|---|---|---|---|
| `federate.total_round_num` | 50 | **70** | Pareto-best rounds in HP search ranged 9–59; 70 captures the peak window for slow-converging configs without entering the universal DPO mode-collapse zone (which hits at rd 49 for fast configs / rd 99 for slow ones). |
| `federate.save_freq` | 50 | **10** | At save_freq=50 we'd save only rd 50/100 — the worst checkpoint. Tier-2 confirmed Pareto-best rounds are never 50 or 99. Saving every 10 rounds catches the peak. |
| `eval.max_samples_for_reward` | 30 (default) | **100** | 30-sample win-rate has ±18pp 95% CI, larger than most config-vs-config differences. 100 samples drops CI to ±9.8pp — adequate for HP-level signal. Paper-quality (Main Table) ideally wants 300+ once we multi-seed. |
| `train.optimizer.lr` | 1e-5 | **5e-5** | Tier-2 fairness check (N=10 93000 vs 93003) showed LR=5e-5 + mid-training stop > LR=1e-5 + mid-training stop on the same config. Both still collapse eventually; 5e-5 just hits a higher peak before collapse. |
| `llm.reward_coeff` | 0.1 | **0.5** | Tier-2 winner (N=10 93002 @ rd 59: 50/73, Σ 123.3) used rc=0.5. Reduced gradient-scale confound by pairing with LR=5e-5. FedBiscuit cfg kept at rc=0.1 since it has no z-conditioning so rc is irrelevant. |

## What the HP search did *not* re-confirm (still open)

- **Stage-1 selector hyperparameters.** The HP search tuned Stage-2 RL
  only. Stage-1 keeps the existing Table-3 defaults: β=0.01, λ=1.0,
  γ=0.1, τ=1.0, prototype_scale=5.0, deep_projection=True,
  logit_dropout=0.5 (Stage-1 specific), max_logvar=−4.0,
  manual_orthogonal_labels=True. Stage-1 uses a classification loss
  and does not suffer from the DPO mode-collapse we saw in Stage-2.
- **The N=100 collapse rate.** All N=100 tier-1 numbers came from the
  30-sample eval path. Our N=10 tier-2 used 100+ samples via per-client
  test data. We have not yet replicated the N=100 numbers under the
  same eval recipe, so the N=100 collapse curve is extrapolated, not
  measured. The first Main Table N=100 run is also our cleanest data
  point for that.
- **Gemma-2B numbers entirely.** All HP search was on Qwen-2 0.5B.
  Gemma-2B may need a different LR/rc — both are larger models with
  different DPO dynamics. Recommend a 1-config probe at Gemma + N=10
  before launching the full Gemma Main Table.

## Evaluation policy for the Main Table

1. **Score every saved checkpoint** (rds 10, 20, 30, 40, 50, 60, 70).
2. **Report the Pareto-best round per config** (max sum(H, HH); tiebreak
   = closest to the H=HH 45° line). **Do not report rd 70 by default —
   the Pareto-best is typically rd 19–59.**
3. **Eval at 100 samples per metric** for HP-level decisions. Bump to
   **300 samples** for the final paper Main Table, ideally averaged
   across **3 seeds**. (Cost estimate: ~$5/eval × 3 seeds × 4 configs
   × 5 eval rounds × 3 N values × 2 models ≈ $1800 — but the experiment
   compute already dwarfs this.)
4. **Always GPT-4o-mini for win-rate scoring**. The "internal model
   fallback" (`_get_winrate_scores_with_internal_model`) is biased and
   was the source of the inflated tier-1 rankings — never trust its
   output for paper claims.

## Main Table sweep matrix

| Model | Method | N values | Cfg |
|---|---|---|---|
| Qwen-2 0.5B | FedDPO | 10, 50, 100 | TBD (no cfg yet — needs creation) |
| Qwen-2 0.5B | FedBiscuit | 10, 50, 100 | `hrl_comparison_fedbiscuit_11200.yaml` + `federate.client_num` override |
| Qwen-2 0.5B | FedVPL | 10, 50, 100 | `hrl_comparison_fedvpl_11201.yaml` + override |
| Qwen-2 0.5B | FedVPA-GP | 10, 50, 100 | `hrl_comparison_kl_ortho_11203.yaml` + override |
| Gemma-2B | (all 4) | 10, 50, 100 | needs new cfgs (model.type + LR) |

24 cells × 3 seeds = 72 RL runs at minimum. At ~7h per run × 5–7 GPUs
in parallel: ~14 days end-to-end on one server. SLURM cluster is much
faster — see `scripts/slurm/main_table/`.

## Compute cost estimate (Qwen-2 0.5B only, 3 seeds)

- 4 methods × 3 N × 3 seeds = 36 RL runs
- Per run: ~7h GPU + ~$15 GPT eval (100-sample, 7 eval rounds)
- Total: ~250 GPU-h + ~$540 GPT
- Bumping to 300 samples: ~$1620 GPT
- Both totals fit comfortably in a research budget.

## Recommended next steps before launching the Main Table

1. **Single-cell N=100 probe** with the new cfg (`hrl_comparison_kl_ortho_11203.yaml`
   + `federate.client_num 100`) to verify the 70-round + save_freq=10
   pipeline at N=100 actually catches the Pareto-best round before
   collapse, and that the 100-sample eval is producing real numbers.
   Wall-clock ~7h.
2. **Multi-seed (3-seed) the winning config** to nail the per-config
   noise floor.
3. **Apply the same edits to Gemma-2B cfgs** once they exist, with a
   1-config probe on Gemma + N=10 first.
4. **Then launch the full Main Table sweep**.

## Open question: collapse mechanism

Tier-2 confirmed all configs eventually drive helpfulness to 0. Worth
investigating (for the discussion section of the paper or a separate
ablation):

- Is the train_loss / train_acc signal predictive of when collapse
  hits? If yes, an automatic early-stopping criterion would replace
  the "pick Pareto-best by hand" rule.
- Does the z prototype drift correlate with collapse? Visualize
  prototype angles + cluster purity around the collapse rounds.
- Does this happen because DPO loss prefers refusals as the model
  becomes confident the harmlessness reward dominates? A KL-to-base
  penalty (PPO-style) might prevent it.

These are nice-to-have; the immediate path to the Main Table doesn't
require resolving them.

## Diff summary (for the commit message / changelog)

```
cfg/main_table/qwen_hhrlhf/hrl_comparison_fedbiscuit_11200.yaml:
  + total_round_num: 50 -> 70
  + save_freq: 50 -> 10
  + lr: 1e-5 -> 5e-5
  + eval.max_samples_for_reward: 100 (new)
  (reward_coeff unchanged — irrelevant for FedBiscuit)

cfg/main_table/qwen_hhrlhf/hrl_comparison_fedvpl_11201.yaml:
cfg/main_table/qwen_hhrlhf/hrl_comparison_kl_only_11202.yaml:
cfg/main_table/qwen_hhrlhf/hrl_comparison_kl_ortho_11203.yaml:
  same as above + reward_coeff: 0.1 -> 0.5
```
