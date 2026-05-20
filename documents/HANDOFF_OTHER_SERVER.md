# Handoff — other-server N=10 Main Table (2026-05-20)

This doc is the runbook for the other server (where the N=10 HP search
ran) to launch the N=10 Main Table sweep. The N=100 sweep runs on
this server in parallel; the two will converge into a single results
table.

## Required reading (in order)

1. **[MAIN_TABLE_SETUP.md](./MAIN_TABLE_SETUP.md)** — the canonical
   hyperparameter recipe for the Main Table, derived from both HP
   searches. Explains every setting.
2. **[HP_SEARCH_RESULTS_N10.md](./HP_SEARCH_RESULTS_N10.md)** — the
   tier-2 result that yielded the recipe.
3. **[HP_SEARCH_RESULTS_N100.md](./HP_SEARCH_RESULTS_N100.md)** —
   N=100 tier-1 (single-seed) results.

After you `git pull`, the four flagship RL cfgs at
`cfg/main_table/qwen_hhrlhf/hrl_comparison_{fedbiscuit_11200,fedvpl_11201,kl_only_11202,kl_ortho_11203}.yaml`
already contain the new recipe (commit `16b76b2`). You don't need to
edit anything — just launch.

## What's in the new recipe (already in the cfgs)

| Setting | Value | Why |
|---|---|---|
| `federate.total_round_num` | 70 | Past the typical peak window, before the DPO collapse zone |
| `federate.save_freq` | 10 | Pareto-best round is never the final round; save every 10 |
| `eval.max_samples_for_reward` | 100 | ±9.8pp CI vs ±18pp at 30 samples |
| `train.optimizer.lr` | 5e-5 | Tier-2 fairness check: 5e-5 hits a higher peak than 1e-5 |
| `llm.reward_coeff` | 0.5 (FedVPL/FedVPA-GP), 0.1 (FedBiscuit) | Tier-2 winner; FedBiscuit doesn't use z so rc is irrelevant |

## N=10 Main Table launch

The cfgs default to `client_num: 10`, so for N=10 you don't need to
override that. Standard launch pattern:

### Stage-1 selectors (4 in parallel on GPUs 0-3)

```bash
cd /home/.../ppfl   # your project root
conda activate biscuit
export PYTHONPATH=$(pwd)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

CKPT_DIR=/hdd/hdd3/kjb/checkpoints   # adjust per your server

# FedBiscuit selector
nohup python -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_comparison_fedbiscuit_10200.yaml \
    device 0 \
    federate.save_to $CKPT_DIR/hhrl_choice_qwen2_mt_n10_fedbiscuit_t10200.ckpt \
    expname mt_n10_fedbiscuit_sel_10200 \
    > outputs/mt_n10_10200.log 2>&1 &

# FedVPL selector
nohup python -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_comparison_fedvpl_10201.yaml \
    device 1 \
    federate.save_to $CKPT_DIR/hhrl_choice_qwen2_mt_n10_fedvpl_t10201.ckpt \
    expname mt_n10_fedvpl_sel_10201 \
    > outputs/mt_n10_10201.log 2>&1 &

# FedVPA-GP (kl_only) selector
nohup python -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_comparison_kl_only_10202.yaml \
    device 2 \
    federate.save_to $CKPT_DIR/hhrl_choice_qwen2_mt_n10_kl_only_t10202.ckpt \
    expname mt_n10_kl_only_sel_10202 \
    > outputs/mt_n10_10202.log 2>&1 &

# FedVPA-GP full (kl_ortho) selector
nohup python -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_comparison_kl_ortho_10203.yaml \
    device 3 \
    federate.save_to $CKPT_DIR/hhrl_choice_qwen2_mt_n10_kl_ortho_t10203.ckpt \
    expname mt_n10_kl_ortho_sel_10203 \
    > outputs/mt_n10_10203.log 2>&1 &
```

**Wall-clock:** ~3-5h each on a single GPU (selector training at N=10).

### Stage-2 RL (after selectors finish, 4 in parallel on GPUs 0-3)

After all 4 selectors save their `final_*.ckpt`, launch RL. Important:
**Stage-2 uses a different entrypoint** — `federatedscope/llm/rlhf/main.py`
— and **requires `--selector-cfg-file`**.

```bash
CFG_DIR=cfg/main_table/qwen_hhrlhf

nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg $CFG_DIR/hrl_comparison_fedbiscuit_11200.yaml \
    --selector-cfg-file $CFG_DIR/fedvpagp_comparison_fedbiscuit_10200.yaml \
    device 0 \
    llm.rlhf_selector_checkpoint $CKPT_DIR/final_hhrl_choice_qwen2_mt_n10_fedbiscuit_t10200.ckpt \
    federate.save_to $CKPT_DIR/hhrl_rlhf_qwen2_mt_n10_fedbiscuit_t11200.ckpt \
    expname mt_n10_fedbiscuit_rl_11200 \
    > outputs/mt_n10_11200.log 2>&1 &

nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg $CFG_DIR/hrl_comparison_fedvpl_11201.yaml \
    --selector-cfg-file $CFG_DIR/fedvpagp_comparison_fedvpl_10201.yaml \
    device 1 \
    llm.rlhf_selector_checkpoint $CKPT_DIR/final_hhrl_choice_qwen2_mt_n10_fedvpl_t10201.ckpt \
    federate.save_to $CKPT_DIR/hhrl_rlhf_qwen2_mt_n10_fedvpl_t11201.ckpt \
    expname mt_n10_fedvpl_rl_11201 \
    > outputs/mt_n10_11201.log 2>&1 &

nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg $CFG_DIR/hrl_comparison_kl_only_11202.yaml \
    --selector-cfg-file $CFG_DIR/fedvpagp_comparison_kl_only_10202.yaml \
    device 2 \
    llm.rlhf_selector_checkpoint $CKPT_DIR/final_hhrl_choice_qwen2_mt_n10_kl_only_t10202.ckpt \
    federate.save_to $CKPT_DIR/hhrl_rlhf_qwen2_mt_n10_kl_only_t11202.ckpt \
    expname mt_n10_kl_only_rl_11202 \
    > outputs/mt_n10_11202.log 2>&1 &

nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg $CFG_DIR/hrl_comparison_kl_ortho_11203.yaml \
    --selector-cfg-file $CFG_DIR/fedvpagp_comparison_kl_ortho_10203.yaml \
    device 3 \
    llm.rlhf_selector_checkpoint $CKPT_DIR/final_hhrl_choice_qwen2_mt_n10_kl_ortho_t10203.ckpt \
    federate.save_to $CKPT_DIR/hhrl_rlhf_qwen2_mt_n10_kl_ortho_t11203.ckpt \
    expname mt_n10_kl_ortho_rl_11203 \
    > outputs/mt_n10_11203.log 2>&1 &
```

**Wall-clock:** ~7-10h each on a single GPU (RL at N=10 with 70 rounds
+ 100-sample eval every 10 rounds).

## How to score the runs

The eval log lines look like:

```
INFO: Evaluated 100 samples for helpfulness winrate using GPT API (gpt-4o-mini): X.XX%
INFO: Evaluated 100 samples for harmlessness winrate using GPT API (gpt-4o-mini): Y.YY%
```

For each run, score every round in {19, 29, 39, 49, 59, 69} and pick
the **Pareto-best** (max H+HH, tiebreak = closest to 45° line). That
is the row in the Main Table. **Do NOT use rd 69 by default** — it is
typically past the peak (Pareto-best rounds in our HP search ranged
9–59).

The Pareto-best ckpt file is `{round}_{save_to_filename}` (e.g.,
`30_hhrl_rlhf_qwen2_mt_n10_kl_ortho_t11203.ckpt`).

## Cross-server coordination

This server (where you are reading this from, if you're the
N=100-side Claude) launched the matching N=100 sweep on
2026-05-20 ~13:30 KST. The 4 selectors are in flight on GPUs 1–4
(PIDs 3052492-3052495); RL will launch on the same GPUs after
selectors finish.

Output naming convention is symmetric:
- This server: `hhrl_choice_qwen2_mt_n100_*_t1020X.ckpt`,
  `hhrl_rlhf_qwen2_mt_n100_*_t1120X.ckpt`
- Other server: `hhrl_choice_qwen2_mt_n10_*_t1020X.ckpt`,
  `hhrl_rlhf_qwen2_mt_n10_*_t1120X.ckpt`

Together you get 8 result cells (2 N × 4 methods). Combined into a
single `documents/MAIN_TABLE_RESULTS.md` after both servers finish.

## After both servers finish the single-seed Main Table

1. If results look clean, **multi-seed (≥3) the FedVPA-GP-full config**
   at the chosen N value(s) — that's the headline number.
2. **Optionally bump eval samples to 300** for the final headline
   numbers (~$5/eval vs $1.50 — negligible cost increase).
3. **Then Gemma-2B**. Apply the same edits to a Gemma cfg (none exist
   yet, needs creation), probe at N=10 first.

## What I (the launching Claude) am NOT doing

- Touching the **Stage-1 selector cfgs**. The HP search tuned Stage-2
  RL only. Stage-1 keeps its current defaults (β=0.01, λ=1.0, etc.).
- **Gemma-2B cfgs**. None exist; create them when the Qwen Main Table
  is settled.
- **FedDPO**. The original Main Table promised 4 methods including
  FedDPO, but FedDPO has been omitted in the comparison cfg set
  (10200-10203 are FedBiscuit / FedVPL / kl_only / kl_ortho). Add it
  later if the paper draft still claims FedDPO as a baseline; for now
  the 4 cfgs we have are FedBiscuit / FedVPL / FedVPA-GP-kl-only /
  FedVPA-GP-full.
