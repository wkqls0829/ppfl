# Z Separation Experiments (TID 10013-10016)

**Status**: Stopped — needs relaunch on SLURM or other server
**Date**: 2026-03-18
**GPUs needed**: 4 (one per experiment)

## Problem

t-SNE of latent z shows no meaningful cluster separation by preference type (helpful vs harmless). z is bypassable because:
1. `latent_projection` is a single Linear(32→2) — too weak
2. Frozen base logits already predict well without z
3. `max_logvar=-3.0` clamps posteriors too tight (σ≈0.22)
4. `vpl_kl_weight=0.1` is too weak to enforce structure

## Code Changes Made (on `refined` branch)

### 1. `federatedscope/llm/trainer/vpl_reward_choice_trainer.py`

**New config options** (read via `getattr`, no registration needed):
- `vpl_deep_projection` (bool, default False): Replace Linear(32→2) with MLP(32→64→32→2 with ReLU)
- `vpl_logit_dropout` (float, default 0.0): During training, randomly zero out base logits per-sample so z must carry the signal

**Changes**:
- Lines ~48-56: Added `self.vpl_deep_projection` and `self.vpl_logit_dropout` init
- Lines ~263-278: Conditional deep projection MLP creation
- Lines ~700-711: Base logit dropout using `ctx.cur_mode` check (not `self.training` — trainer is not nn.Module)

### 2. `federatedscope/llm/trainer/vpl_gp_reward_choice_trainer.py`

**Bug fix**: The GP trainer was re-creating `VariationalEncoderGP` WITHOUT passing `max_logvar` or tau anneal params, overwriting the correctly-configured encoder from the parent `__init__`. Now it reuses the parent's encoder.

### 3. `federatedscope/llm/llm_local/z_visualization.py`

**Bug fix**: Added `matplotlib.use('Agg')` before `import matplotlib.pyplot`. Without this, t-SNE visualization crashes the process when X server disconnects (the cause of the Round 10 crash).

## Experiment Configs Created

All in `cfg/main_table/qwen_hhrlhf/`:

| TID | Config file | What changed vs 10003 (baseline) |
|-----|-------------|----------------------------------|
| 10013 | `fedvpagp_relaxed_kl_10013.yaml` | `max_logvar: 0.0`, `kl_weight: 0.5` |
| 10014 | `fedvpagp_deep_proj_dropout_10014.yaml` | `deep_projection: True`, `logit_dropout: 0.5` |
| 10015 | `fedvpagp_combined_10015.yaml` | All above combined (`kl_weight: 0.5`) |
| 10016 | `fedvpagp_combined_maxkl_10016.yaml` | All above + `kl_weight: 1.0` |

Everything else identical to 10003 (Qwen2-0.5B, HH-RLHF, 10 clients, 50 rounds, LoRA r=8).

## Launch Script

`scripts/server/main_table/run_qwen_hhrlhf_zsep_10013_10016.sh`

Uses GPUs 3,4,5,6. To run on different GPUs, edit `CUDA_VISIBLE_DEVICES` in the script.

## To Relaunch

```bash
conda activate biscuit

# Option A: Use the script (GPUs 3-6)
bash scripts/server/main_table/run_qwen_hhrlhf_zsep_10013_10016.sh

# Option B: Run individually (adjust GPU as needed)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/path/to/ppfl:$PYTHONPATH

CUDA_VISIBLE_DEVICES=3 nohup python -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_relaxed_kl_10013.yaml \
    > outputs/10013.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 nohup python -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_deep_proj_dropout_10014.yaml \
    > outputs/10014.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup python -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_combined_10015.yaml \
    > outputs/10015.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup python -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_combined_maxkl_10016.yaml \
    > outputs/10016.log 2>&1 &
```

## What to Check After Training

1. t-SNE plots in `exp/test_fedvpagp_*_t1001{3,4,5,6}/cross_client_z_tsne_round_*.png`
2. WandB project `fvpl-selector` — compare `vpl_kl_loss`, `vpl_reconstruction_loss`, `acc`
3. Key indicators of z separation working:
   - KL loss should be moderate (not near 0 = collapse, not huge = instability)
   - Reconstruction loss should still decrease (z is helping, not hurting)
   - t-SNE should show 2 distinct clusters aligned with helpful/harmless clients

## Previous Experiments (Completed, all on this server)

| TID | Variant | Status |
|-----|---------|--------|
| 10000 | FedBiscuit | Done (Mar 11) |
| 10001 | FedVPL | Done (Mar 12) |
| 10002 | VPL-GP no ortho | Done (Mar 12) |
| 10003 | FedVPA-GP full (baseline) | Done (Mar 12) |
| 10004 | FedVPA-GP proto10 | Done (Mar 12) |
| 10005 | FedVPA-GP no_diff | Done (Mar 12) |
| 10006 | FedVPA-GP tight_prior | Done (Mar 13) |
| 10007 | FedVPA-GP manual_labels | Done (Mar 13) |
| 10008 | FedVPA-GP tight_manual | Done (Mar 13) |
| 10009 | FedVPA-GP tighter_prior | Done (Mar 13) |
| 10010 | FedVPA-GP siamese | Done (Mar 13) |
| 10011 | FedVPA-GP highlr | Done (Mar 13) |
| 10012 | FedVPA-GP highlr_smallproto | Done (Mar 13) |
| 10013-10016 | Z-separation fixes | NOT STARTED (crashed at R10 due to matplotlib X server bug, now fixed) |
