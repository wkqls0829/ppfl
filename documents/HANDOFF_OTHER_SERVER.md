# Handoff to other-server Claude — 2026-05-17

Welcome. This doc is everything you need to pick up the FedVPA-GP work on
this server. Read this first, then `CLAUDE.md` (in repo root) for the
codebase guide.

## What you're being asked to do

Run the **N=10 HP search** described in `documents/HP_SEARCH_PLAN.md`.

The N=100 HP search will run on the other server (the one this doc came
from). When both finish, the winning hyperparameters become the basis
for the full Main Table sweep.

## Where the work stands

- Branch: `refined` on `origin/wkqls0829/ppfl` (HEAD = `346a562` —
  fix vpl federation, z-conditioning, eval fallback + add HP-search
  plan). `git pull` to get everything below.
- All session work (bug fixes + HP plan + this doc) is already pushed.

## Project in one paragraph

FedVPA-GP is a federated RLHF method submitted to ICML 2026. Stage 1
trains a federated **variational selector** that infers a per-client
latent preference vector `z`, regularized by a **Federated Mixture
Prior** over peer clients' posteriors + an **orthogonal loss** that
pulls `z` toward category prototypes. Stage 2 runs **z-conditional DPO**
on the server using that selector as the reward model. Evaluation: GPT-
4o-mini head-to-head win-rate vs the base model on HH-RLHF, scored
separately for helpfulness and harmlessness. The full algorithm and
notation live in `CLAUDE.md` and `Federated Variational Preference
Alignment.pdf`.

## What this session did

1. **Fixed 6 bugs** in the federated VPL training loop. The biggest
   one: the server was aggregating VPL components (variational
   encoder, latent projection, etc.) but never broadcasting them back
   to clients — so each client trained against a stale local copy.
   Other fixes: dtype mutation of `z_to_embedding`, posterior-stat
   aggregation via the law of total variance instead of empirical
   sample stats, z-collection no longer gated on visualization
   frequency, plus small cleanups.
2. **Discovered and fixed a 7th bug** in the eval path. Selector
   training with sparse client sampling (e.g. 10/100 per round) leaves
   60% of clients without saved z. The previous fallback in
   `winrate_metrics.py` ("use first available client") used a single
   client's z for every missing sample — typically a harmless
   client, which deflated helpfulness win-rates whenever the saved
   dict was sparse. The fix fills missing client IDs with the mean of
   same-category clients (`fill_missing_client_z_with_category_mean`
   in `federatedscope/llm/rlhf/load_vpl_components.py`). Wired into
   `winrate_metrics.py` and three load sites in
   `federatedscope/llm/rlhf/standalone_training.py`.
3. **Ran a single-cell sanity check** on Qwen-2 0.5B HH-RLHF
   (FedBiscuit vs FedVPA-GP at N=10 and N=100). 50 rounds each, full
   trajectories captured. Detailed results in
   `documents/RESUME_STATE.md`.
4. **Designed the HP search** in `documents/HP_SEARCH_PLAN.md` — see
   below.

## Headline results from the sanity check

Best-Pareto-checkpoint comparison (round selected per method by
max(H+HH) on the Pareto front, not round 50):

| | FedBiscuit (best round) | FedVPA-GP (best round) | Δ (H, HH) |
|---|---|---|---|
| **N=10** | rd 39: 46.7 / 90.0 (Σ 136.7) | rd 19: 33.33 / 86.67 (Σ 120.0) | **−13.4 / −3.3** |
| **N=100** | rd 19: 53.3 / 90.0 (Σ 143.3) | rd 9: 80.0 / 83.33 (Σ 163.3) | **+26.7 / −6.67** |

Read: **FedVPA-GP wins big at N=100 (+33% on helpful) but loses at
N=10**. The N=10 deficit is plausibly real (the mixture prior
degenerates when there are only 5 peers, so the method's whole
regularization mechanism doesn't fire) but also dominated by 30-sample
GPT-eval noise. Multi-seed averaging would settle it — the HP search
is one way to also try to close the gap.

## Your runbook: N=10 HP search

### Configs to test

Five single-knob variations from the current baseline + one combined-best,
per `documents/HP_SEARCH_PLAN.md`:

| Name | TID (sel/RL) | β | λ | reward_coeff |
|---|---|---|---|---|
| baseline | 90200 / 91200 | 0.01 | 1.0 | 0.1 |
| kl_low | 90201 / 91201 | **0.001** | 1.0 | 0.1 |
| kl_high | 90202 / 91202 | **0.1** | 1.0 | 0.1 |
| ortho_high | 90203 / 91203 | 0.01 | **5.0** | 0.1 |
| reward_high | 90204 / 91204 | 0.01 | 1.0 | **0.5** |
| combined_best | 90205 / 91205 | TBD (pick after first 5 finish) | TBD | TBD |

### Step 1: generate cfg files

The HP plan deliberately left these as templates. The simplest
approach: copy the existing single-cell configs and override the
swept knobs via CLI when launching (no new yaml needed). The
single-cell configs are already correct for n=10 — they have
`client_num: 10` and `sample_client_num: 5`:

- Selector base: `cfg/main_table/qwen_hhrlhf/fedvpagp_z_hybrid_adrop_kmeans_10212.yaml`
- RL base: `cfg/main_table/qwen_hhrlhf/hrl_z_hybrid_adrop_kmeans_11212.yaml`

CLI overrides will keep the runs reproducible without yaml proliferation.

### Step 2: launch the 5 selectors in parallel

Each on its own GPU (0-4). Stage 1 takes ~5h per run.

```bash
cd /home/kjb/ppfl
conda activate biscuit
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/home/kjb/ppfl

# baseline (90200, GPU 0)
nohup python -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_z_hybrid_adrop_kmeans_10212.yaml \
    device 0 \
    federate.save_to /hdd/hdd3/kjb/checkpoints/hhrl_choice_qwen2_hpsearch_n10_baseline_t90200.ckpt \
    expname hpsearch_n10_baseline_90200 \
    > outputs/90200.log 2>&1 &

# kl_low (90201, GPU 1)
nohup python -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_z_hybrid_adrop_kmeans_10212.yaml \
    device 1 \
    llm.vpl_kl_weight 0.001 \
    federate.save_to /hdd/hdd3/kjb/checkpoints/hhrl_choice_qwen2_hpsearch_n10_kl_low_t90201.ckpt \
    expname hpsearch_n10_kl_low_90201 \
    > outputs/90201.log 2>&1 &

# kl_high (90202, GPU 2)
nohup python -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_z_hybrid_adrop_kmeans_10212.yaml \
    device 2 \
    llm.vpl_kl_weight 0.1 \
    federate.save_to /hdd/hdd3/kjb/checkpoints/hhrl_choice_qwen2_hpsearch_n10_kl_high_t90202.ckpt \
    expname hpsearch_n10_kl_high_90202 \
    > outputs/90202.log 2>&1 &

# ortho_high (90203, GPU 3)
nohup python -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_z_hybrid_adrop_kmeans_10212.yaml \
    device 3 \
    llm.vpl_orthogonal_weight 5.0 \
    federate.save_to /hdd/hdd3/kjb/checkpoints/hhrl_choice_qwen2_hpsearch_n10_ortho_high_t90203.ckpt \
    expname hpsearch_n10_ortho_high_90203 \
    > outputs/90203.log 2>&1 &

# reward_high (90204, GPU 4) - reward_coeff is a Stage-2 knob; selector
# stays at baseline. Use the same selector ckpt as baseline (or share
# 90200's ckpt and skip this selector run if you prefer).
nohup python -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_z_hybrid_adrop_kmeans_10212.yaml \
    device 4 \
    federate.save_to /hdd/hdd3/kjb/checkpoints/hhrl_choice_qwen2_hpsearch_n10_reward_high_t90204.ckpt \
    expname hpsearch_n10_reward_high_90204 \
    > outputs/90204.log 2>&1 &
```

(Note: reward_high uses the same selector config — only Stage 2 differs.
You can save GPU time by reusing the baseline selector ckpt for 91204.)

### Step 3: launch the 5 RL stages after selectors finish

Wait for `final_*` checkpoints to appear in `/hdd/hdd3/kjb/checkpoints/`,
then launch Stage 2 RL. **Stage 2 uses a different entrypoint** —
`federatedscope/llm/rlhf/main.py` — and requires `--selector-cfg-file`.
Using `federatedscope/main.py` for RL crashes with `KeyError: 'win_data'`.

Stage 2 takes ~6h per run for N=10 (n=100 is much slower).

```bash
# baseline RL (91200, GPU 0)
nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/hrl_z_hybrid_adrop_kmeans_11212.yaml \
    --selector-cfg-file cfg/main_table/qwen_hhrlhf/fedvpagp_z_hybrid_adrop_kmeans_10212.yaml \
    device 0 \
    llm.rlhf_selector_checkpoint /hdd/hdd3/kjb/checkpoints/final_hhrl_choice_qwen2_hpsearch_n10_baseline_t90200.ckpt \
    federate.save_to /hdd/hdd3/kjb/checkpoints/hhrl_rlhf_qwen2_hpsearch_n10_baseline_t91200.ckpt \
    expname hpsearch_n10_baseline_rl_91200 \
    > outputs/91200.log 2>&1 &

# kl_low RL (91201, GPU 1)
nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/hrl_z_hybrid_adrop_kmeans_11212.yaml \
    --selector-cfg-file cfg/main_table/qwen_hhrlhf/fedvpagp_z_hybrid_adrop_kmeans_10212.yaml \
    device 1 \
    llm.vpl_kl_weight 0.001 \
    llm.rlhf_selector_checkpoint /hdd/hdd3/kjb/checkpoints/final_hhrl_choice_qwen2_hpsearch_n10_kl_low_t90201.ckpt \
    federate.save_to /hdd/hdd3/kjb/checkpoints/hhrl_rlhf_qwen2_hpsearch_n10_kl_low_t91201.ckpt \
    expname hpsearch_n10_kl_low_rl_91201 \
    > outputs/91201.log 2>&1 &

# kl_high RL (91202, GPU 2)
nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/hrl_z_hybrid_adrop_kmeans_11212.yaml \
    --selector-cfg-file cfg/main_table/qwen_hhrlhf/fedvpagp_z_hybrid_adrop_kmeans_10212.yaml \
    device 2 \
    llm.vpl_kl_weight 0.1 \
    llm.rlhf_selector_checkpoint /hdd/hdd3/kjb/checkpoints/final_hhrl_choice_qwen2_hpsearch_n10_kl_high_t90202.ckpt \
    federate.save_to /hdd/hdd3/kjb/checkpoints/hhrl_rlhf_qwen2_hpsearch_n10_kl_high_t91202.ckpt \
    expname hpsearch_n10_kl_high_rl_91202 \
    > outputs/91202.log 2>&1 &

# ortho_high RL (91203, GPU 3)
nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/hrl_z_hybrid_adrop_kmeans_11212.yaml \
    --selector-cfg-file cfg/main_table/qwen_hhrlhf/fedvpagp_z_hybrid_adrop_kmeans_10212.yaml \
    device 3 \
    llm.vpl_orthogonal_weight 5.0 \
    llm.rlhf_selector_checkpoint /hdd/hdd3/kjb/checkpoints/final_hhrl_choice_qwen2_hpsearch_n10_ortho_high_t90203.ckpt \
    federate.save_to /hdd/hdd3/kjb/checkpoints/hhrl_rlhf_qwen2_hpsearch_n10_ortho_high_t91203.ckpt \
    expname hpsearch_n10_ortho_high_rl_91203 \
    > outputs/91203.log 2>&1 &

# reward_high RL (91204, GPU 4) — uses baseline selector + reward_coeff override
nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/hrl_z_hybrid_adrop_kmeans_11212.yaml \
    --selector-cfg-file cfg/main_table/qwen_hhrlhf/fedvpagp_z_hybrid_adrop_kmeans_10212.yaml \
    device 4 \
    llm.reward_coeff 0.5 \
    llm.rlhf_selector_checkpoint /hdd/hdd3/kjb/checkpoints/final_hhrl_choice_qwen2_hpsearch_n10_baseline_t90200.ckpt \
    federate.save_to /hdd/hdd3/kjb/checkpoints/hhrl_rlhf_qwen2_hpsearch_n10_reward_high_t91204.ckpt \
    expname hpsearch_n10_reward_high_rl_91204 \
    > outputs/91204.log 2>&1 &
```

### Step 4: scoring

For each run, extract the win-rate trajectory at rounds 9, 19, 29, 39, 49:

```bash
grep -E "test_helpfulness_winrate|test_harmlessness_winrate" outputs/9120{0,1,2,3,4}.log
```

Then for each run identify the Pareto-best round (max H+HH on the Pareto
front) and compare to the baseline (91200). The winner is the config with
max(H+HH) at its Pareto-best round.

### Step 5: combined_best (if any single-knob beats baseline)

If `kl_low` and `ortho_high` both beat baseline, stack them into a single
combined config (TID 90205/91205) and run that as the headline N=10
config. If nothing beats baseline, baseline IS the N=10 answer — skip.

## Important warnings

- **Anthropic/hh-rlhf was removed from HF Hub.** `git pull` includes
  a fallback that loads from the local arrow cache automatically
  (`_load_local_arrow_hh_rlhf` in `federatedscope/llm/dataloader/hh_rlhf.py`).
  If your server doesn't have the cache at the default path, set
  `HH_RLHF_LOCAL_CACHE` (or `HH_RLHF_HARMLESS_DIR` and
  `HH_RLHF_HELPFUL_DIR`) before launching. The fallback identifies
  the subsets by train row count (42537 harmless, 43835 helpful) so
  any cache layout works as long as those files exist.
- The N=10 selector dict typically has ~7-10/10 clients populated. The
  category-aware fill kicks in for the missing ones; the warning
  "client_average_z_dict was missing N/10 clients" in the log is the
  fix doing its job, not a bug.
- 30-sample GPT evals are very noisy (±15pp std error). Differences
  under ~10pp are not reliably distinguishable from a single seed.
- The "feature extractor size mismatch" warning in RL logs is cosmetic
  (selector cfg's feature extractor dim differs from singlecell
  selector — the latent projection and z_to_embedding load cleanly
  and that's what z-conditioning needs).
- Round 50 ckpt is the only thing saved on disk — but it's rarely the
  Pareto-best round. For final paper numbers, save intermediate rounds
  (this is a separate TODO not in scope for the HP search).
- Stage 1 uses `federatedscope/main.py`; Stage 2 uses
  `federatedscope/llm/rlhf/main.py` and **requires** `--selector-cfg-file`.

## Compute estimate

| Phase | Time on 5 GPUs in parallel |
|---|---|
| 5 selectors | ~5h |
| 5 RLs | ~6h |
| Score, pick combined | ~30 min |
| combined_best sel+RL (if any) | ~11h |
| **Total** | **~22h end-to-end** |

## What to do when you finish

1. Update `documents/RESUME_STATE.md` (or write a new
   `documents/HP_SEARCH_RESULTS_N10.md`) with:
   - Pareto-best round per config (H, HH, Σ)
   - Identified winner
   - Recommended next step (multi-seed the winner, or proceed to full
     Main Table with the winner's hyperparameters)
2. `git add`/`commit`/`push` the results doc to `refined`.
3. Send a one-paragraph summary to the user.

## Files you should read before starting

In priority order:
1. `CLAUDE.md` — codebase overview, algorithm, config knobs.
2. `documents/HP_SEARCH_PLAN.md` — the plan this handoff implements.
3. `documents/RESUME_STATE.md` — what happened in the last session.
4. `cfg/main_table/qwen_hhrlhf/fedvpagp_z_hybrid_adrop_kmeans_10212.yaml`
   and `hrl_z_hybrid_adrop_kmeans_11212.yaml` — base configs for the
   single-cell sanity check; CLI overrides keep them reproducible.
5. `documents/EXPERIMENT_GUIDE.md` — general experiment runbook.

## Files that recently changed (`git log -p` for detail)

- `federatedscope/llm/llm_local/{server,client}.py` — VPL broadcast +
  z-collection fix.
- `federatedscope/llm/trainer/vpl_reward_choice_trainer.py` — dtype
  cast + posterior aggregation fixes.
- `federatedscope/llm/rlhf/load_vpl_components.py` — new
  `fill_missing_client_z_with_category_mean` helper.
- `federatedscope/llm/metric/winrate_metrics.py` — apply
  category-aware fill on load.
- `federatedscope/llm/rlhf/standalone_training.py` — apply the same
  fill at three Stage-2 load sites + use config-declared client_num
  over `max(dict.keys())`.

Good luck.
