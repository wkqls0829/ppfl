# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Implementation of **FedVPA-GP** (Federated Variational Preference Alignment with Gumbel-Softmax Prior), submitted to **ICML 2026** and currently under review. Built on Alibaba's FederatedScope framework.

**Core problem**: Existing federated alignment methods (FedDPO, FedBiscuit) enforce a monolithic reward model that averages out conflicting user preferences (e.g., helpfulness vs. harmlessness). Naive VPL in FL suffers from **posterior collapse** due to local data scarcity and heterogeneity.

**Our solution (FedVPA-GP)**: Two mechanisms to overcome posterior collapse in federated VPL:
1. **Federated Mixture Prior**: Aggregates peer client posteriors as a dynamic prior `p_mixture(z) = Σ w_j · N(z; μ_j, σ_j²)` with learnable Gumbel-Softmax weights (Eq. 6-8 in paper)
2. **Orthogonal Loss** (CLOP-based): Enforces separation of preference prototypes in latent space via pull loss + orthonormality constraint (Eq. 9)

**Paper contributions**:
- Federated variational preference alignment that captures diverse user intents while preserving privacy
- Stabilized variational inference via Federated Mixture Prior + Orthogonal Loss to prevent posterior collapse
- Empirical validation on HH-RLHF: FedVPA-GP significantly outperforms FedDPO, FedBiscuit, FedVPL across Qwen-2 0.5B and Gemma-2B at N∈{10,50,100} clients, with robust generalization to unseen clients

## Environment & Setup

```bash
# Conda environment (always use 'biscuit')
conda activate biscuit

# Install (editable, with LLM support)
pip install -e .[llm]

# PyTorch CUDA install (if needed)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

## Common Commands

```bash
# Run an experiment
python federatedscope/main.py --cfg cfg/vpl-gp/hrl-ultrafeedback-50001.yaml

# Override config values via CLI
python federatedscope/main.py --cfg config.yaml federate.total_round_num 30 train.optimizer.lr 1e-5

# Run all tests
python tests/run.py

# Run specific test pattern
python tests/run.py --pattern test_toy_lr.py

# Linting
flake8 federatedscope/
yapf -i -r federatedscope/

# Pre-commit hooks (yapf, flake8, pyroma)
pre-commit run --all-files

# Monitor experiment
tail -f outputs/{tid}.log

# Check running experiments
ps aux | grep "python.*main.py" | grep -v grep
```

## Experiment Execution

### Two-Stage Training (Paper Sec. 4.4)

Each experiment has two stages:
1. **Stage 1 — Federated Selector Training (HHST)**: Trains a variational binary preference selector via FL. Each client learns posterior q_ϕ(z|D_i), predicts choices conditioned on z. Base LLM frozen; only VPL components (feature extractor, variational encoder, latent projection) and LoRA adapters are trained. Loss = L_recon + β·L_KL + λ·L_ortho (Eq. 10).
2. **Stage 2 — Conditional RLHF (HRL)**: Centralized DPO on server using the converged selector as reward model. Policy conditioned on inferred z via Z-TO-EMBEDDING injection into input embeddings. Evaluated by GPT-4 win-rate.

### Two Environments

| Environment | Scripts | WORK_DIR | Use case |
|-------------|---------|----------|----------|
| **SLURM cluster** | `scripts/slurm/main_table/` | `/home2/jbkoo/ppfl` | Main Table experiments, batch submission |
| **Local server** | `scripts/server/` | `/home/kjb/ppfl` | DP experiments, VPL-GP dev, debugging |

### Running Experiments

```bash
# Local server (always activate biscuit first)
conda activate biscuit
CUDA_VISIBLE_DEVICES=3 bash scripts/server/vpl-gp/hrl-ultrafeedback-50001.sh

# SLURM cluster
sbatch scripts/slurm/main_table/run_selector_gemma.sh fedvpagp 10 62130
sbatch scripts/slurm/main_table/run_rl_gemma.sh fedvpagp 10 63130 62130
```

### GPU Allocation Rules

- **Local server**: Set `device` in YAML to the desired GPU index directly (0-6). No `CUDA_VISIBLE_DEVICES` needed.
- **SLURM cluster**: YAML `device` must be `0` (SLURM sets `CUDA_VISIBLE_DEVICES` automatically).

### Experiment ID (TID) Ranges

| Range | Algorithm | Description |
|-------|-----------|-------------|
| 10000-19999 | FedDPO | DPO baseline |
| 20000-29999 | FedBiscuit | Multi-LoRA baseline (U=3 adapters) |
| 30000-39999 | FedVPL | Naive VPL (no GP prior, no orthogonal loss) |
| 40000-49999 | VPL-GP baseline | VPL-GP without orthogonal loss |
| 50000-59999 | VPL-GP orthogonal | VPL-GP with orthogonal loss |
| 62xxx/63xxx | Main Table | Selector (62xxx) and RL (63xxx) for paper Table 1 |
| 10100-10119 | Z-separation & KL ablation | Logvar cap, KL weight sweeps, ortho-only variants |
| 10200-10203 | Main comparison (selector) | FedBiscuit, FedVPL, FedVPA-GP KL-only, FedVPA-GP full |
| 11200-11203 | Main comparison (RL) | Corresponding Stage 2 DPO for 10200-10203 |
| 11117 | RL ortho-only | Stage 2 DPO for ortho-only selector (10117) |

### Script Template Pattern

All experiment scripts follow this structure:
```bash
tid={experiment_id}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

nohup python -u federatedscope/main.py \
    --cfg cfg/{algorithm}/{config}.yaml \
    federate.save_to /hdd/hdd3/kjb/checkpoints/{name}_${tid}.ckpt \
    expname "{expname}_t${tid}" \
    > outputs/${tid}.log 2>&1 &
```

### Creating New Experiments

1. Copy an existing YAML config: `cp cfg/vpl-gp/hhst-40100.yaml cfg/vpl-gp/hhst-{new_tid}.yaml`
2. Modify: `device`, hyperparameters
3. Copy and modify script: `cp scripts/server/vpl-gp/hhst-40100.sh scripts/server/vpl-gp/hhst-{new_tid}.sh`
4. Update `tid` and `--cfg` path in the script

## Architecture

### Entry Point & Execution Flow

`federatedscope/main.py` -> parses config YAML -> `get_data()` -> builds server/client workers -> `FedRunner` orchestrates FL rounds.

### Core FL Framework (`federatedscope/core/`)

- `fed_runner.py` - Orchestrates federated learning rounds
- `workers/server.py` / `workers/client.py` - Server/client logic (message passing, aggregation)
- `configs/` - YACS-based configuration system. `cfg_llm.py` has VPL-specific params
- `aggregators/` - Model aggregation strategies (FedAvg, etc.)

### LLM Module (`federatedscope/llm/`) - Main Research Code

**Trainers** (`llm/trainer/`) - Inheritance: `BaseTrainer` -> `TorchTrainer` -> `RewardChoiceTrainer` -> `VPLRewardChoiceTrainer`
- `vpl_reward_choice_trainer.py` (47K) - Core VPL trainer with variational inference
- `vpl_gp_reward_choice_trainer.py` - VPL-GP extension with Gumbel-Softmax
- `reward_choice_trainer.py` - Binary preference (DPO-style) baseline
- `PPO_reward_trainer.py` - PPO RL trainer

**Models** (`llm/model/`)
- `variational_encoder.py` - Standard VPL encoder (input: chosen/rejected embeddings -> mu, logvar)
- `variational_encoder_gp.py` - VPL-GP encoder with Gumbel-Softmax prior, extends VariationalEncoder
- `adapter_builder.py` - PEFT/LoRA adapter setup
- `model_builder.py` - HuggingFace model initialization

**VPL-GP Server/Client** (`llm/llm_local/`)
- `server.py` - Aggregates client z-distributions (mu, logvar) into mixture prior, computes balanced orthogonal labels via k-means, broadcasts prior + labels
- `client.py` - Sends (mu, logvar) to server after adapter filtering, receives aggregated prior + labels, calls `update_prior_from_server()`

**Metrics** (`llm/metric/`)
- `winrate_metrics.py` - GPT-4o-mini evaluation for helpfulness/harmlessness win-rates

### Component Registry (`register.py`)

All trainers, models, datasets, metrics are registered via `register_*()` functions. Key registered names:
- Trainers: `llmrewardtrainer`, `vplrewardchoicetrainer`, `vplgprewardchoicetrainer`, `llmdporewardtrainer`, `llmdporewardchoicetrainer`, `llmrewardchoicetrainer`
- Data: `ultrafeedback@llm`, `hh-rlhf@llm`
- Models: `huggingface_llm`

### Configuration (`cfg/`)

YAML configs drive experiments. Key subdirectories: `vpl-gp/`, `main_table/`, `feddpo/`, `fedbiscuit/`, `fedvpl/`, `llama-vplgp/`.

## FedVPA-GP Algorithm (Paper Sec. 4)

### Local Loss Function (Eq. 10)

```
L_i(θ,ϕ) = L_recon + β · L_reg + λ · L_ortho
```
- **L_recon**: Negative log-likelihood of preference prediction conditioned on z
- **L_reg**: KL(q_ϕ(z|D_i) || p_mixture(z)) — prior matching via Federated Mixture Prior
- **L_ortho**: Orthogonal loss for preference separation

### Inference Network (Sec. 4.1)

- Extract hidden representations h_A, h_B from frozen base LLM
- Compute **difference embedding**: Δh = h_chosen − h_rejected (excluding prompt tokens)
- Feature extractor (MLP) distills preference-specific signals from Δh
- Variational encoder outputs posterior: q_ϕ(z|D_i) = N(z; μ_i, σ²_i·I)
- Reparameterization trick: z_i = μ_i + σ_i ⊙ ε, ε ~ N(0,I)
- **Latent Conditional Reward**: logits(s_A, s_B | z_i) = logits_base(s_A, s_B) + f_θ(z_i) (Eq. 2, 11)

### Federated Mixture Prior (Sec. 4.1)

```
p_mixture(z) = Σ_j∈S  w_j · N(z; μ_j, σ²_j·I)     (Eq. 6)
```
- Weights computed via **Gumbel-Softmax relaxation** (Eq. 8): w_j = exp((log π_j + g_j)/τ) / Σ_k exp((log π_k + g_k)/τ)
- π_j are learnable logits, g_j ~ Gumbel(0,1), τ is temperature
- KL computed via **log-sum-exp trick** for numerical stability (Eq. 7)
- Model automatically learns to upweight informative peers with similar preference structures

### Orthogonal Loss (Sec. 4.2, CLOP-based)

Prevents posterior collapse by enforcing separation of preference prototypes:
```
L_orthogonal(z) = ||z − p_{y*_i}||² + γ · ||PP^T − I_M||²_F    (Eq. 9)
```
- M learnable prototype vectors initialized via QR decomposition (orthonormal basis)
- Server assigns labels y*_i via balanced k-means on collected client means {μ̄_i}
- Pull term aligns z to assigned prototype; orthonormality constraint keeps prototypes separated

### VPL Configuration Options

#### Basic VPL
| Option | Default | Description |
|--------|---------|-------------|
| `vpl_latent_dim` | 32 | Latent space dimension (16-64) |
| `vpl_kl_weight` | 0.1 | KL divergence weight (0.01-1.0) |
| `vpl_feature_method` | `choice_logits` | Feature extraction method |
| `vpl_use_feature_difference` | False | Use embedding difference (chosen-rejected) |
| `vpl_use_difference_only` | False | Use only difference, remove general info |

#### GP Prior
| Option | Default | Description |
|--------|---------|-------------|
| `vpl_use_gp_prior` | False | Enable mixture prior from other clients |
| `vpl_gp_temperature` | 1.0 | Gumbel-Softmax temperature (0.5-2.0) |

#### Orthogonal Loss
| Option | Default | Description |
|--------|---------|-------------|
| `vpl_orthogonal_weight` | 0.0 | Pull loss weight (0=disabled, CLOP recommends 10.0) |
| `vpl_orthogonal_orthonorm_weight` | 0.1 | Orthonormal constraint weight |
| `vpl_use_manual_orthogonal_labels` | False | True=ground-truth data-category labels (recommended), False=k-means |
| `vpl_deep_projection` | False | Replace Linear(32→2) with MLP(32→64→32→2) |
| `vpl_logit_dropout` | 0.0 | Drop base logits during training so z must carry signal |
| `vpl_num_prototypes` | num_clients | Number of prototypes (auto 2 for hh-rlhf) |
| `vpl_prototype_scale` | 5.0 | Prototype distance from origin (2.0-10.0) |
| `vpl_tsne_visualize_freq` | 10 | t-SNE visualization frequency (rounds) |

### Feature Extraction Options

```
vpl_use_feature_difference
  ├─ False → choice_logits: [logit_A_chosen, logit_B_chosen, logit_A_rejected, logit_B_rejected]
  └─ True
      ├─ vpl_use_difference_only=False → [chosen_emb, rejected_emb, difference] (3×emb_dim)
      └─ vpl_use_difference_only=True → [difference] only (emb_dim) ⭐ Recommended
```

### Paper Hyperparameters (Table 3) — Updated after refinement experiments

| Parameter | Selector (Stage 1) | RL (Stage 2) |
|-----------|-------------------|--------------|
| Learning rate | 1e-4 (Gemma), 1e-5 (Qwen) | 1e-4 (Gemma), 1e-5 (Qwen) |
| Batch size | 4 | 1 |
| Grad accum steps | 8 | 32 (Qwen), 4 (Gemma) |
| Local update steps | 30 | 30 |
| Total rounds | 50 | 50 |
| KL weight (β) | 0.01 | – |
| Orthogonal weight (λ) | 1.0 | – |
| Orthonorm weight (γ) | 0.1 | – |
| Gumbel-Softmax temp (τ) | 1.0 | – |
| Prototype scale | 5.0 | – |
| Latent dim (d) | 32 | – |
| Max logvar | -4.0 | – |
| Deep projection | True (32→64→32→2) | – |
| Logit dropout | 0.5 | – |
| Manual orthogonal labels | True | – |
| LoRA rank/alpha/dropout | 8 / 16 / 0.05 | 8 / 16 / 0.05 |
| Reward coefficient | – | 0.1 |

### Recommended Config (FedVPA-GP full)

```yaml
llm:
  vpl_latent_dim: 32
  vpl_kl_weight: 0.01
  vpl_use_feature_difference: True
  vpl_use_difference_only: True
  vpl_use_gp_prior: True
  vpl_gp_temperature: 1.0
  vpl_max_logvar: -4.0
  vpl_deep_projection: True
  vpl_logit_dropout: 0.5
  vpl_orthogonal_weight: 1.0
  vpl_orthogonal_orthonorm_weight: 0.1
  vpl_use_manual_orthogonal_labels: True
  vpl_num_prototypes: 2
  vpl_prototype_scale: 5.0
```

### Server/Client Communication Flow (Algorithms 1 & 2)

1. Server broadcasts model params (θ,ϕ) + mixture prior {(μ_j, σ²_j), w_j} + orthogonal labels to sampled clients S^t
2. Client updates local prior (or uses N(0,I) if round 1), trains E local steps on D_i with loss L_recon + β·L_KL + λ·L_ortho
3. Client sends updated (θ_i, ϕ_i, μ̄_i, σ̄²_i, |D_i|) to server — **prior_logits excluded from FedAvg** so each client keeps its own Gumbel-Softmax weights
4. Server aggregates via FedAvg, collects z-distributions, assigns orthogonal labels (manual labels based on data category, not k-means)

### Stage 2 Z-Conditional DPO

Stage 2 DPO is conditioned on client z vectors:
1. `client_average_z_dict` from Stage 1 checkpoint maps each client_id to its mean z
2. Each DPO training sample has z injected: `inputs_embeds = input_embeddings(input_ids) + z_to_embedding(z)`
3. Both ref model and policy model see z-conditioned inputs
4. `z_to_embedding` (Linear: latent_dim → embedding_dim) is trainable during DPO
5. Generation/evaluation also conditioned on z (set `rlhf_use_variational_generation: True`)
6. Entry point: `federatedscope/llm/rlhf/main.py` (NOT `federatedscope/main.py`), requires `--selector-cfg-file`

## Paper Experiments (Sec. 5)

### Main Table (Table 1)

GPT-4 win-rate (%) on HH-RLHF with strict Non-IID partition (50% helpful clients, 50% harmless clients).

- **Models**: Qwen-2 0.5B, Gemma-2B
- **Methods**: FedDPO, FedBiscuit, FedVPL, **FedVPA-GP** (ours)
- **Client counts**: N ∈ {10, 50, 100} (sampling: 5/round for N=10, 10/round for N=50,100)
- **Total**: 2 models × 4 methods × 3 client counts = 24 experiments (× 2 stages = 48)
- **Key result**: FedVPA-GP achieves Pareto improvement — higher win-rates in BOTH helpfulness AND harmlessness

### Ablation Study (Fig. 4)

Four variants compared: FedVPL → +Ortho → +GB Prior → FedVPA-GP (full). Both components contribute; combined gives best trade-off.

### Unseen Client Generalization (Table 2)

20 clients (10 helpful, 10 harmless), 5+5 seen for training, 5+5 unseen for eval. FedVPA-GP maintains high win-rates on unseen clients via inference alone (no parameter updates needed).

### Baselines
- **FedDPO** (Ye et al., 2024): Standard federated DPO, monolithic reward — `llmdporewardchoicetrainer`
- **FedBiscuit** (Wu et al., 2024): Multi-LoRA adapters (U=3), coarse personalization — `llmrewardchoicetrainer`
- **FedVPL**: Naive VPL in FL with fixed N(0,I) prior, no orthogonal loss — `vplrewardchoicetrainer` with `vpl_use_gp_prior: False`
- **FedVPA-GP** (ours): Full method — `vplrewardchoicetrainer` or `vplgprewardchoicetrainer` with `vpl_use_gp_prior: True`

## Monitoring & Outputs

- **Logs**: `outputs/{tid}.log`
- **Checkpoints**: `/hdd/hdd3/kjb/checkpoints/*_{tid}.ckpt`
- **t-SNE plots**: `exp/{expname}/cross_client_z_tsne_round_{round_num}.png`
- **WandB projects**: `fvpl-selector` (HHST), `fvpl-rl` (HRL)
- **WandB metrics**: `vpl_total_loss`, `vpl_reconstruction_loss`, `vpl_kl_loss`, `vpl_orthogonal_loss`, `train_avg_loss`, `acc`

## Common Errors

- **`CUDA error: invalid device ordinal`**: YAML `device` must be 0, use `CUDA_VISIBLE_DEVICES` for physical GPU
- **`ModuleNotFoundError: No module named 'torch'`**: Not in `biscuit` conda env
- **protobuf import error (LLaMA)**: `pip install "protobuf<4.21.0"`
- **OOM**: Reduce `batch_size`, increase `grad_accum_step`, reduce `tok_len`/`max_new_token`, ensure `is_enable_half: True`

## Documentation Reference

- `Federated Variational Preference Alignment.pdf` — The ICML 2026 submission paper
- `docs/` — Technical docs: VPL architecture, KL loss math, GP prior, orthogonal loss, configuration options
- `documents/` — Operational docs: experiment guides, GPU allocation, API setup, troubleshooting
- `docs/VPL_CONFIGURATION_OPTIONS.md` — Complete VPL config reference
- `documents/EXPERIMENT_GUIDE.md` — Full experiment execution guide
- `documents/CONTEXT_FOR_NEW_SESSION.md` — Detailed project context with implementation specifics

## Code Style

- Max line length: 79 chars
- Formatter: yapf (v0.32.0)
- Linter: flake8 (permissive ignore list, see `.flake8`)
- Double quotes for strings
- Python >= 3.9, PyTorch >= 1.13.0
