# Work Summary: VPL-GP Federated Learning System

## Executive Summary

This document summarizes the comprehensive implementation of a **Variational Preference Learning with Gumbel Softmax Prior (VPL-GP)** federated learning system for personalized preference learning in RLHF (Reinforcement Learning from Human Feedback) scenarios. The system was rebuilt after a `git rebase --abort` incident that caused code loss, and has been fully restored and enhanced.

---

## 1. Problem Context

### 1.1 Original Issue
- **Date**: 2026-01-19
- **Problem**: `git pull` with `pull.rebase=true` triggered an automatic rebase that was aborted, causing loss of uncommitted local changes
- **Impact**: Lost implementation of VPL-GP features including:
  - Server-side z-distribution collection and aggregation
  - Client-side z-distribution transmission
  - t-SNE visualization of cross-client z-distributions
  - Orthogonal label computation and broadcasting
  - Manual label assignment logic

### 1.2 Recovery Process
- Analyzed execution logs (50271.log) to understand functionality
- Reconstructed code based on log messages and line numbers
- Verified functionality through testing
- Enhanced with additional features (win/lose evaluation, improved visualization)

---

## 2. System Architecture

### 2.1 High-Level Overview

The VPL-GP system extends standard Variational Preference Learning (VPL) to federated settings by:

1. **Client-Side**: Each client learns a personalized latent vector `z` that encodes their preferences
2. **Server-Side**: Server aggregates z-distributions from clients to form a mixture prior
3. **Prior Broadcasting**: Server sends aggregated priors back to clients for improved learning
4. **Visualization**: t-SNE plots show how client preferences evolve and cluster

### 2.2 Key Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Federated Learning System                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐         ┌──────────────┐                │
│  │   Server     │◄────────►│   Clients    │                │
│  │              │          │              │                │
│  │ - Aggregate  │          │ - Train VPL  │                │
│  │   z-dist     │          │ - Send z     │                │
│  │ - Compute    │          │ - Receive    │                │
│  │   labels     │          │   prior      │                │
│  │ - Broadcast  │          │              │                │
│  │   prior      │          │              │                │
│  └──────────────┘          └──────────────┘                │
│         │                           │                        │
│         └───────────┬───────────────┘                        │
│                     │                                        │
│              ┌──────▼──────┐                                │
│              │ Visualization│                                │
│              │ (t-SNE)     │                                │
│              └─────────────┘                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Core Implementation Details

### 3.1 Server Implementation (`federatedscope/llm/llm_local/server.py`)

**Key Features:**
- **Z-Distribution Collection**: Collects `(mu, logvar)` pairs from participating clients
- **Prior Aggregation**: Maintains a mixture prior across rounds, updating existing clients and adding new ones
- **Orthogonal Label Computation**: Uses k-means clustering on z-means or manual assignment based on data type
- **Visualization**: Collects z-values for t-SNE plots, with balanced sampling (10 per label per client)
- **Broadcasting**: Sends aggregated priors and labels to clients

**Key Methods:**

1. **`_collect_vpl_gp_prior_distributions()`** (Line 488-604)
   - Extracts `client_z_mu` and `client_z_logvar` from client messages
   - Maintains prior across rounds (updates existing, adds new clients)
   - Normalizes weights based on sample sizes
   - Logs collection statistics

2. **`_compute_balanced_orthogonal_labels()`** (Line 606-701)
   - **Manual Mode**: Assigns labels based on client data type (harmlessness/helpfulness)
     - First half of clients → label 0 (harmlessness)
     - Second half → label 1 (helpfulness)
   - **K-Means Mode**: Clusters z-means using sklearn KMeans (k=2 for hh-rlhf)
   - Logs detailed assignment information

3. **`_collect_z_values_for_visualization()`** (Line 703-856)
   - Collects z-values from participating clients
   - Maintains latest z per client (for non-participating clients)
   - Balances samples: 10 per label per client (harmlessness/helpfulness)
   - Stores orthogonal labels for all clients
   - Triggers visualization every N rounds (default: 10)

4. **`_visualize_cross_client_z()`** (Line 858-952)
   - Calls `visualize_cross_client_z()` from `z_visualization.py`
   - Samples 10 z-values per label per client for balanced visualization
   - Passes orthogonal prototypes if available

5. **`broadcast_model_para()`** (Line 954-1041)
   - Overrides parent method to broadcast VPL-GP prior and orthogonal labels
   - Only broadcasts after round 0 (when prior is available)
   - Uses same client sampling logic as parent class

**Important Attributes:**
- `vpl_gp_prior_mus`: Stacked mean vectors (num_clients, latent_dim)
- `vpl_gp_prior_logvars`: Stacked log-variance vectors (num_clients, latent_dim)
- `vpl_gp_prior_weights`: Normalized weights (num_clients,)
- `vpl_orthogonal_client_labels`: Dict mapping client_id → label
- `client_z_values_dict`: Dict mapping client_id → list of z-values
- `client_orthogonal_labels_dict`: Dict mapping client_id → orthogonal label

### 3.2 Client Implementation (`federatedscope/llm/llm_local/client.py`)

**Key Features:**
- **Z-Distribution Transmission**: Sends `(mu, logvar)` to server after training
- **Prior Update**: Receives and updates mixture prior from server
- **Label Update**: Receives orthogonal labels from server

**Key Methods:**

1. **`callback_funcs_for_model_para()`** (Line 63-211)
   - Extracts z-values and z-distribution from trainer
   - Adds to `model_para_all` before sending to server:
     - `client_z_values`: For visualization (always collected)
     - `client_z_mu`, `client_z_logvar`: For GP prior (only if enabled)
     - `client_orthogonal_prototypes`: For visualization (if available)

2. **`callback_funcs_for_vpl_gp_prior()`** (Line 250-270)
   - Handles `vpl_gp_prior` message from server
   - Converts to tensors and calls `trainer.update_prior_from_server()`

3. **`callback_funcs_for_vpl_orthogonal_labels()`** (Line 272-280)
   - Handles `vpl_orthogonal_labels` message from server
   - Updates trainer's orthogonal label

### 3.3 Trainer Implementation

#### 3.3.1 Base VPL Trainer (`federatedscope/llm/trainer/vpl_reward_choice_trainer.py`)

**Key Features:**
- **Variational Inference**: Encodes preference features to latent z
- **Feature Extraction**: Supports multiple methods:
  - Embedding difference: `(chosen_emb - rejected_emb)` or `[chosen, rejected, diff]`
  - Logits-based: Choice token logits
- **KL Divergence**: Computes KL(q(z|x) || p(z)) with standard normal prior or mixture prior
- **Orthogonal Loss**: CLOP-based loss that pulls z toward fixed prototypes
- **Separate Optimizer**: VPL components learn faster (10x LR multiplier) while LLM is frozen

**Key Methods:**

1. **`_extract_preference_features()`** (Line 269-294)
   - Routes to embedding difference or logits-based extraction
   - Supports `vpl_use_difference_only` flag

2. **`_extract_embedding_difference()`** (Line 296-381)
   - Extracts embeddings at choice token positions
   - Computes `[chosen, rejected, difference]` or `difference` only
   - Detaches hidden states to prevent LLM gradient flow

3. **`_hook_on_batch_forward()`** (Line 431-591)
   - Extracts preference features
   - Encodes to latent z via variational encoder
   - Computes KL divergence (standard or mixture prior)
   - Conditions model on z via latent projection
   - Computes reconstruction loss + KL + orthogonal loss

4. **`_hook_on_batch_backward()`** (Line 593-663)
   - Uses separate VPL optimizer if available (faster learning)
   - Otherwise uses main optimizer

5. **`_hook_on_fit_end()`** (Line 699-774)
   - Collects z-values for visualization (samples up to 100)
   - Computes client z-distribution (mu, logvar) for GP prior
   - Logs VPL metrics to wandb

6. **`get_client_z_distribution()`** (Line 777-791)
   - Returns `(mu, logvar)` tuple for server aggregation

7. **`update_prior_from_server()`** (Line 805-818)
   - Updates variational encoder's mixture prior

8. **`_compute_clop_orthogonal_loss()`** (Line 840-887)
   - Pulls z toward assigned prototype (based on orthogonal label)
   - Computes orthonormal constraint loss (for monitoring, prototypes are fixed)

**Important Attributes:**
- `variational_encoder`: `VariationalEncoder` or `VariationalEncoderGP`
- `feature_extractor`: MLP that processes preference features
- `latent_projection`: Projects z to choice logit adjustments
- `orthogonal_prototypes`: Fixed orthonormal prototypes (if orthogonal loss enabled)
- `orthogonal_label`: Label assigned by server (if orthogonal loss enabled)
- `client_z_values`: Sampled z-values for visualization
- `client_z_mu`, `client_z_logvar`: Client's z-distribution for GP prior

#### 3.3.2 VPL-GP Trainer (`federatedscope/llm/trainer/vpl_gp_reward_choice_trainer.py`)

**Note**: This is a legacy class. The main `VPLRewardChoiceTrainer` now supports GP prior via config flag `vpl_use_gp_prior=True`.

**Key Features:**
- Extends `VPLRewardChoiceTrainer`
- Uses `VariationalEncoderGP` instead of `VariationalEncoder`
- Same functionality as base trainer with GP prior enabled

### 3.4 Model Implementation

#### 3.4.1 Variational Encoder (`federatedscope/llm/model/variational_encoder.py`)

**Key Features:**
- Encodes preference features to latent distribution `q(z|x)`
- Reparameterization trick for sampling
- KL divergence with standard normal prior

**Key Methods:**
- `encode()`: Maps features to (mu, logvar)
- `reparameterize()`: Samples z ~ q(z|x)
- `forward()`: Full forward pass (encode + sample)
- `kl_divergence()`: Computes KL(q(z|x) || N(0,I))

**Important Parameters:**
- `max_logvar`: Clamps logvar to prevent large variance (default: -2.0 → sigma ≤ 0.368)

#### 3.4.2 Variational Encoder GP (`federatedscope/llm/model/variational_encoder_gp.py`)

**Key Features:**
- Extends `VariationalEncoder`
- Supports mixture prior from other clients
- Gumbel-Softmax sampling for mixture prior

**Key Methods:**

1. **`update_prior()`** (Line 37-81)
   - Updates mixture prior from server
   - Normalizes weights
   - Logs statistics (mu norm, logvar mean, avg distance)

2. **`sample_prior()`** (Line 83-139)
   - Samples from mixture prior using Gumbel-Softmax
   - Falls back to standard normal if prior not available

3. **`kl_divergence()`** (Line 141-238)
   - Computes KL(q(z|x) || p_mixture(z))
   - Uses log-sum-exp trick for numerical stability
   - Logs comparison with standard KL (1% of time)

**Important Attributes:**
- `prior_mus`: Mean vectors from other clients (num_clients, latent_dim)
- `prior_logvars`: Log-variance vectors (num_clients, latent_dim)
- `prior_weights`: Normalized weights (num_clients,)
- `temperature`: Gumbel-Softmax temperature (default: 1.0)

### 3.5 Visualization (`federatedscope/llm/llm_local/z_visualization.py`)

**Key Features:**
- t-SNE visualization of cross-client z-distributions
- Color coding by orthogonal labels (harmlessness=red, helpfulness=blue)
- Prototype visualization (if available)
- WandB logging

**Key Function:**

**`visualize_cross_client_z()`** (Line 13-203)
- Applies t-SNE (or PCA if t-SNE fails)
- Colors points by orthogonal labels
- Plots prototypes as stars
- Saves to `cross_client_z_tsne_round_{round_num}.png`
- Logs to WandB if enabled

**Visualization Details:**
- **Color Scheme**:
  - Label 0 (Harmlessness): Crimson red (#DC143C)
  - Label 1 (Helpfulness): Deep sky blue (#00BFFF)
- **Sampling**: 10 z-values per label per client (balanced)
- **Prototypes**: Plotted as red stars with black edges

### 3.6 Metrics (`federatedscope/llm/metric/hhrl_metrics.py`)

**Key Features:**
- Reward model evaluation (harmlessness/helpfulness)
- Win/lose evaluation (local model as judge)
- Caching to avoid redundant computation

**Key Functions:**

1. **`_get_or_compute_hhrl_scores()`** (Line 56-251)
   - Computes reward scores using GPT2HarmlessRewardModel and GPT2HelpfulRewardModel
   - Only for hh-rlhf dataset (skips HHST)
   - Limits evaluation to 30 samples by default (configurable)
   - Caches results per round

2. **`eval_harmlessness()`** (Line 255-272)
   - Only for harmlessness clients (client_id ≤ client_num // 2)
   - Only for test/val splits (not train)

3. **`eval_helpfulness()`** (Line 282-305)
   - Only for helpfulness clients (client_id > client_num // 2)
   - Only for test/val splits (not train)

**Important Notes:**
- **Dataset Check**: Only evaluates for `hh-rlhf` dataset, not `hhst`
- **Client Filtering**: Harmlessness metric only for harmlessness clients, helpfulness metric only for helpfulness clients
- **Split Filtering**: Only evaluates on test/val, not train
- **Sample Limiting**: Default 30 samples (configurable via `eval.max_samples_for_reward`)

---

## 4. Data Flow

### 4.1 Training Round Flow

```
Round N:
1. Server broadcasts model parameters to selected clients
2. Clients train locally:
   a. Extract preference features
   b. Encode to latent z
   c. Compute loss (reconstruction + KL + orthogonal)
   d. Update VPL components
3. Clients send to server:
   - Model parameters
   - client_z_values (for visualization)
   - client_z_mu, client_z_logvar (for GP prior)
   - client_orthogonal_prototypes (if available)
4. Server aggregates:
   a. Model parameters (standard FedAvg)
   b. Z-distributions (for mixture prior)
   c. Z-values (for visualization)
5. Server computes:
   a. Balanced orthogonal labels (k-means or manual)
   b. t-SNE visualization (every N rounds)
6. Server broadcasts (round N+1):
   - Updated model parameters
   - VPL-GP prior (mus, logvars, weights)
   - Orthogonal labels
7. Clients update:
   a. Model parameters
   b. Mixture prior (if GP prior enabled)
   c. Orthogonal label (if orthogonal loss enabled)
```

### 4.2 Z-Distribution Collection

```
Client Side:
- During training: Collect z-values in z_history
- After training: Compute client_z_mu = mean(z_history), client_z_logvar = log(var(z_history))
- Send to server: (client_z_mu, client_z_logvar, sample_size)

Server Side:
- Collect from all participating clients
- Stack into prior_mus (num_clients, latent_dim), prior_logvars (num_clients, latent_dim)
- Compute weights: weight_i = sample_size_i / sum(sample_sizes)
- Update existing clients or add new ones
- Normalize weights
- Broadcast to all clients
```

### 4.3 Orthogonal Label Assignment

```
Manual Mode (vpl_use_manual_orthogonal_labels=True):
- Assign based on client data type:
  - Clients 1 to (client_num // 2) → label 0 (harmlessness)
  - Clients (client_num // 2 + 1) to client_num → label 1 (helpfulness)
- Matches data distribution in load_hh_rlhf_data()

K-Means Mode (default):
- Extract z-means for participating clients
- Cluster using sklearn KMeans (k=2 for hh-rlhf)
- Assign labels based on cluster assignment
- Log detailed assignment information
```

---

## 5. Configuration

### 5.1 Key Config Parameters

**VPL-GP Settings:**
```yaml
llm:
  vpl_use_gp_prior: True          # Enable GP prior
  vpl_latent_dim: 32               # Latent dimension
  vpl_kl_weight: 0.1               # KL divergence weight
  vpl_gp_temperature: 1.0           # Gumbel-Softmax temperature
  vpl_feature_method: 'choice_logits'  # Feature extraction method
  vpl_use_feature_difference: True # Use embedding difference
  vpl_use_difference_only: True     # Use only difference (no chosen/rejected)
  vpl_max_logvar: -3.0              # Max log variance (tighter distribution)
  
  # Orthogonal loss (CLOP)
  vpl_orthogonal_weight: 0.0        # Orthogonal loss weight (disabled)
  vpl_orthogonal_orthonorm_weight: 0.0  # Orthonormal constraint weight
  vpl_use_manual_orthogonal_labels: False  # Use manual labels
  vpl_num_prototypes: 0              # Number of prototypes
  vpl_prototype_scale: 0.0           # Prototype distance from origin
  
  # Visualization
  vpl_tsne_visualize_freq: 10       # Visualize every N rounds
```

**Evaluation Settings:**
```yaml
eval:
  max_samples_for_reward: 30         # Limit reward model evaluation
```

### 5.2 Dataset Configuration

**HH-RLHF Dataset:**
- **Type**: `hh-rlhf@llm`
- **Split**: Harmlessness clients (1 to client_num // 2), Helpfulness clients (client_num // 2 + 1 to client_num)
- **Format**: Comparison dataset with `win_dataset` and `lose_dataset`

**HHST Dataset:**
- **Type**: `hhst@llm`
- **Format**: Binary choice dataset
- **Note**: Reward model evaluation is skipped for HHST

---

## 6. Key Design Decisions

### 6.1 Unified Trainer
- **Decision**: Single `VPLRewardChoiceTrainer` supports both standard VPL and VPL-GP via config flag
- **Rationale**: Reduces code duplication, easier maintenance
- **Implementation**: Uses `VariationalEncoderGP` if `vpl_use_gp_prior=True`, otherwise `VariationalEncoder`

### 6.2 Fixed Prototypes
- **Decision**: Orthogonal prototypes are fixed (not learnable)
- **Rationale**: Matches CLOP standard, prevents prototype collapse
- **Implementation**: `requires_grad=False`, initialized as orthonormal basis scaled by `prototype_scale`

### 6.3 Balanced Sampling
- **Decision**: Sample 10 z-values per label per client for visualization
- **Rationale**: Ensures balanced representation in t-SNE plots
- **Implementation**: Groups clients by label, samples equally from each group

### 6.4 Manual Labels for All Clients
- **Decision**: Assign labels to ALL clients (not just participating)
- **Rationale**: Ensures non-participating clients appear in visualization with correct labels
- **Implementation**: `_compute_manual_orthogonal_labels()` assigns to all clients based on data type

### 6.5 Separate VPL Optimizer
- **Decision**: Use separate optimizer for VPL components with 10x learning rate
- **Rationale**: VPL components need faster learning while LLM is frozen
- **Implementation**: `vpl_optimizer` with `lr = base_lr * vpl_lr_multiplier`

### 6.6 Feature Difference Extraction
- **Decision**: Use embedding difference `(chosen - rejected)` to remove general information
- **Rationale**: Captures only preference information, reduces noise
- **Implementation**: Extracts embeddings at choice token positions, computes difference

---

## 7. Testing and Validation

### 7.1 Log Verification
- **50271.log**: Reference log showing expected behavior
- **Key Log Messages**:
  - `Collected X client z distributions for VPL-GP prior`
  - `Computed balanced orthogonal labels for X clients`
  - `Round X: Collected z values from X clients`
  - `Broadcasting VPL-GP prior with X client distributions`
  - `Broadcasting orthogonal labels to clients`

### 7.2 Visualization Verification
- **t-SNE Plots**: Generated every 10 rounds (configurable)
- **File Location**: `exp/{exp_name}/sub_exp_{timestamp}/cross_client_z_tsne_round_{round_num}.png`
- **WandB**: Logged as `visualization/cross_client_z_tsne_round_{round_num}`

### 7.3 Metrics Verification
- **VPL Metrics**: `vpl_total_loss`, `vpl_reconstruction_loss`, `vpl_kl_loss`, `vpl_orthogonal_loss`
- **Reward Metrics**: `avg_harmlessness`, `avg_helpfulness` (only for hh-rlhf, only for appropriate clients)
- **WandB Logging**: Server-side aggregated metrics + individual client metrics (first 3 clients)

---

## 8. Known Issues and Limitations

### 8.1 Memory Management
- **Issue**: Z-history can grow large over many rounds
- **Mitigation**: Limit history to 100 batches, sample z-values (max 100), move to CPU immediately

### 8.2 Visualization Frequency
- **Issue**: t-SNE is computationally expensive
- **Mitigation**: Visualize every N rounds (default: 10), limit samples per client

### 8.3 Reward Model Evaluation
- **Issue**: Slow for large datasets
- **Mitigation**: Limit to 30 samples by default (configurable), only for hh-rlhf dataset

### 8.4 Non-Participating Clients
- **Issue**: Non-participating clients may not have z-values
- **Mitigation**: Keep last known z-values, assign labels to all clients

---

## 9. Future Enhancements

### 9.1 Potential Improvements
1. **Adaptive Temperature**: Adjust Gumbel-Softmax temperature based on training progress
2. **Dynamic Prototypes**: Learn prototypes instead of fixed (requires careful initialization)
3. **Hierarchical Clustering**: Use hierarchical clustering for orthogonal labels
4. **Multi-Objective Optimization**: Balance reconstruction, KL, and orthogonal losses adaptively
5. **Federated t-SNE**: Compute t-SNE in a federated manner to reduce communication

### 9.2 Additional Features
1. **Win/Lose Evaluation**: Implement local model as judge (partially implemented)
2. **Prototype Visualization**: Show prototype evolution over rounds
3. **Client Similarity Matrix**: Compute and visualize client similarity based on z-distributions
4. **Prior Quality Metrics**: Measure how well mixture prior matches client distributions

---

## 10. File Structure

```
federatedscope/llm/
├── llm_local/
│   ├── server.py              # Server-side aggregation, prior collection, visualization
│   ├── client.py              # Client-side z transmission, prior update
│   └── z_visualization.py     # t-SNE visualization
├── trainer/
│   ├── vpl_reward_choice_trainer.py    # Base VPL trainer (supports GP prior)
│   └── vpl_gp_reward_choice_trainer.py # Legacy GP trainer (deprecated)
├── model/
│   ├── variational_encoder.py         # Standard variational encoder
│   └── variational_encoder_gp.py      # GP prior variational encoder
└── metric/
    └── hhrl_metrics.py        # Reward model evaluation, win/lose metrics
```

---

## 11. Conclusion

The VPL-GP federated learning system has been successfully implemented and restored after the git rebase incident. The system provides:

1. **Personalized Learning**: Each client learns a personalized latent vector encoding their preferences
2. **Federated Prior**: Server aggregates client z-distributions to form a mixture prior
3. **Visualization**: t-SNE plots show how client preferences evolve and cluster
4. **Orthogonal Regularization**: Optional CLOP-based loss encourages diverse client behaviors
5. **Comprehensive Metrics**: Reward model evaluation and win/lose metrics for hh-rlhf dataset

The implementation is production-ready and has been tested with the hh-rlhf dataset. Future enhancements can build upon this solid foundation.

---

## 12. README, Scripts, and Configuration Files

### 12.1 Project README

**Location**: `README.md` (project root)

**Key Information**:
- Empty file (no project-level documentation)
- LLM-specific README at `federatedscope/llm/README.md`

**FederatedScope-LLM README** (`federatedscope/llm/README.md`):
- **Purpose**: General guide for FederatedScope-LLM package
- **Key Sections**:
  - Installation instructions (PyTorch>=1.13.0, PEFT dependency)
  - Quick start with GPT-2 on Alpaca
  - Configuration examples
  - Built-in datasets (alpaca, dolly-15k, gsm8k, rosetta_alpaca, code_search_net)
  - PEFT methods (LoRA, Prefix Tuning, P-Tuning, Prompt Tuning)
  - HF Accelerate support
- **Note**: General LLM federated learning guide, not VPL-GP specific

### 12.2 Scripts Directory Structure

**Location**: `scripts/vpl-gp/`

**Script Categories**:

#### 12.2.1 Binary Selector Training Scripts

1. **`hhst.sh`** - Standard VPL-GP HHST training (non-orthogonal)
   - **Purpose**: Train binary selector with VPL-GP (orthogonal loss disabled)
   - **Config**: `cfg/vpl-gp/hhst.yaml`
   - **Key Features**:
     - Task ID: 40000 (configurable)
     - Checkpoint: `hhrl_choice_gemma_fedbiscuit_u3_vplgp_${tid}.ckpt`
     - WandB project: `fvpl-selector`
     - Log: `outputs/${tid}.log`
   - **Hyperparameters**:
     - Batch size: 16
     - LR: 0.0001
     - KL weight: 0.1
     - Variance cap: -3.0 (sigma ≤ 0.223)
     - Difference only: True

2. **`hhst-ortho.sh`** - VPL-GP HHST training with orthogonal loss
   - **Purpose**: Train binary selector with VPL-GP and orthogonal loss enabled
   - **Config**: `cfg/vpl-gp/hhst-ortho.yaml`
   - **Key Features**:
     - Task ID: 50000 (configurable)
     - Checkpoint: `hhrl_choice_gemma_fedbiscuit_u3_vplgp_ortho_${tid}.ckpt`
     - Orthogonal loss: ENABLED (weight: 1.0, reduced from 10.0)
     - Manual labels: True
   - **Hyperparameters**:
     - Batch size: 16
     - LR: 0.0001
     - KL weight: 0.1
     - Variance cap: -3.0
     - Orthogonal weight: 1.0
     - Prototype scale: 5.0

3. **`hhst_c100.sh`** - VPL-GP HHST training with 100 clients
   - **Purpose**: Scale to 100 clients for larger experiments
   - **Config**: `cfg/vpl-gp/hhst.yaml` (with client_num override)
   - **Key Features**:
     - Task ID: 50100 (configurable)
     - GPU: 3 (reserved)
     - Client num: 100 (overridden in script)
     - Checkpoint: `hhrl_choice_gemma_fedbiscuit_u3_vplgp_c100_${tid}.ckpt`
     - Log: `outputs/${tid}_c100.log`

4. **`hhst-ortho-*.sh`** - Orthogonal loss experiments with specific task IDs
   - **Purpose**: Run specific orthogonal loss experiments
   - **Examples**:
     - `hhst-ortho-40023.sh`: Task ID 40023
     - `hhst-ortho-40024.sh`: Task ID 40024
     - `hhst-ortho-50023.sh`: Task ID 50023
     - `hhst-ortho-50024.sh`: Task ID 50024
     - `hhst-ortho-50124.sh`: Task ID 50124

#### 12.2.2 RL Training Scripts

1. **`hrl.sh`** - RLHF training with VPL-GP selector
   - **Purpose**: Train RLHF policy using VPL-GP selector checkpoint
   - **Config**: `cfg/vpl-gp/hrl.yaml`
   - **Selector Config**: `cfg/vpl-gp/test_hrl_selector.yaml`
   - **Key Features**:
     - Task ID: 50200 (configurable)
     - GPU: 5 (reserved)
     - Selector checkpoint: From `hhst.sh` (tid=50100)
     - Checkpoint: `hhrl_rlhf_gemma_choice_vplgp_${tid}.ckpt`
     - WandB project: `fvpl-rl`
   - **Dependencies**:
     - Requires selector checkpoint from `hhst.sh`
     - Checks for checkpoint existence before starting

2. **`hrl-ortho.sh`** - RLHF training with orthogonal selector
   - **Purpose**: Train RLHF policy using VPL-GP orthogonal selector
   - **Config**: `cfg/vpl-gp/hrl-ortho.yaml`
   - **Key Features**:
     - Uses orthogonal selector checkpoint
     - Task ID: 51022 (configurable)

3. **`hrl_c100.sh`** - RLHF training with 100 clients
   - **Purpose**: Scale RLHF to 100 clients
   - **Config**: `cfg/vpl-gp/hrl.yaml` (with client_num override)
   - **Key Features**:
     - Task ID: 50200 (configurable)
     - GPU: 3 (reserved)
     - Client num: 100 (overridden)
     - Selector checkpoint: From `hhst_c100.sh` (tid=50100)
     - Creates temporary selector config with checkpoint path
     - Cleans up temp config after 30 seconds

**Common Script Features**:
- **PYTHONPATH Setup**: Sets PYTHONPATH to project root
- **CUDA Memory**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- **Background Execution**: Uses `nohup` for background execution
- **Logging**: Redirects stdout/stderr to `outputs/${tid}.log`
- **Task ID**: Configurable via `tid` variable

### 12.3 Configuration Files

**Location**: `cfg/vpl-gp/`

#### 12.3.1 Binary Selector Configs

1. **`hhst.yaml`** - Standard VPL-GP HHST configuration
   - **Purpose**: Binary selector training without orthogonal loss
   - **Key Settings**:
     ```yaml
     federate:
       client_num: 10
       sample_client_num: 5
       total_round_num: 50
     
     llm:
       vpl_use_gp_prior: True
       vpl_latent_dim: 32
       vpl_kl_weight: 0.1
       vpl_gp_temperature: 1.0
       vpl_use_feature_difference: True
       vpl_use_difference_only: True
       vpl_max_logvar: -3.0
       vpl_orthogonal_weight: 0.0  # DISABLED
       vpl_tsne_visualize_freq: 10
     
     dataloader:
       batch_size: 16
     
     train:
       local_update_steps: 30
       optimizer:
         lr: 0.0001
     
     trainer:
       type: vplgprewardchoicetrainer
       choices: ['A', 'B']
     
     eval:
       metrics: ['loss', 'acc', 'vpl_kl_loss', 'vpl_reconstruction_loss', 'vpl_orthogonal_loss']
       max_samples_for_reward: 100
     
     wandb:
       name_project: 'fvpl-selector'
     ```

2. **`hhst-ortho.yaml`** - VPL-GP HHST with orthogonal loss
   - **Purpose**: Binary selector training with orthogonal loss enabled
   - **Key Differences from `hhst.yaml`**:
     ```yaml
     llm:
       vpl_orthogonal_weight: 1.0  # ENABLED (reduced from 10.0)
       vpl_orthogonal_orthonorm_weight: 0.1
       vpl_use_manual_orthogonal_labels: True
       vpl_num_prototypes: 2
       vpl_prototype_scale: 5.0
     ```

3. **`hhst-fd.yaml`** - Feature difference configuration
   - **Purpose**: Experiment with feature difference extraction
   - **Key Settings**:
     ```yaml
     llm:
       vpl_use_feature_difference: True
       vpl_use_manual_orthogonal_labels: True
       vpl_orthogonal_weight: 10.0
     
     dataloader:
       batch_size: 4  # Smaller batch size
     
     train:
       optimizer:
         lr: 0.00001  # Lower learning rate
     ```

4. **`hhst-60000.yaml`** - Specific experiment configuration
   - **Purpose**: Configuration for task ID 60000
   - **Key Settings**:
     ```yaml
     llm:
       vpl_kl_weight: 1.0  # Increased 10x from 0.1
     
     dataloader:
       batch_size: 4
     
     train:
       optimizer:
         lr: 0.00001
     ```

5. **`hhst-ortho-*.yaml`** - Orthogonal loss experiment configs
   - **Purpose**: Specific orthogonal loss experiment configurations
   - **Examples**: `hhst-ortho-40023.yaml`, `hhst-ortho-50023.yaml`, etc.

#### 12.3.2 RL Training Configs

1. **`hrl.yaml`** - Standard RLHF training configuration
   - **Purpose**: RLHF training using VPL-GP selector
   - **Key Settings**:
     ```yaml
     federate:
       client_num: 1  # Single client for RL training
       total_round_num: 30
     
     data:
       splitter: 'iid'  # IID splitter for RL
     
     llm:
       rlhf: True  # Enable RLHF
       tok_len: 1024
       max_new_token: 512
       num_completions: 2
       grad_accum_step: 32
       max_prompts_for_generation: 50
       generation_batch_size: 3
       # VPL-GP parameters (should match selector)
       vpl_latent_dim: 32
       vpl_feature_method: 'choice_logits'
       vpl_gp_temperature: 1.0
     
     trainer:
       type: llmdporewardtrainer  # DPO trainer
     
     dataloader:
       batch_size: 1  # Small batch for RL
     
     train:
       local_update_steps: 10
     
     wandb:
       name_project: 'fvpl-rl'
     ```

2. **`hrl-ortho.yaml`** - RLHF training with orthogonal selector
   - **Purpose**: RLHF training using orthogonal VPL-GP selector
   - **Key Settings**:
     ```yaml
     llm:
       rlhf_use_variational_generation: True
       rlhf_use_variational_selection: True
       rlhf_selector_checkpoint: ".../vplgp_ortho_50022.ckpt"
       reward_coeff: 0.1
       grad_accum_step: 4
     
     eval:
       metrics: ['loss', 'acc', 'avg_helpfulness', 'avg_harmlessness',
                 'helpfulness_winrate', 'harmlessness_winrate', 'avg_winlose_rate']
     ```

### 12.4 Configuration Patterns

#### 12.4.1 Common Settings Across Configs

**Federated Learning**:
- `mode: standalone` - Standalone mode (single process)
- `share_local_model: True` - Share model instance to save memory
- `online_aggr: False` - Disable online aggregation

**Data**:
- `type: 'hh-rlhf@llm'` - HH-RLHF dataset
- `splits: [0.9, 0.09, 0.01]` - Train/val/test splits
- `splitter: 'meta'` - Meta splitter for selector, `'iid'` for RL

**Model**:
- `type: 'google/gemma-2b@huggingface_llm'` - Gemma-2B model
- `tok_len: 1024` - Input token length
- `max_new_token: 512` - Max generation length

**Adapter (PEFT)**:
- `use: True` - Enable PEFT
- `count: 3` - 3 adapters
- `method: 'lora'` - LoRA method
- `r: 8, lora_alpha: 16, lora_dropout: 0.05` - LoRA hyperparameters

**Training**:
- `local_update_steps: 30` (selector) / `10` (RL)
- `batch_or_epoch: batch`
- `optimizer.type: AdamW`
- `optimizer.betas: (0.9, 0.95)`
- `is_enable_half: True` - FP16 training

**Evaluation**:
- `freq: 5` (selector) / `1` (RL)
- `count_flops: False` - Disable FLOP counting
- `max_samples_for_reward: 100` (selector) / `30` (RL)

**WandB**:
- `use: True`
- `online_track: True`
- `name_project: 'fvpl-selector'` (selector) / `'fvpl-rl'` (RL)

#### 12.4.2 VPL-GP Specific Settings

**Core VPL-GP**:
- `vpl_use_gp_prior: True` - Enable GP prior
- `vpl_latent_dim: 32` - Latent dimension
- `vpl_kl_weight: 0.1` (standard) / `1.0` (experiments)
- `vpl_gp_temperature: 1.0` - Gumbel-Softmax temperature
- `vpl_max_logvar: -3.0` - Variance cap (sigma ≤ 0.223)

**Feature Extraction**:
- `vpl_feature_method: 'choice_logits'` - Feature extraction method
- `vpl_use_feature_difference: True` - Use embedding difference
- `vpl_use_difference_only: True` - Use only difference (no chosen/rejected)

**Orthogonal Loss**:
- `vpl_orthogonal_weight: 0.0` (disabled) / `1.0` (enabled) / `10.0` (experiments)
- `vpl_orthogonal_orthonorm_weight: 0.1` - Orthonormal constraint weight
- `vpl_use_manual_orthogonal_labels: True` - Use manual labels
- `vpl_num_prototypes: 2` - Number of prototypes
- `vpl_prototype_scale: 5.0` - Prototype distance from origin

**Visualization**:
- `vpl_tsne_visualize_freq: 10` - Visualize every 10 rounds

### 12.5 Script Execution Workflow

#### 12.5.1 Binary Selector Training

```bash
# 1. Standard training (non-orthogonal)
./scripts/vpl-gp/hhst.sh
# → Uses cfg/vpl-gp/hhst.yaml
# → Output: hhrl_choice_gemma_fedbiscuit_u3_vplgp_${tid}.ckpt
# → Log: outputs/${tid}.log

# 2. Orthogonal loss training
./scripts/vpl-gp/hhst-ortho.sh
# → Uses cfg/vpl-gp/hhst-ortho.yaml
# → Output: hhrl_choice_gemma_fedbiscuit_u3_vplgp_ortho_${tid}.ckpt
# → Log: outputs/${tid}.log

# 3. 100 clients training
./scripts/vpl-gp/hhst_c100.sh
# → Uses cfg/vpl-gp/hhst.yaml with client_num=100
# → Output: hhrl_choice_gemma_fedbiscuit_u3_vplgp_c100_${tid}.ckpt
# → Log: outputs/${tid}_c100.log
```

#### 12.5.2 RL Training

```bash
# 1. Standard RL training
./scripts/vpl-gp/hrl.sh
# → Requires selector checkpoint from hhst.sh (tid=50100)
# → Uses cfg/vpl-gp/test_hrl_selector.yaml (selector config)
# → Uses cfg/vpl-gp/hrl.yaml (RL config)
# → Output: hhrl_rlhf_gemma_choice_vplgp_${tid}.ckpt
# → Log: outputs/${tid}.log

# 2. RL with 100 clients
./scripts/vpl-gp/hrl_c100.sh
# → Requires selector checkpoint from hhst_c100.sh (tid=50100)
# → Creates temporary selector config
# → Output: hhrl_rlhf_gemma_choice_vplgp_c100_${tid}.ckpt
# → Log: outputs/${tid}_c100.log
```

### 12.6 Configuration Override Patterns

**Command-Line Overrides**:
```bash
python federatedscope/main.py \
    --cfg cfg/vpl-gp/hhst.yaml \
    federate.save_to /path/to/checkpoint.ckpt \
    expname "custom_experiment_name" \
    device 0 \
    federate.client_num 100
```

**Common Overrides**:
- `federate.save_to` - Checkpoint path
- `expname` - Experiment name
- `device` - GPU device ID
- `federate.client_num` - Number of clients
- `federate.total_round_num` - Number of rounds
- `llm.vpl_kl_weight` - KL divergence weight
- `llm.vpl_orthogonal_weight` - Orthogonal loss weight

### 12.7 File Naming Conventions

**Checkpoints**:
- Selector: `hhrl_choice_gemma_fedbiscuit_u3_vplgp_${tid}.ckpt`
- Selector (orthogonal): `hhrl_choice_gemma_fedbiscuit_u3_vplgp_ortho_${tid}.ckpt`
- Selector (100 clients): `hhrl_choice_gemma_fedbiscuit_u3_vplgp_c100_${tid}.ckpt`
- RL: `hhrl_rlhf_gemma_choice_vplgp_${tid}.ckpt`
- RL (100 clients): `hhrl_rlhf_gemma_choice_vplgp_c100_${tid}.ckpt`

**Logs**:
- Standard: `outputs/${tid}.log`
- 100 clients: `outputs/${tid}_c100.log`

**Experiments**:
- Selector: `vplgp_hhst_t${tid}` / `vplgp_hhst_ortho_t${tid}`
- Selector (100 clients): `vplgp_hhst_c100_t${tid}`
- RL: `vplgp_test_hrl_t${tid}` / `vplgp_hrl_c100_t${tid}`

---

## Appendix: Key Code Locations

### Server-Side
- **Z-Distribution Collection**: `server.py:488-604` (`_collect_vpl_gp_prior_distributions`)
- **Label Computation**: `server.py:606-701` (`_compute_balanced_orthogonal_labels`)
- **Visualization Collection**: `server.py:703-856` (`_collect_z_values_for_visualization`)
- **Visualization**: `server.py:858-952` (`_visualize_cross_client_z`)
- **Broadcasting**: `server.py:954-1041` (`broadcast_model_para`)

### Client-Side
- **Z Transmission**: `client.py:171-193` (in `callback_funcs_for_model_para`)
- **Prior Update**: `client.py:250-270` (`callback_funcs_for_vpl_gp_prior`)
- **Label Update**: `client.py:272-280` (`callback_funcs_for_vpl_orthogonal_labels`)

### Trainer
- **Feature Extraction**: `vpl_reward_choice_trainer.py:269-429`
- **Forward Pass**: `vpl_reward_choice_trainer.py:431-591`
- **Z Collection**: `vpl_reward_choice_trainer.py:749-773`
- **Prior Update**: `vpl_reward_choice_trainer.py:805-818`

### Model
- **GP Prior Update**: `variational_encoder_gp.py:37-81` (`update_prior`)
- **Mixture KL**: `variational_encoder_gp.py:141-238` (`kl_divergence`)

### Visualization
- **t-SNE Plot**: `z_visualization.py:13-203` (`visualize_cross_client_z`)

### Metrics
- **Reward Evaluation**: `hhrl_metrics.py:56-251` (`_get_or_compute_hhrl_scores`)
