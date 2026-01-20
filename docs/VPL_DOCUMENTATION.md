# Variational Preference Learning (VPL) Documentation

## Overview

Variational Preference Learning (VPL) is a framework for learning from human feedback, specifically binary preference labels. It models user-specific preferences by inferring latent variables that capture individual differences in preference patterns.

**Reference**: [Variational Preference Learning](https://github.com/WEIRDLabUW/vpl)

## Core Concepts

### 1. Latent Variable Model

VPL assumes that each user has a latent preference vector `z` that captures their individual preferences:

- **Posterior**: `q_ψ(z | prefs)` - Encoder that infers latent `z` from preference data
- **Prior**: `p(z)` - Prior distribution over latents (typically standard normal)
- **Likelihood**: `p(y | z, x)` - Choice probability conditioned on latent `z` and input `x`

### 2. Evidence Lower Bound (ELBO)

The training objective is to maximize the ELBO:

```
ELBO = E_q(z|x)[log p(y|z,x)] - KL(q(z|x) || p(z))
     = Reconstruction Loss - KL Divergence
```

- **Reconstruction Loss**: Measures how well the model predicts choices given the inferred latent
- **KL Divergence**: Regularizes the posterior to stay close to the prior

## Implementation

### File Structure

```
federatedscope/llm/trainer/vpl_reward_choice_trainer.py  # Main trainer
federatedscope/llm/model/variational_encoder.py          # Encoder network
```

### Key Components

#### 1. Variational Encoder

The encoder `q_ψ(z | x)` maps preference features to a latent distribution:

```python
class VariationalEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims):
        # Encoder network: x -> [μ, log σ²]
        self.encoder = nn.Sequential(...)
    
    def forward(self, x):
        # Returns: μ, log σ²
        mu, logvar = self.encoder(x)
        # Reparameterization trick: z = μ + σ * ε, ε ~ N(0,1)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return z, mu, logvar
```

#### 2. Feature Extraction

VPL extracts features from preference data to feed into the encoder:

**Option 1: Choice Logits**
- Extract logits for choice tokens (A/B)
- Concatenate logits from both choices
- Input: `[logit_A_chosen, logit_B_chosen, logit_A_rejected, logit_B_rejected]`
- Config: `vpl_feature_method: 'choice_logits'`, `vpl_use_feature_difference: False`

**Option 2: Embedding Difference (Full)**
- Extract embeddings for chosen and rejected responses
- Compute difference: `chosen_emb - rejected_emb`
- Input: `[chosen_emb, rejected_emb, chosen_emb - rejected_emb]` (3 * embedding_dim)
- Config: `vpl_use_feature_difference: True`, `vpl_use_difference_only: False`
- **Note**: Includes general information from chosen/rejected embeddings

**Option 3: Embedding Difference (Difference Only)** ⭐ Recommended
- Extract embeddings for chosen and rejected responses
- Compute difference: `chosen_emb - rejected_emb`
- Input: `chosen_emb - rejected_emb` only (embedding_dim)
- Config: `vpl_use_feature_difference: True`, `vpl_use_difference_only: True`
- **Advantage**: Removes general information, keeps only preference signal
- **Use case**: When you want z to capture only preference differences, not response-specific information

#### 3. Latent Conditioning

The inferred latent `z` is used to condition the model's predictions:

```python
# Project latent to choice logit adjustments
latent_adjustment = self.latent_projection(z)  # (batch, num_choices)

# Add to logits
conditioned_logits = original_logits + latent_adjustment
```

## Configuration

### Basic VPL Setup

```yaml
llm:
  vpl_latent_dim: 32              # Dimension of latent space
  vpl_kl_weight: 0.1              # Weight for KL divergence
  vpl_feature_method: 'choice_logits'  # or 'embedding_difference'
  vpl_use_feature_difference: True    # Use embedding difference
  vpl_use_difference_only: True       # Use only difference (removes general info)
  vpl_use_llm_feature_extractor: True  # Use MLP feature extractor

trainer:
  type: vplrewardchoicetrainer
  choices: ['A', 'B']
```

### Hyperparameters

- **`vpl_latent_dim`**: Dimension of latent space (default: 32)
  - Larger values: More expressive, but harder to regularize
  - Smaller values: Better regularization, but less expressive

- **`vpl_kl_weight`**: Weight for KL divergence (default: 0.1)
  - Higher values: Stronger regularization, posterior closer to prior
  - Lower values: More flexible posterior, but may overfit

- **`vpl_feature_method`**: Feature extraction method
  - `'choice_logits'`: Use logits at choice token positions
  - `'embedding_difference'`: Use embedding difference (recommended)

- **`vpl_use_feature_difference`**: Use embedding difference (default: False)
  - `True`: Extract `chosen_emb - rejected_emb` from hidden states
  - `False`: Use logits-based features

- **`vpl_use_difference_only`**: Use only difference embedding (default: False)
  - `True`: Input to feature extractor is `difference` only (removes general info)
  - `False`: Input is `[chosen, rejected, difference]` (includes general info)
  - **Recommended**: `True` for preference-only learning

- **`vpl_use_llm_feature_extractor`**: Use MLP feature extractor (default: True)
  - `True`: Use deeper MLP network (512 → 256 → 128)
  - `False`: Use simpler feature extraction

## Training Process

### Forward Pass

1. **Extract Features**: From preference data (logits or embeddings)
2. **Encode**: `z, μ, log σ² = encoder(features)`
3. **Sample**: `z ~ q(z|x)` using reparameterization trick
4. **Condition**: Adjust model logits based on `z`
5. **Compute Loss**: Reconstruction + KL divergence

### Loss Components

```python
# Reconstruction loss (cross-entropy)
reconstruction_loss = CrossEntropyLoss(conditioned_logits, labels)

# KL divergence
kl_loss = KL(q(z|x) || p(z))
         = -0.5 * Σ(1 + log σ² - μ² - σ²)

# Total loss
vpl_loss = reconstruction_loss + vpl_kl_weight * kl_loss
```

## Evaluation Metrics

- **`vpl_kl_loss`**: Average KL divergence (measures regularization)
- **`vpl_reconstruction_loss`**: Average reconstruction loss (measures fit)
- **`loss`**: Total VPL loss
- **`acc`**: Choice prediction accuracy

## Use Cases

1. **Personalized Preference Learning**: Learn user-specific preferences
2. **Preference Disentanglement**: Separate general from personal preferences
3. **Few-shot Learning**: Leverage latent structure for better generalization

## Advantages

- **Interpretable**: Latent `z` captures user preferences
- **Regularized**: KL divergence prevents overfitting
- **Flexible**: Can condition on various input features
- **Scalable**: Works with large language models via adapters

## Limitations

- **Computational Overhead**: Additional encoder network and sampling
- **Hyperparameter Sensitivity**: Requires tuning KL weight
- **Feature Quality**: Depends on quality of extracted features

## References

- Original VPL Paper: [Variational Preference Learning](https://github.com/WEIRDLabUW/vpl)
- Implementation: `federatedscope/llm/trainer/vpl_reward_choice_trainer.py`
