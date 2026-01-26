# KL Loss Forward and Backward Pass: VPL with GP Prior

## Overview

This document explains in detail how the KL divergence loss is computed in the forward pass and how gradients flow in the backward pass for VPL (Variational Preference Learning) with GP (Gumbel-Softmax Prior).

## Mathematical Background

### Standard VPL KL Divergence

In standard VPL, the KL divergence is:
```
KL(q(z|x) || N(0, I)) = KL(q(z|x) || p_standard(z))
```

where:
- `q(z|x)` is the posterior distribution (variational encoder output)
- `p_standard(z) = N(0, I)` is the standard normal prior

### VPL-GP KL Divergence

In VPL-GP, we use a **mixture prior** from other clients:
```
p_mixture(z) = Σ_i w_i * N(z; μ_i, σ_i²)
```

The KL divergence becomes:
```
KL(q(z|x) || p_mixture(z)) = E_q(z|x)[log q(z|x) - log p_mixture(z)]
```

where:
- `q(z|x) = N(z; μ, σ²)` is the posterior from the variational encoder
- `p_mixture(z) = Σ_i w_i * N(z; μ_i, σ_i²)` is the mixture prior from other clients
- `w_i` are mixture weights (typically uniform: `w_i = 1/num_clients`)

---

## Forward Pass

### Step 1: Feature Extraction

```python
# Input: preference data (choices A and B)
# Output: extracted features

features = feature_extractor(preference_data)  # (batch_size, feature_dim)
```

**Gradient Flow**: `feature_extractor` parameters receive gradients ✓

### Step 2: Variational Encoding

```python
# Pass features through variational encoder
mu, logvar = variational_encoder(features)
# mu: (batch_size, latent_dim)
# logvar: (batch_size, latent_dim)
```

**Gradient Flow**: 
- `variational_encoder.parameters()` receive gradients ✓
- `mu` and `logvar` are differentiable w.r.t. encoder parameters ✓

### Step 3: Sample z from Posterior

```python
# Reparameterization trick: z = μ + ε * σ
std = exp(0.5 * logvar)  # σ = exp(0.5 * logvar)
eps = randn_like(std)    # ε ~ N(0, 1) (random noise, no gradient)
z = mu + eps * std       # z ~ q(z|x)
# z: (batch_size, latent_dim)
```

**Key Points**:
- `eps` is **NOT differentiable** (random noise)
- `z` is **differentiable** w.r.t. `mu` and `logvar` through the reparameterization trick
- `z` maintains gradients back to `variational_encoder` parameters ✓

**Gradient Flow**:
```
∂z/∂μ = 1
∂z/∂logvar = ε * exp(0.5 * logvar) * 0.5 = ε * σ * 0.5
```

### Step 4: Compute log q(z|x)

```python
# log q(z|x) = -0.5 * Σ[log(2π) + logvar + (z - μ)² / var]

log_q = -0.5 * sum(
    log(2π) + logvar + (z - mu).pow(2) / exp(logvar),
    dim=-1
)  # (batch_size,)
```

**Gradient Flow**:
- `log_q` is differentiable w.r.t. `mu`, `logvar`, and `z` ✓
- Since `z` is differentiable, gradients flow back to encoder parameters ✓

**Derivatives**:
```
∂log_q/∂μ = (z - μ) / var
∂log_q/∂logvar = -0.5 + 0.5 * (z - μ)² / var
∂log_q/∂z = -(z - μ) / var
```

### Step 5: Compute log p_mixture(z) (Log-Sum-Exp Trick)

```python
# For each component i in the mixture:
log_p_components = []
for i in range(num_components):
    mu_i = prior_mus[i]      # (latent_dim,) - NO gradient (fixed prior)
    logvar_i = prior_logvars[i]  # (latent_dim,) - NO gradient (fixed prior)
    weight_i = prior_weights[i]  # scalar - NO gradient (fixed weight)
    
    # log N(z; μ_i, σ_i²) = -0.5 * Σ[log(2π) + logvar_i + (z - μ_i)² / var_i]
    log_p_i = -0.5 * sum(
        log(2π) + logvar_i + (z - mu_i).pow(2) / exp(logvar_i),
        dim=-1
    )  # (batch_size,)
    
    # Add log weight
    log_p_i = log_p_i + log(weight_i + 1e-8)
    log_p_components.append(log_p_i)

# Stack: (num_components, batch_size)
log_p_stack = stack(log_p_components, dim=0)

# Log-Sum-Exp Trick for numerical stability:
log_p_max = max(log_p_stack, dim=0, keepdim=True)[0]  # (1, batch_size)
log_p_mixture = log_p_max + log(
    sum(exp(log_p_stack - log_p_max), dim=0) + 1e-8
)  # (batch_size,)
```

**Key Points**:
- `prior_mus`, `prior_logvars`, `prior_weights` are **NOT differentiable** (fixed from other clients)
- `log_p_mixture` is **differentiable** w.r.t. `z` only ✓
- Since `z` is differentiable, gradients flow back through `z` to encoder parameters ✓

**Gradient Flow**:
- `log_p_mixture` receives gradients from KL loss
- Gradients flow through `z` back to `mu` and `logvar`
- **NO gradient flows to prior parameters** (they are fixed)

**Derivatives** (w.r.t. z):
```
∂log_p_mixture/∂z = Σ_i [softmax_i * ∂log_p_i/∂z]
where softmax_i = exp(log_p_i - log_p_max) / Σ_j exp(log_p_j - log_p_max)
```

### Step 6: Compute KL Divergence

```python
# KL = E_q(z|x)[log q(z|x) - log p_mixture(z)]
kl = (log_q - log_p_mixture).mean()  # scalar
```

**Gradient Flow**:
- `kl` is differentiable w.r.t. `log_q` and `log_p_mixture` ✓
- Gradients flow back through:
  1. `log_q` → `mu`, `logvar` → `variational_encoder`
  2. `log_p_mixture` → `z` → `mu`, `logvar` → `variational_encoder`

---

## Backward Pass

### Gradient Flow Path

When `loss.backward()` is called, gradients flow as follows:

```
Total Loss
  ↓
KL Loss (scaled by vpl_kl_weight)
  ↓
┌─────────────────────────────────┐
│  ∂KL/∂log_q                     │
│  → ∂log_q/∂μ, ∂log_q/∂logvar    │
│  → ∂μ/∂encoder, ∂logvar/∂encoder│
│  → encoder.parameters()         │
└─────────────────────────────────┘
  ↓
┌─────────────────────────────────┐
│  ∂KL/∂log_p_mixture             │
│  → ∂log_p_mixture/∂z            │
│  → ∂z/∂μ, ∂z/∂logvar            │
│  → ∂μ/∂encoder, ∂logvar/∂encoder│
│  → encoder.parameters()         │
└─────────────────────────────────┘
```

### Detailed Gradient Computations

#### 1. Gradient from log_q

```python
∂KL/∂log_q = 1/batch_size  # (scalar)

∂log_q/∂μ = (z - μ) / var  # (batch_size, latent_dim)
∂log_q/∂logvar = -0.5 + 0.5 * (z - μ)² / var  # (batch_size, latent_dim)

∂KL/∂μ = (∂KL/∂log_q) * (∂log_q/∂μ) = (z - μ) / (batch_size * var)
∂KL/∂logvar = (∂KL/∂log_q) * (∂log_q/∂logvar) = [-0.5 + 0.5 * (z - μ)² / var] / batch_size
```

#### 2. Gradient from log_p_mixture

```python
∂KL/∂log_p_mixture = -1/batch_size  # (scalar, negative because of subtraction)

# Log-sum-exp derivative (softmax-weighted sum)
∂log_p_mixture/∂z = Σ_i [softmax_i * ∂log_p_i/∂z]
                   = Σ_i [softmax_i * (μ_i - z) / var_i]  # (batch_size, latent_dim)

∂KL/∂z = (∂KL/∂log_p_mixture) * (∂log_p_mixture/∂z)
       = -1/batch_size * Σ_i [softmax_i * (μ_i - z) / var_i]

# Backprop through reparameterization
∂z/∂μ = 1
∂z/∂logvar = ε * σ * 0.5

∂KL/∂μ += (∂KL/∂z) * (∂z/∂μ) = (∂KL/∂z) * 1
∂KL/∂logvar += (∂KL/∂z) * (∂z/∂logvar) = (∂KL/∂z) * ε * σ * 0.5
```

#### 3. Final Gradients to Encoder Parameters

```python
# Gradients flow from μ and logvar to encoder parameters
∂KL/∂encoder_params = (∂KL/∂μ) * (∂μ/∂encoder_params) 
                    + (∂KL/∂logvar) * (∂logvar/∂encoder_params)
```

---

## Important Properties

### 1. Prior Parameters are Fixed

**Key Point**: The mixture prior parameters (`prior_mus`, `prior_logvars`, `prior_weights`) are **NOT trainable** in the forward/backward pass. They are:
- Fixed from other clients' distributions (updated only by server)
- Used only for computing `log_p_mixture(z)`
- Do **NOT** receive gradients

**Why?**
- The prior represents what **other clients** have learned
- It serves as a **regularization** term, not a learnable component
- Only the **posterior** (`q(z|x)`) is optimized to match this prior

### 2. Reparameterization Trick Enables Gradient Flow

**Without reparameterization**:
```python
z = sample_from_normal(mu, logvar)  # NO gradient flow through sampling
```

**With reparameterization**:
```python
z = mu + eps * exp(0.5 * logvar)  # Gradient flows through mu and logvar
```

This allows gradients to flow from `KL(z)` back to encoder parameters.

### 3. Log-Sum-Exp for Numerical Stability

**Problem**: Computing `log(Σ_i exp(log_p_i))` directly can cause numerical overflow/underflow.

**Solution**: Log-sum-exp trick:
```python
log(Σ_i exp(a_i)) = max(a_i) + log(Σ_i exp(a_i - max(a_i)))
```

This is numerically stable because:
- `exp(a_i - max(a_i))` is bounded between `exp(0) = 1` and `exp(-∞) = 0`
- No overflow/underflow issues

### 4. KL Loss Minimization Effect

When KL loss is minimized:
- `q(z|x)` becomes closer to `p_mixture(z)`
- The posterior `q(z|x)` learns to match the mixture prior from other clients
- This encourages **clustering** of similar clients' preferences

**Interpretation**:
- If client A has similar preferences to client B, their `z` distributions should be similar
- The mixture prior `p_mixture(z)` encodes this similarity
- Minimizing KL makes the current client's `z` align with similar clients' `z`

---

## Code Implementation Details

### Forward Pass (in `kl_divergence` method)

```python
def kl_divergence(self, mu, logvar, use_gumbel_prior=True):
    # 1. Sample z from posterior (reparameterization)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)  # Random, no gradient
    z = mu + eps * std  # Differentiable
    
    # 2. Compute log q(z|x)
    log_q = -0.5 * torch.sum(
        np.log(2 * np.pi) + logvar + (z - mu).pow(2) / torch.exp(logvar),
        dim=-1
    )
    
    # 3. Compute log p_mixture(z) using log-sum-exp
    log_p_components = []
    for i in range(num_components):
        mu_i = self.prior_mus[i]  # Fixed, no gradient
        logvar_i = self.prior_logvars[i]  # Fixed, no gradient
        weight_i = self.prior_weights[i]  # Fixed, no gradient
        
        log_p_i = -0.5 * torch.sum(
            np.log(2 * np.pi) + logvar_i + (z - mu_i).pow(2) / torch.exp(logvar_i),
            dim=-1
        )
        log_p_i = log_p_i + torch.log(weight_i + 1e-8)
        log_p_components.append(log_p_i)
    
    log_p_stack = torch.stack(log_p_components, dim=0)
    log_p_max = torch.max(log_p_stack, dim=0, keepdim=True)[0]
    log_p_mixture = log_p_max.squeeze(0) + torch.log(
        torch.sum(torch.exp(log_p_stack - log_p_max), dim=0) + 1e-8
    )
    
    # 4. Compute KL divergence
    kl = (log_q - log_p_mixture).mean()
    
    return kl
```

### Backward Pass (Automatic)

PyTorch's autograd automatically computes gradients:
```python
# Forward pass
kl_loss = kl_divergence(mu, logvar)
total_loss = reconstruction_loss + vpl_kl_weight * kl_loss

# Backward pass (automatic)
total_loss.backward()

# Gradients are computed for:
# - variational_encoder.parameters() ✓
# - feature_extractor.parameters() ✓
# - latent_projection.parameters() ✓
# - prior_mus, prior_logvars, prior_weights ✗ (fixed)
```

---

## Summary

### Forward Pass
1. Extract features → encode to `μ, logvar` → sample `z` via reparameterization
2. Compute `log q(z|x)` (differentiable w.r.t. encoder)
3. Compute `log p_mixture(z)` using log-sum-exp (differentiable w.r.t. `z` only)
4. Compute `KL = mean(log_q - log_p_mixture)`

### Backward Pass
1. Gradients flow from KL loss → `log_q` and `log_p_mixture`
2. From `log_q`: directly to `μ, logvar` → encoder parameters
3. From `log_p_mixture`: through `z` → `μ, logvar` → encoder parameters
4. **NO gradients** to prior parameters (they are fixed)

### Key Insight
The KL loss acts as a **regularization term** that pulls the posterior `q(z|x)` toward the mixture prior `p_mixture(z)` from other clients, enabling federated clustering of preferences while maintaining differentiability.
