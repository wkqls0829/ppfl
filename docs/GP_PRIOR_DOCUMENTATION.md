# Gumbel-Softmax Prior (GP Prior) Documentation

## Overview

Gumbel-Softmax Prior (GP Prior) extends VPL by using a mixture prior instead of a standard normal prior. The prior for each client is a weighted combination of other clients' posterior distributions, enabling knowledge sharing across clients in federated learning.

**Reference**: VMTL paper (https://arxiv.org/abs/2111.05323)

## Core Concepts

### 1. Mixture Prior

Instead of using a standard normal prior `p(z) = N(0, I)`, GP Prior uses a mixture:

```
p_mixture(z) = Σ_i w_i * N(z; μ_i, σ_i²)
```

where:
- `μ_i, σ_i²`: Mean and variance from client `i`'s posterior
- `w_i`: Weight for client `i`'s distribution
- Sum over all other clients (excluding current client)

### 2. Gumbel-Softmax Sampling

To make the mixture selection differentiable, we use Gumbel-Softmax:

```python
# Sample from mixture using Gumbel-Softmax
gumbel_noise = -log(-log(U))  # U ~ Uniform(0,1)
logits = log(weights) + gumbel_noise / temperature
probs = softmax(logits / temperature)
selected_component = sample from mixture using probs
```

- **Temperature**: Controls sharpness of distribution
  - High temperature: More uniform, smoother gradients
  - Low temperature: More discrete, closer to true sampling

### 3. KL Divergence with Mixture Prior

The KL divergence becomes:

```
KL(q(z|x) || p_mixture(z)) = E_q[log q(z|x)] - E_q[log p_mixture(z)]
```

where `log p_mixture(z)` uses log-sum-exp trick for numerical stability:

```python
log_p_components = [log p_i(z) + log w_i for each component i]
log_p_max = max(log_p_components)
log_p_mixture = log_p_max + log(Σ exp(log_p_i - log_p_max))
```

## Implementation

### File Structure

```
federatedscope/llm/model/variational_encoder_gp.py  # GP Prior encoder
federatedscope/llm/trainer/vpl_reward_choice_trainer.py  # Trainer integration
```

### Key Components

#### 1. VariationalEncoderGP

Extends `VariationalEncoder` to support mixture prior:

```python
class VariationalEncoderGP(VariationalEncoder):
    def __init__(self, input_dim, latent_dim, hidden_dims, 
                 temperature, num_clients):
        super().__init__(input_dim, latent_dim, hidden_dims)
        self.temperature = temperature
        self.num_clients = num_clients
        
        # Prior components (updated from server)
        self.prior_mus = None      # (num_clients, latent_dim)
        self.prior_logvars = None  # (num_clients, latent_dim)
        self.prior_weights = None  # (num_clients,)
    
    def update_prior(self, client_mus, client_logvars, client_weights):
        """Update mixture prior from server"""
        self.prior_mus = client_mus
        self.prior_logvars = client_logvars
        self.prior_weights = client_weights
```

#### 2. Prior Update Flow

1. **Client Training**: Each client trains locally and computes `z` distribution
2. **Client Aggregation**: Server collects `(μ_i, log σ_i²)` from all clients
3. **Prior Update**: Server broadcasts mixture components to clients
4. **Next Round**: Clients use updated mixture prior for KL divergence

#### 3. KL Divergence Calculation

```python
def kl_divergence(self, mu, logvar, use_gumbel_prior=True):
    if self.prior_mus is None:
        # Fallback to standard normal prior
        return super().kl_divergence(mu, logvar)
    
    # Sample z from posterior
    z = self.reparameterize(mu, logvar)
    
    # Compute log q(z|x)
    log_q = -0.5 * Σ(log(2π) + logvar + (z - μ)²/exp(logvar))
    
    # Compute log p_mixture(z) using log-sum-exp
    log_p_components = []
    for i in range(num_components):
        log_p_i = log N(z; μ_i, σ_i²) + log w_i
        log_p_components.append(log_p_i)
    
    log_p_mixture = log_sum_exp(log_p_components)
    
    # KL divergence
    kl = (log_q - log_p_mixture).mean()
    return kl
```

## Configuration

### GP Prior Setup

```yaml
llm:
  vpl_use_gp_prior: True          # Enable GP prior
  vpl_gp_temperature: 1.0         # Gumbel-Softmax temperature
  vpl_latent_dim: 32              # Latent dimension
  vpl_kl_weight: 0.1              # KL weight

federate:
  client_num: 10                  # Number of clients

trainer:
  type: vplgprewardchoicetrainer  # Use GP prior trainer
```

### Hyperparameters

- **`vpl_gp_temperature`**: Gumbel-Softmax temperature (default: 1.0)
  - Higher values: Smoother gradients, more exploration
  - Lower values: Sharper distribution, closer to discrete sampling

- **`vpl_kl_weight`**: Weight for KL divergence (default: 0.1)
  - Balances reconstruction vs. prior regularization

## Training Process

### Round-based Training

1. **Round Start**: Clients receive updated mixture prior from server
2. **Local Training**: 
   - Extract features from local data
   - Encode to get `z, μ, log σ²`
   - Compute KL divergence against mixture prior
   - Update model parameters
3. **Round End**: 
   - Compute client's `z` distribution (mean over batches)
   - Send `(μ_client, log σ²_client)` to server
4. **Server Aggregation**:
   - Collect distributions from all clients
   - Compute mixture weights (uniform or based on similarity)
   - Broadcast updated prior to clients

### Mixture Weight Calculation

Currently uses uniform weights:

```python
weights = torch.ones(num_clients) / num_clients
```

Future improvements:
- Similarity-based weights (closer clients get higher weights)
- Learned weights via attention mechanism

## Advantages

1. **Knowledge Sharing**: Clients learn from each other's preferences
2. **Better Regularization**: Mixture prior provides richer structure
3. **Federated Learning**: Naturally fits federated setting
4. **Personalization**: Each client maintains own posterior while sharing prior

## Limitations

1. **Communication Overhead**: Need to send `z` distributions each round
2. **Cold Start**: Early rounds have limited prior information
3. **Heterogeneity**: May not work well if clients are very different
4. **Numerical Stability**: Log-sum-exp trick needed for mixture KL

## Comparison with Standard VPL

| Aspect | Standard VPL | VPL-GP |
|--------|-------------|--------|
| Prior | `N(0, I)` | Mixture of client distributions |
| KL Divergence | Simple closed form | Requires log-sum-exp |
| Knowledge Sharing | None | Across clients |
| Communication | Model parameters only | Model + z distributions |
| Regularization | Global prior | Personalized prior |

## References

- VMTL Paper: [Variational Multi-Task Learning](https://arxiv.org/abs/2111.05323)
- Implementation: `federatedscope/llm/model/variational_encoder_gp.py`
- Trainer: `federatedscope/llm/trainer/vpl_reward_choice_trainer.py`
