# Orthogonal Loss (CLOP) Documentation

## Overview

Orthogonal Loss is based on the CLOP (Contrastive Learning with Orthonormal Prototypes) framework, designed to prevent neural collapse in contrastive learning. In the context of VPL-GP, it encourages embeddings to form orthogonal linear subspaces, improving representation diversity and preventing collapse.

**Reference**: [CLOP Paper](https://arxiv.org/pdf/2403.18699)

## Core Concepts

### 1. Neural Collapse Problem

Neural collapse is a phenomenon where:
- Embeddings converge to a lower-dimensional space
- Class representations become linearly dependent
- Model loses expressiveness and generalization

### 2. Orthonormal Prototypes

CLOP uses a set of orthonormal prototypes `P = [p_1, p_2, ..., p_k]` where:
- `p_i^T p_j = 0` for `i ≠ j` (orthogonal)
- `||p_i|| = 1` for all `i` (normalized)
- `P^T P = I` (orthonormal matrix)

### 3. Loss Components

The orthogonal loss consists of two parts:

#### Pull Loss
Pulls embeddings towards their assigned prototypes:

```
L_pull = (1/|B|) Σ ||z(x) - p_y||²
```

where:
- `z(x)`: Embedding for sample `x`
- `p_y`: Prototype assigned to sample `x`
- `y`: Orthogonal label (from server or auto-assigned)

#### Orthonormal Constraint
Enforces prototypes to remain orthonormal:

```
L_orthonorm = ||P^T P - I||²_F
```

where `||·||_F` is the Frobenius norm.

### 4. Total Orthogonal Loss

```
L_orthogonal = λ_pull * L_pull + λ_orthonorm * L_orthonorm
```

## Implementation

### File Structure

```
federatedscope/llm/trainer/vpl_reward_choice_trainer.py  # Loss computation
federatedscope/llm/llm_local/server.py                   # Label assignment
```

### Key Components

#### 1. Prototype Initialization

```python
# Initialize orthonormal prototypes
num_prototypes = num_clients  # or configurable
self.orthogonal_prototypes = nn.Parameter(
    torch.randn(num_prototypes, latent_dim) * 2.0
)

# Orthonormalize using QR decomposition
Q, R = torch.linalg.qr(self.orthogonal_prototypes.T)
self.orthogonal_prototypes.data = Q.T
```

#### 2. Label Assignment

**Option 1: Manual Labels (from Server)**
- Server performs balanced k-means clustering on client `z` means
- Assigns orthogonal labels to clients
- Broadcasts labels to clients

**Option 2: Auto Assignment**
- Compute similarity: `similarities = z @ prototypes.T`
- Assign to closest prototype: `label = argmax(similarities)`

#### 3. Loss Computation

```python
def _compute_clop_orthogonal_loss(self, z, labels=None):
    # Orthonormalize prototypes (QR decomposition)
    Q, R = torch.linalg.qr(self.orthogonal_prototypes.T)
    self.orthogonal_prototypes.data = Q.T
    
    # Assign labels
    if self.vpl_use_manual_orthogonal_labels:
        orthogonal_labels = self.orthogonal_label  # From server
    else:
        similarities = z @ self.orthogonal_prototypes.T
        orthogonal_labels = torch.argmax(similarities, dim=1)
    
    # Pull loss
    selected_prototypes = self.orthogonal_prototypes[orthogonal_labels]
    pull_loss = torch.mean((z - selected_prototypes) ** 2)
    
    # Orthonormal constraint
    PTP = self.orthogonal_prototypes @ self.orthogonal_prototypes.T
    identity = torch.eye(num_prototypes, device=z.device)
    orthonorm_loss = torch.norm(PTP - identity, p='fro') ** 2
    
    # Total loss
    orthogonal_loss = (self.vpl_orthogonal_weight * pull_loss + 
                      self.vpl_orthogonal_orthonorm_weight * orthonorm_loss)
    
    return orthogonal_loss, pull_loss, orthonorm_loss
```

#### 4. Integration with VPL Loss

```python
# In _hook_on_batch_forward
vpl_loss = reconstruction_loss + vpl_kl_weight * kl_loss

# Add orthogonal loss
if self.vpl_orthogonal_weight > 0.0:
    orthogonal_loss, _, _ = self._compute_clop_orthogonal_loss(z)
    vpl_loss = vpl_loss + orthogonal_loss
```

## Configuration

### Orthogonal Loss Setup

```yaml
llm:
  vpl_orthogonal_weight: 10.0              # Pull loss weight
  vpl_orthogonal_orthonorm_weight: 0.1     # Orthonormal constraint weight
  vpl_use_manual_orthogonal_labels: True  # Use server-computed labels
  vpl_num_prototypes: 10                   # Number of prototypes (optional)
```

### Hyperparameters

- **`vpl_orthogonal_weight`**: Weight for pull loss (default: 10.0)
  - Higher values: Stronger pull towards prototypes
  - Lower values: More flexible embeddings

- **`vpl_orthogonal_orthonorm_weight`**: Weight for orthonormal constraint (default: 0.1)
  - Higher values: Stricter orthonormality
  - Lower values: More flexible prototype structure

- **`vpl_use_manual_orthogonal_labels`**: Use server labels (default: True)
  - `True`: Server computes balanced labels via k-means
  - `False`: Auto-assign to closest prototype

## Training Process

### Server-side Label Assignment

1. **Collect Client Distributions**: Gather `z` means from all clients
2. **Clustering**: Perform balanced k-means clustering
3. **Label Assignment**: Assign orthogonal labels to clients
4. **Broadcast**: Send labels to clients

### Client-side Training

1. **Receive Label**: Get orthogonal label from server
2. **Forward Pass**: 
   - Compute embeddings `z`
   - Compute orthogonal loss
   - Add to total loss
3. **Backward Pass**: Update both model and prototypes

### Prototype Maintenance

Prototypes are orthonormalized each forward pass:

```python
# QR decomposition ensures orthonormality
Q, R = torch.linalg.qr(self.orthogonal_prototypes.T)
self.orthogonal_prototypes.data = Q.T
```

## Advantages

1. **Prevents Neural Collapse**: Maintains full-rank embedding space
2. **Better Separation**: Orthogonal subspaces improve class separation
3. **Interpretable**: Prototypes provide interpretable structure
4. **Scalable**: Works with any number of prototypes

## Limitations

1. **Additional Parameters**: Prototypes add `num_prototypes * latent_dim` parameters
2. **Hyperparameter Sensitivity**: Requires tuning of loss weights
3. **Communication**: Server needs to compute and broadcast labels
4. **Computational Overhead**: QR decomposition each forward pass

## Comparison with Standard VPL

| Aspect | Standard VPL | VPL + Orthogonal Loss |
|--------|-------------|----------------------|
| Embedding Space | May collapse | Full-rank, orthogonal |
| Regularization | KL divergence only | KL + Orthogonal |
| Prototypes | None | Learnable orthonormal |
| Label Assignment | N/A | Server or auto |
| Representation | Flexible | Structured |

## Use Cases

1. **Multi-client Federated Learning**: Prevent collapse across clients
2. **Few-shot Learning**: Better generalization with structured embeddings
3. **Interpretable Representations**: Prototypes provide interpretability
4. **Diverse Embeddings**: Maintain diversity in latent space

## References

- CLOP Paper: [Preventing Collapse in Contrastive Learning with Orthonormal Prototypes](https://arxiv.org/pdf/2403.18699)
- Implementation: `federatedscope/llm/trainer/vpl_reward_choice_trainer.py`
- Server Logic: `federatedscope/llm/llm_local/server.py`
