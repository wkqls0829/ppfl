# Mixture Prior p_mixture(z) 계산 방식

## 개요

VPL-GP에서 사용하는 mixture prior는 여러 클라이언트들의 z 분포를 가중 평균한 것입니다:

```
p_mixture(z) = Σ_i w_i * N(z; μ_i, σ_i²)
```

여기서:
- `w_i`: 클라이언트 i의 가중치 (Σ w_i = 1)
- `μ_i`: 클라이언트 i의 z 분포 평균
- `σ_i²`: 클라이언트 i의 z 분포 분산
- `N(z; μ_i, σ_i²)`: 정규 분포

## 1. Prior 업데이트 (`update_prior`)

서버에서 다른 클라이언트들의 z 분포를 받아서 mixture prior를 구성합니다:

```python
def update_prior(self, client_mus, client_logvars, client_weights):
    """
    Args:
        client_mus: (num_clients, latent_dim) - 각 클라이언트의 μ
        client_logvars: (num_clients, latent_dim) - 각 클라이언트의 log(σ²)
        client_weights: (num_clients,) - 각 클라이언트의 가중치
    """
    # 가중치 정규화 (합이 1이 되도록)
    self.prior_weights = client_weights / client_weights.sum()
    
    # 저장
    self.prior_mus = client_mus      # (num_clients, latent_dim)
    self.prior_logvars = client_logvars  # (num_clients, latent_dim)
```

**예시:**
```
클라이언트 0: μ₀ = [0.5, 0.3, ...], σ₀² = [0.1, 0.2, ...], w₀ = 0.3
클라이언트 1: μ₁ = [0.2, 0.4, ...], σ₁² = [0.15, 0.25, ...], w₁ = 0.4
클라이언트 2: μ₂ = [0.8, 0.1, ...], σ₂² = [0.2, 0.1, ...], w₂ = 0.3

p_mixture(z) = 0.3 * N(z; μ₀, σ₀²) + 0.4 * N(z; μ₁, σ₁²) + 0.3 * N(z; μ₂, σ₂²)
```

## 2. KL Divergence 계산 (`kl_divergence`)

KL divergence `KL(q(z|x) || p_mixture(z))`를 계산할 때 mixture prior를 사용합니다.

### 2.1. Posterior q(z|x) 계산

```python
# q(z|x)에서 z 샘플링
std = torch.exp(0.5 * logvar)  # σ = exp(0.5 * log(σ²))
eps = torch.randn_like(std)    # ε ~ N(0, 1)
z = mu + eps * std             # z ~ N(μ, σ²)

# log q(z|x) 계산
log_q = -0.5 * Σ[log(2π) + logvar + (z - mu)² / exp(logvar)]
```

### 2.2. Mixture Prior p_mixture(z) 계산

각 클라이언트의 정규 분포를 계산하고 가중 합을 구합니다:

```python
# 각 클라이언트 i에 대해
for i in range(num_clients):
    μ_i = self.prior_mus[i]           # (latent_dim,)
    logvar_i = self.prior_logvars[i]   # (latent_dim,)
    w_i = self.prior_weights[i]       # scalar
    
    # log N(z; μ_i, σ_i²) 계산
    log_p_i = -0.5 * Σ[log(2π) + logvar_i + (z - μ_i)² / exp(logvar_i)]
    
    # 가중치 추가
    log_p_i = log_p_i + log(w_i)
    
    log_p_components.append(log_p_i)
```

### 2.3. Log-Sum-Exp Trick

수치 안정성을 위해 log-sum-exp trick을 사용합니다:

```python
# log p_mixture(z) = log(Σ_i w_i * N(z; μ_i, σ_i²))
#                  = log(Σ_i exp(log(w_i) + log N(z; μ_i, σ_i²)))

log_p_stack = torch.stack(log_p_components, dim=0)  # (num_clients, batch_size)

# Log-sum-exp trick: log(Σ exp(a_i)) = max(a_i) + log(Σ exp(a_i - max(a_i)))
log_p_max = torch.max(log_p_stack, dim=0, keepdim=True)[0]  # (1, batch_size)

log_p_mixture = log_p_max.squeeze(0) + torch.log(
    torch.sum(torch.exp(log_p_stack - log_p_max), dim=0) + 1e-8
)  # (batch_size,)
```

**수학적 설명:**
```
log(Σ_i exp(a_i)) = max_i(a_i) + log(Σ_i exp(a_i - max_i(a_i)))
```

이렇게 하면:
- `exp(a_i - max_i(a_i))`가 0~1 사이로 제한되어 overflow 방지
- `max_i(a_i)`를 빼고 더해서 정확한 값 유지

### 2.4. KL Divergence 최종 계산

```python
# KL = E_q[log q(z|x) - log p_mixture(z)]
kl = (log_q - log_p_mixture).mean()
```

## 3. 전체 계산 과정 예시

### 입력
```python
# 현재 클라이언트의 posterior
mu = [0.4, 0.3, ...]      # (batch_size, latent_dim)
logvar = [-1.0, -0.5, ...]  # (batch_size, latent_dim)

# 다른 클라이언트들의 분포 (서버에서 받음)
prior_mus = [
    [0.5, 0.3, ...],  # 클라이언트 0
    [0.2, 0.4, ...],  # 클라이언트 1
    [0.8, 0.1, ...]   # 클라이언트 2
]  # (3, latent_dim)

prior_logvars = [
    [-1.1, -0.6, ...],  # 클라이언트 0
    [-1.2, -0.7, ...],  # 클라이언트 1
    [-0.9, -0.8, ...]    # 클라이언트 2
]  # (3, latent_dim)

prior_weights = [0.3, 0.4, 0.3]  # (3,)
```

### Step 1: z 샘플링
```python
std = exp(0.5 * logvar) = [0.606, 0.779, ...]
eps ~ N(0, 1)
z = mu + eps * std = [0.4 + 0.606*ε₁, 0.3 + 0.779*ε₂, ...]
```

### Step 2: log q(z|x) 계산
```python
log_q = -0.5 * Σ[log(2π) + logvar + (z - mu)² / exp(logvar)]
     = -0.5 * Σ[0.693 + (-1.0) + (0.606*ε₁)² / 0.368, ...]
     ≈ -2.5  # 예시 값
```

### Step 3: 각 클라이언트의 log p_i 계산
```python
# 클라이언트 0
log_p_0 = -0.5 * Σ[log(2π) + logvar_0 + (z - μ₀)² / exp(logvar_0)]
        + log(0.3)
        ≈ -2.8 + (-1.2) = -4.0

# 클라이언트 1
log_p_1 = -0.5 * Σ[log(2π) + logvar_1 + (z - μ₁)² / exp(logvar_1)]
        + log(0.4)
        ≈ -2.6 + (-0.9) = -3.5

# 클라이언트 2
log_p_2 = -0.5 * Σ[log(2π) + logvar_2 + (z - μ₂)² / exp(logvar_2)]
        + log(0.3)
        ≈ -3.0 + (-1.2) = -4.2
```

### Step 4: Log-Sum-Exp
```python
log_p_stack = [[-4.0], [-3.5], [-4.2]]  # (3, batch_size)
log_p_max = -3.5

log_p_mixture = -3.5 + log(exp(-4.0 - (-3.5)) + exp(-3.5 - (-3.5)) + exp(-4.2 - (-3.5)))
              = -3.5 + log(exp(-0.5) + exp(0) + exp(-0.7))
              = -3.5 + log(0.606 + 1.0 + 0.497)
              = -3.5 + log(2.103)
              = -3.5 + 0.743
              = -2.757
```

### Step 5: KL Divergence
```python
kl = (log_q - log_p_mixture).mean()
   = (-2.5 - (-2.757)).mean()
   = 0.257
```

## 4. 표준 VPL vs VPL-GP 비교

### 표준 VPL
```python
# Prior: p(z) = N(0, I)
log_p = -0.5 * Σ[log(2π) + 0 + z² / 1]
      = -0.5 * Σ[log(2π) + z²]

kl = (log_q - log_p).mean()
```

### VPL-GP
```python
# Prior: p_mixture(z) = Σ_i w_i * N(z; μ_i, σ_i²)
log_p_mixture = log(Σ_i w_i * exp(log N(z; μ_i, σ_i²)))
              = log_sum_exp([log(w_i) + log N(z; μ_i, σ_i²) for i])

kl = (log_q - log_p_mixture).mean()
```

## 5. Gumbel-Softmax (선택적)

샘플링 시 Gumbel-Softmax를 사용하여 mixture component를 선택할 수 있습니다:

```python
def sample_prior(self, batch_size, use_gumbel=True):
    # Gumbel noise 생성
    gumbel_noise = -log(-log(U + ε) + ε)  # U ~ Uniform(0,1)
    
    # Gumbel-Softmax로 component 선택
    log_weights = log(w_i)
    gumbel_logits = (log_weights + gumbel_noise) / temperature
    component_probs = softmax(gumbel_logits)  # (batch_size, num_components)
    
    # 각 component에서 샘플링하고 가중 평균
    z_samples = Σ_i component_probs[i] * sample_from_N(μ_i, σ_i²)
```

하지만 KL divergence 계산에서는 Gumbel-Softmax를 사용하지 않고 직접 mixture를 계산합니다.

## 6. 핵심 포인트

1. **Mixture Prior**: 여러 클라이언트의 분포를 가중 평균
2. **Log-Sum-Exp**: 수치 안정성을 위한 trick
3. **KL Divergence**: `E_q[log q(z|x) - log p_mixture(z)]`
4. **가중치**: 각 클라이언트의 중요도 (정규화되어 합이 1)

## 7. 코드 위치

- **Prior 업데이트**: `federatedscope/llm/model/variational_encoder_gp.py::update_prior`
- **KL 계산**: `federatedscope/llm/model/variational_encoder_gp.py::kl_divergence`
- **샘플링**: `federatedscope/llm/model/variational_encoder_gp.py::sample_prior`
