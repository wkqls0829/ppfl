# Gumbel-Softmax Differentiability in VPL-GP

## 개요

VMTL 논문 (Variational Multi-Task Learning with Gumbel-Softmax Priors)을 참고하여 VPL-GP에서 Gumbel-Softmax를 사용한 mixture prior의 differentiable한 부분을 설명합니다.

## VMTL 논문의 핵심 아이디어

### 1. Mixture Prior 구성

```
p_t(z) = Σ_{k≠t} α_{t,k} * q_k(z | D_k)
```

여기서:
- `q_k(z | D_k)`: Task k의 variational posterior
- `α_{t,k}`: Task t가 task k의 posterior를 얼마나 사용할지 결정하는 가중치
- `α_{t,k}`는 **Gumbel-Softmax로 학습**됨 (differentiable!)

### 2. Gumbel-Softmax의 Differentiability

Gumbel-Softmax는 discrete categorical 선택을 continuous relaxation으로 근사하여 미분 가능하게 만듭니다:

```python
# Discrete selection (non-differentiable)
α = one_hot(argmax(log_weights + gumbel_noise))

# Gumbel-Softmax relaxation (differentiable)
α = softmax((log_weights + gumbel_noise) / temperature)
```

**핵심**: Temperature가 낮을수록 discrete에 가깝지만, temperature > 0이면 항상 differentiable합니다.

## 현재 VPL-GP 구현 분석

### 1. KL Divergence 계산 (현재 구현)

```python
def kl_divergence(self, mu, logvar, use_gumbel_prior=True):
    # z 샘플링
    z = mu + eps * std  # (batch_size, latent_dim)
    
    # log q(z|x) 계산
    log_q = -0.5 * Σ[log(2π) + logvar + (z - mu)² / exp(logvar)]
    
    # log p_mixture(z) = log(Σ_i w_i * N(z; μ_i, σ_i²))
    log_p_components = []
    for i in range(num_components):
        log_p_i = log N(z; μ_i, σ_i²) + log(w_i)
        log_p_components.append(log_p_i)
    
    # Log-sum-exp trick
    log_p_mixture = log_sum_exp(log_p_components)
    
    # KL = E_q[log q(z|x) - log p_mixture(z)]
    kl = (log_q - log_p_mixture).mean()
```

**Differentiable한 부분:**
- ✅ `log_q`: `mu`, `logvar`에 대해 미분 가능
- ✅ `log_p_i`: `μ_i`, `logvar_i`, `w_i`에 대해 미분 가능
- ✅ `log_p_mixture`: log-sum-exp는 미분 가능
- ✅ `kl`: 모든 파라미터에 대해 미분 가능

### 2. Sample Prior (현재 구현)

```python
def sample_prior(self, batch_size, use_gumbel=True):
    if use_gumbel:
        # Gumbel-Softmax로 component 선택
        log_weights = log(w_i)
        gumbel_noise = -log(-log(U + ε) + ε)
        gumbel_logits = (log_weights + gumbel_noise) / temperature
        component_probs = softmax(gumbel_logits)  # Differentiable!
        
        # 각 component에서 샘플링하고 가중 평균
        z_samples = Σ_i component_probs[i] * sample_from_N(μ_i, σ_i²)
    else:
        # 단순 가중 평균
        z_samples = Σ_i w_i * sample_from_N(μ_i, σ_i²)
```

**Differentiable한 부분:**
- ✅ `component_probs`: Gumbel-Softmax는 미분 가능
- ✅ `z_samples`: `component_probs`, `μ_i`, `σ_i`에 대해 미분 가능

**문제점:**
- `sample_prior`는 KL divergence 계산에서 사용되지 않음
- KL divergence는 직접 log-sum-exp로 계산하므로 Gumbel-Softmax의 이점을 활용하지 못함

## VMTL 방식과의 차이점

### VMTL 방식

```python
# 1. Mixture weights를 Gumbel-Softmax로 학습
α_logits = α_network(x)  # Learnable network
α = gumbel_softmax(α_logits, temperature=τ)

# 2. Prior를 mixture로 구성
p_mixture(z) = Σ_k α_k * q_k(z | D_k)

# 3. KL divergence 계산
kl = KL(q_t(z|x) || p_mixture(z))
    = E_q[log q_t(z|x) - log(Σ_k α_k * q_k(z|D_k))]

# 4. Backpropagation
# - α_logits → α → p_mixture → kl (모두 differentiable!)
```

**핵심**: Mixture weights `α`가 **학습 가능한 파라미터**입니다!

### 현재 VPL-GP 방식

```python
# 1. Mixture weights는 서버에서 받은 고정값
w_i = client_weights  # Fixed (not learnable)

# 2. Prior를 mixture로 구성
p_mixture(z) = Σ_i w_i * N(z; μ_i, σ_i²)

# 3. KL divergence 계산
kl = KL(q(z|x) || p_mixture(z))
    = E_q[log q(z|x) - log(Σ_i w_i * N(z; μ_i, σ_i²))]

# 4. Backpropagation
# - w_i는 고정이므로 gradient 없음
# - μ_i, σ_i는 다른 클라이언트의 것이므로 gradient 없음
# - 오직 q(z|x)의 μ, logvar만 학습됨
```

**차이점**: 
- VMTL: Mixture weights를 학습 (α_network)
- VPL-GP: Mixture weights는 고정 (서버에서 받음)

## 개선 방향: Learnable Mixture Weights

VMTL 방식을 참고하여 mixture weights를 학습 가능하게 만들 수 있습니다:

```python
class VariationalEncoderGP(VariationalEncoder):
    def __init__(self, ...):
        super().__init__(...)
        
        # Learnable mixture weight network (VMTL 방식)
        self.alpha_network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_clients)  # 각 클라이언트에 대한 logit
        )
        
        # Fixed prior distributions (서버에서 받음)
        self.prior_mus = None
        self.prior_logvars = None
    
    def compute_mixture_weights(self, x):
        """
        VMTL 방식: Input x로부터 mixture weights를 학습
        """
        alpha_logits = self.alpha_network(x)  # (batch_size, num_clients)
        alpha = F.gumbel_softmax(alpha_logits, tau=self.temperature, hard=False)
        return alpha  # (batch_size, num_clients)
    
    def kl_divergence(self, mu, logvar, x=None):
        """
        Learnable mixture weights를 사용한 KL divergence
        """
        # z 샘플링
        z = mu + eps * std
        
        # log q(z|x) 계산
        log_q = -0.5 * Σ[log(2π) + logvar + (z - mu)² / exp(logvar)]
        
        # Learnable mixture weights 계산
        if x is not None:
            alpha = self.compute_mixture_weights(x)  # (batch_size, num_clients)
        else:
            alpha = self.prior_weights.unsqueeze(0)  # Fallback
        
        # log p_mixture(z) = log(Σ_i α_i * N(z; μ_i, σ_i²))
        log_p_components = []
        for i in range(num_clients):
            log_p_i = log N(z; μ_i, σ_i²)
            # Batch-wise weights
            alpha_i = alpha[:, i:i+1]  # (batch_size, 1)
            log_p_i = log_p_i + log(alpha_i + 1e-8)
            log_p_components.append(log_p_i)
        
        # Log-sum-exp
        log_p_mixture = log_sum_exp(log_p_components)
        
        # KL divergence
        kl = (log_q - log_p_mixture).mean()
        
        return kl
```

**장점:**
- ✅ Mixture weights가 학습 가능 (α_network)
- ✅ 각 샘플마다 다른 mixture weights 사용 가능
- ✅ Gumbel-Softmax로 discrete selection 근사
- ✅ 모든 부분이 differentiable

## Differentiable 경로 요약

### 현재 구현 (Fixed Weights)

```
Input x
  ↓
Feature Extractor (differentiable)
  ↓
Variational Encoder (differentiable)
  ↓
μ, logvar (learnable)
  ↓
z = μ + ε * σ (reparameterization, differentiable)
  ↓
KL = E_q[log q(z|x) - log(Σ_i w_i * N(z; μ_i, σ_i²))]
  ↓
Backprop: μ, logvar만 업데이트 (w_i는 고정)
```

### 개선된 구현 (Learnable Weights, VMTL 방식)

```
Input x
  ↓
Feature Extractor (differentiable)
  ↓
Variational Encoder (differentiable)
  ↓
μ, logvar (learnable)
  ↓
α_network(x) → α_logits (learnable)
  ↓
Gumbel-Softmax(α_logits) → α (differentiable)
  ↓
z = μ + ε * σ (reparameterization, differentiable)
  ↓
KL = E_q[log q(z|x) - log(Σ_i α_i * N(z; μ_i, σ_i²))]
  ↓
Backprop: μ, logvar, α_network 모두 업데이트
```

## 핵심 포인트

1. **Gumbel-Softmax**: Discrete selection을 continuous relaxation으로 근사 → differentiable
2. **Log-Sum-Exp**: Mixture prior의 log 확률 계산 → differentiable
3. **Reparameterization Trick**: z 샘플링 → differentiable
4. **Learnable Weights**: VMTL 방식으로 mixture weights를 학습 가능하게 → 더 유연한 학습

## 참고

- VMTL 논문: [arXiv:2111.05323](https://arxiv.org/abs/2111.05323)
- VMTL GitHub: [autumn9999/VMTL](https://github.com/autumn9999/VMTL)
- Gumbel-Softmax: Jang et al., "Categorical Reparameterization with Gumbel-Softmax", ICLR 2017
