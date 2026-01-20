# KL Loss와 Orthogonal Loss의 Differentiability 분석

이 문서는 KL loss와 orthogonal loss를 minimize할 때 어떤 부분이 differentiable한지 상세히 설명합니다.

## 개요

VPL-GP에서 두 가지 주요 loss가 있습니다:
1. **KL Divergence Loss**: `KL(q(z|x) || p(z))` 또는 `KL(q(z|x) || p_mixture(z))`
2. **Orthogonal Loss**: CLOP 기반의 pull loss와 orthonormal constraint

각 loss의 differentiability를 분석하여 어떤 파라미터가 학습되는지 명확히 합니다.

---

## 1. KL Divergence Loss

### 1.1 표준 VPL (Standard Normal Prior)

#### 수식
```
KL(q(z|x) || N(0, I)) = -0.5 * Σ(1 + logvar - μ² - exp(logvar))
```

#### Differentiable 부분

**학습되는 파라미터**:
1. **Variational Encoder의 파라미터** (`self.variational_encoder.encoder`, `self.variational_encoder.fc_mu`, `self.variational_encoder.fc_logvar`)
   - `μ = encoder(x)` → `fc_mu(h)`
   - `logvar = encoder(x)` → `fc_logvar(h)`
   - **Gradient flow**: `KL loss` → `μ, logvar` → `encoder weights`

2. **Feature Extractor의 파라미터** (`self.feature_extractor`)
   - `x = feature_extractor(preference_features)`
   - **Gradient flow**: `KL loss` → `μ, logvar` → `x` → `feature_extractor weights`

#### 코드 위치
```python
# federatedscope/llm/model/variational_encoder.py
def kl_divergence(self, mu, logvar):
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return kl.mean()  # Differentiable w.r.t. mu and logvar
```

**Gradient path**:
```
KL loss → mu, logvar → encoder(x) → encoder weights
                      → feature_extractor(preference_features) → feature_extractor weights
```

### 1.2 VPL-GP (Mixture Prior)

#### 수식
```
KL(q(z|x) || p_mixture(z)) = E_q[log q(z|x) - log p_mixture(z)]

where:
  log q(z|x) = -0.5 * Σ(log(2π) + logvar + (z - μ)² / var)
  log p_mixture(z) = log(Σ_i w_i * N(z; μ_i, σ_i²))
                    = max(log_p_i) + log(Σ exp(log_p_i - max(log_p_i)))  [log-sum-exp trick]
```

#### Differentiable 부분

**학습되는 파라미터**:
1. **Variational Encoder의 파라미터** (동일)
   - `μ, logvar`는 여전히 encoder로부터 생성됨
   - **Gradient flow**: `KL loss` → `μ, logvar` → `encoder weights`

2. **Feature Extractor의 파라미터** (동일)
   - `x = feature_extractor(preference_features)`
   - **Gradient flow**: `KL loss` → `μ, logvar` → `x` → `feature_extractor weights`

**학습되지 않는 부분**:
- **Prior components** (`self.prior_mus`, `self.prior_logvars`, `self.prior_weights`)
  - 서버에서 수집된 다른 클라이언트들의 z-distribution
  - **`torch.no_grad()`로 고정** (서버에서 계산되어 브로드캐스트됨)
  - Gradient가 흐르지 않음

#### 코드 위치
```python
# federatedscope/llm/model/variational_encoder_gp.py
def kl_divergence(self, mu, logvar, use_gumbel_prior=True):
    # Sample z from q(z|x) (reparameterization trick)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)  # Random noise (not differentiable)
    z = mu + eps * std  # Differentiable w.r.t. mu and logvar
    
    # Compute log q(z|x) - Differentiable w.r.t. mu and logvar
    log_q = -0.5 * torch.sum(
        np.log(2 * np.pi) + logvar + (z - mu).pow(2) / torch.exp(logvar),
        dim=-1
    )
    
    # Compute log p_mixture(z) - Uses prior_mus, prior_logvars (NOT differentiable)
    log_p_components = []
    for i in range(num_components):
        mu_i = self.prior_mus[i]  # Fixed (from server)
        logvar_i = self.prior_logvars[i]  # Fixed (from server)
        weight_i = self.prior_weights[i]  # Fixed (from server)
        
        log_p_i = -0.5 * torch.sum(...)  # Uses fixed mu_i, logvar_i
        log_p_components.append(log_p_i)
    
    # Log-sum-exp trick (differentiable w.r.t. log_p_components, but they use fixed priors)
    log_p_mixture = log_p_max + torch.log(torch.sum(torch.exp(...), dim=0))
    
    # KL = E_q[log q - log p_mixture]
    kl = (log_q - log_p_mixture).mean()  # Differentiable w.r.t. mu, logvar only
    return kl
```

**Gradient path**:
```
KL loss → log_q (differentiable w.r.t. mu, logvar)
       → log_p_mixture (uses fixed prior_mus, prior_logvars - NO gradient)
       → mu, logvar → encoder(x) → encoder weights
                    → feature_extractor(preference_features) → feature_extractor weights
```

**핵심 포인트**:
- `z = μ + ε * σ` (reparameterization trick)로 인해 `z`는 `μ, logvar`에 대해 differentiable
- `log_q(z|x)`는 `μ, logvar`에 대해 differentiable
- `log_p_mixture(z)`는 `prior_mus, prior_logvars`를 사용하지만, 이들은 **고정값**이므로 gradient가 흐르지 않음
- 따라서 KL loss는 **posterior encoder만 학습**하고, **prior는 학습하지 않음**

---

## 2. Orthogonal Loss

### 2.1 수식

```
L_orthogonal = λ_pull * L_pull + λ_orthonorm * L_orthonorm

where:
  L_pull = ||z - prototype_label||²  (Pull loss)
  L_orthonorm = ||P^T P - I||²        (Orthonormal constraint)
```

### 2.2 Differentiable 부분

#### Pull Loss

**수식**:
```
L_pull = ||z - prototype_label||²
       = Σ_i (z_i - prototype_label[i])²
```

**학습되는 파라미터**:
1. **Variational Encoder의 파라미터**
   - `z = reparameterize(μ, logvar)` → `μ, logvar = encoder(x)`
   - **Gradient flow**: `L_pull` → `z` → `μ, logvar` → `encoder weights`

2. **Feature Extractor의 파라미터**
   - `x = feature_extractor(preference_features)`
   - **Gradient flow**: `L_pull` → `z` → `μ, logvar` → `x` → `feature_extractor weights`

3. **Orthogonal Prototypes** (`self.orthogonal_prototypes`)
   - `prototype_label = self.orthogonal_prototypes[label]`
   - **Gradient flow**: `L_pull` → `prototype_label` → `self.orthogonal_prototypes`
   - **핵심**: Prototypes도 학습됨!

#### Orthonormal Constraint

**수식**:
```
L_orthonorm = ||P^T P - I||²
            = Σ_i Σ_j (P^T P - I)_ij²
```

**학습되는 파라미터**:
1. **Orthogonal Prototypes** (`self.orthogonal_prototypes`)
   - `P = self.orthogonal_prototypes` (shape: `(num_prototypes, latent_dim)`)
   - `P^T P`는 `P`에 대해 differentiable
   - **Gradient flow**: `L_orthonorm` → `P^T P` → `self.orthogonal_prototypes`

**학습되지 않는 부분**:
- **Orthogonal Label** (`self.orthogonal_label`)
  - 서버에서 할당된 label (정수)
  - Gradient가 흐르지 않음

### 2.3 코드 위치

```python
# federatedscope/llm/trainer/vpl_reward_choice_trainer.py
def _compute_clop_orthogonal_loss(self, z, labels=None):
    # Get orthogonal label (from server, NOT differentiable)
    orthogonal_label = self.orthogonal_label  # Integer label
    
    # Get prototype for this label (DIFFERENTIABLE - prototypes are learnable)
    prototype = self.orthogonal_prototypes[orthogonal_label]  # (latent_dim,)
    
    # Pull loss: ||z - prototype||² (DIFFERENTIABLE w.r.t. z and prototype)
    pull_loss = torch.norm(z - prototype, dim=-1).pow(2).mean()
    
    # Orthonormal constraint: ||P^T P - I||² (DIFFERENTIABLE w.r.t. P)
    P = self.orthogonal_prototypes  # (num_prototypes, latent_dim)
    PTP = torch.matmul(P, P.T)  # (num_prototypes, num_prototypes)
    I = torch.eye(P.shape[0], device=P.device)
    orthonorm_loss = torch.norm(PTP - I, p='fro').pow(2)
    
    # Total loss
    orthogonal_loss = self.vpl_orthogonal_weight * pull_loss + \
                     self.vpl_orthogonal_orthonorm_weight * orthonorm_loss
    
    return orthogonal_loss, pull_loss, orthonorm_loss
```

**Gradient path**:
```
Orthogonal Loss → pull_loss → z (differentiable) → μ, logvar → encoder weights
                              → prototype (differentiable) → orthogonal_prototypes
              → orthonorm_loss → P^T P → orthogonal_prototypes
```

**QR Decomposition 주의사항**:
```python
# QR decomposition은 gradient를 차단함 (with torch.no_grad())
with torch.no_grad():
    Q, R = torch.linalg.qr(self.orthogonal_prototypes.T)
    self.orthogonal_prototypes.data = Q.T * prototype_scale
```
- QR decomposition은 **gradient를 차단**하는 연산
- 하지만 loss 계산 시에는 **원래 `orthogonal_prototypes`를 사용**하므로 gradient가 흐름
- QR은 **정규화 목적**으로만 사용 (orthonormality 유지)

---

## 3. 전체 Gradient Flow

### 3.1 Forward Pass

```
Input (preference_features)
  ↓
Feature Extractor (learnable)
  ↓
x (extracted features)
  ↓
Variational Encoder (learnable)
  ↓
μ, logvar (posterior parameters)
  ↓
Reparameterization: z = μ + ε * σ
  ↓
z (latent vector)
  ↓
Latent Projection (learnable)
  ↓
latent_adjustment
  ↓
Conditioned Logits = logits + latent_adjustment
  ↓
Reconstruction Loss
```

### 3.2 Backward Pass (Gradient Flow)

#### KL Loss Gradient
```
KL Loss
  ↓
μ, logvar (differentiable)
  ↓
Variational Encoder weights (learnable)
  ↓
x (differentiable)
  ↓
Feature Extractor weights (learnable)
```

#### Orthogonal Loss Gradient
```
Orthogonal Loss
  ↓
├─→ Pull Loss
│   ├─→ z (differentiable)
│   │   └─→ μ, logvar → Encoder weights (learnable)
│   │   └─→ x → Feature Extractor weights (learnable)
│   └─→ prototype (differentiable)
│       └─→ orthogonal_prototypes (learnable)
│
└─→ Orthonorm Loss
    └─→ P^T P (differentiable)
        └─→ orthogonal_prototypes (learnable)
```

### 3.3 학습되는 파라미터 요약

| Loss | 학습되는 파라미터 | 학습되지 않는 부분 |
|------|------------------|-------------------|
| **KL Loss** | • Variational Encoder weights<br>• Feature Extractor weights | • Prior components (prior_mus, prior_logvars, prior_weights)<br>• Random noise (ε) |
| **Pull Loss** | • Variational Encoder weights<br>• Feature Extractor weights<br>• Orthogonal Prototypes | • Orthogonal Label (정수) |
| **Orthonorm Loss** | • Orthogonal Prototypes | - |

---

## 4. 핵심 포인트

### 4.1 Reparameterization Trick

```python
z = μ + ε * σ  # where ε ~ N(0, 1)
```

- `ε`는 **random noise**이므로 gradient가 흐르지 않음
- 하지만 `z`는 `μ, logvar`에 대해 **differentiable**
- 이로 인해 KL loss와 orthogonal loss 모두 `z`를 통해 `μ, logvar`로 gradient가 흐름

### 4.2 Prior Components는 고정

VPL-GP에서:
- `prior_mus`, `prior_logvars`, `prior_weights`는 **서버에서 계산**되어 브로드캐스트됨
- 클라이언트에서는 **`torch.no_grad()`로 고정**
- KL loss는 **posterior encoder만 학습**하고, prior는 학습하지 않음

### 4.3 Orthogonal Prototypes는 학습됨

- `self.orthogonal_prototypes`는 **`nn.Parameter`**로 정의되어 학습됨
- Pull loss와 orthonorm loss 모두 prototypes를 업데이트
- QR decomposition은 **정규화 목적**으로만 사용 (gradient 차단)

### 4.4 Orthogonal Label은 고정

- `self.orthogonal_label`은 **서버에서 할당**된 정수 label
- Gradient가 흐르지 않음
- 하지만 **해당 label의 prototype**은 학습됨

---

## 5. 실제 학습 과정

### 5.1 한 번의 Forward-Backward Pass

```python
# Forward
z, mu, logvar = variational_encoder(extracted_features)
kl_loss = variational_encoder.kl_divergence(mu, logvar)
orthogonal_loss, pull_loss, orthonorm_loss = _compute_clop_orthogonal_loss(z)
total_loss = reconstruction_loss + kl_weight * kl_loss + orthogonal_loss

# Backward
total_loss.backward()  # Gradient 계산

# Gradient가 흐르는 경로:
# 1. reconstruction_loss → latent_projection → z → mu, logvar → encoder → feature_extractor
# 2. kl_loss → mu, logvar → encoder → feature_extractor
# 3. pull_loss → z → mu, logvar → encoder → feature_extractor
#              → prototype → orthogonal_prototypes
# 4. orthonorm_loss → P^T P → orthogonal_prototypes
```

### 5.2 Optimizer Update

```python
optimizer.step()  # 다음 파라미터들이 업데이트됨:
# - feature_extractor.parameters()
# - variational_encoder.parameters()
# - latent_projection.parameters()
# - orthogonal_prototypes (if orthogonal loss enabled)
```

---

## 6. 요약

### Differentiable 부분

1. **KL Loss**:
   - ✅ Variational Encoder weights
   - ✅ Feature Extractor weights
   - ❌ Prior components (고정)

2. **Orthogonal Loss**:
   - ✅ Variational Encoder weights (pull loss를 통해)
   - ✅ Feature Extractor weights (pull loss를 통해)
   - ✅ Orthogonal Prototypes (pull loss + orthonorm loss)
   - ❌ Orthogonal Label (고정)

### 핵심 메시지

- **KL loss**는 **posterior encoder**를 학습하여 `q(z|x)`를 prior에 가깝게 만듦
- **Orthogonal loss**는 **z embedding**과 **prototypes**를 동시에 학습하여 preference space에서 클라이언트 그룹을 분리
- **Prior components**는 서버에서 계산되어 고정되므로, 클라이언트에서는 학습되지 않음
- **Reparameterization trick**으로 인해 sampling 과정도 differentiable하여 gradient가 흐름
