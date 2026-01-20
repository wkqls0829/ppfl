# KL Loss Forward/Backward Pass 분석

## 문제: KL Loss가 Minimize되지 않음

KL loss가 minimize되지 않는 원인을 분석하기 위해 forward/backward pass를 자세히 살펴봅니다.

---

## 1. KL Divergence 수식

### Standard VPL (Fixed Prior)
```
KL(q(z|x) || p(z)) where p(z) = N(0, I)

KL = -0.5 * Σ[1 + logvar - mu² - exp(logvar)]
```

### VPL-GP (Mixture Prior)
```
KL(q(z|x) || p_mixture(z)) where p_mixture(z) = Σ_i w_i * N(z; μ_i, σ_i²)

KL = E_q[log q(z|x) - log p_mixture(z)]
   = E_q[log q(z|x)] - E_q[log(Σ_i w_i * N(z; μ_i, σ_i²))]
```

---

## 2. Forward Pass 구성

### Step 1: Feature Extraction
```python
# Input: logits, labels, hidden_states
preference_features = self._extract_preference_features(...)
# Output: (batch_size, feature_dim)

# Deep feature extraction
extracted_features = self.feature_extractor(preference_features)
# Output: (batch_size, 128)
```

### Step 2: Variational Encoding
```python
# Encode to latent parameters
z, mu, logvar = self.variational_encoder(extracted_features)
# mu: (batch_size, latent_dim)
# logvar: (batch_size, latent_dim)
# z: (batch_size, latent_dim) - sampled using reparameterization trick
```

**Reparameterization Trick**:
```python
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)  # Random noise (NOT differentiable)
z = mu + eps * std  # Differentiable w.r.t. mu and logvar
```

### Step 3: KL Divergence 계산

#### Standard VPL
```python
def kl_divergence(self, mu, logvar):
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return kl.mean()
```

**Gradient Flow**:
- `mu`: ✅ Differentiable (encoder output)
- `logvar`: ✅ Differentiable (encoder output)
- `kl`: ✅ Differentiable w.r.t. mu and logvar

#### VPL-GP (Mixture Prior)
```python
def kl_divergence(self, mu, logvar, use_gumbel_prior=True):
    # Sample z from q(z|x)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std  # (batch_size, latent_dim)
    
    # Compute log q(z|x)
    log_q = -0.5 * torch.sum(
        np.log(2 * np.pi) + logvar + (z - mu).pow(2) / torch.exp(logvar),
        dim=-1
    )  # (batch_size,)
    
    # Compute log p_mixture(z) using log-sum-exp
    log_p_components = []
    for i in range(num_components):
        mu_i = self.prior_mus[i]  # FIXED (from server)
        logvar_i = self.prior_logvars[i]  # FIXED (from server)
        weight_i = self.prior_weights[i]  # FIXED (from server)
        
        log_p_i = -0.5 * torch.sum(
            np.log(2 * np.pi) + logvar_i + (z - mu_i.unsqueeze(0)).pow(2) / torch.exp(logvar_i.unsqueeze(0)),
            dim=-1
        ) + torch.log(weight_i + 1e-8)
        log_p_components.append(log_p_i)
    
    # Log-sum-exp trick
    log_p_stack = torch.stack(log_p_components, dim=0)
    log_p_max = torch.max(log_p_stack, dim=0, keepdim=True)[0]
    log_p_mixture = log_p_max.squeeze(0) + torch.log(
        torch.sum(torch.exp(log_p_stack - log_p_max), dim=0) + 1e-8
    )
    
    # KL = E_q[log q(z|x) - log p_mixture(z)]
    kl = (log_q - log_p_mixture).mean()
    return kl
```

**Gradient Flow**:
- `mu`: ✅ Differentiable (encoder output)
- `logvar`: ✅ Differentiable (encoder output)
- `z`: ✅ Differentiable w.r.t. mu and logvar (reparameterization trick)
- `log_q`: ✅ Differentiable w.r.t. mu and logvar
- `log_p_mixture`: ✅ Differentiable w.r.t. z (and thus mu, logvar)
- `kl`: ✅ Differentiable w.r.t. mu and logvar

**FIXED (Not Differentiable)**:
- `prior_mus`: ❌ Fixed (from server, not updated)
- `prior_logvars`: ❌ Fixed (from server, not updated)
- `prior_weights`: ❌ Fixed (from server, not updated)

### Step 4: Latent Conditioning
```python
# Project z to logit adjustments
latent_adjustment = self.latent_projection(z)  # (batch, num_choices)

# Apply to logits
conditioned_logits = new_logits + latent_adjustment_expanded

# Reconstruction loss
reconstruction_loss = CrossEntropyLoss(conditioned_logits, new_labels)
```

### Step 5: Total Loss
```python
vpl_loss = reconstruction_loss + self.vpl_kl_weight * kl_loss

# Add orthogonal loss if enabled
if self.vpl_orthogonal_weight > 0.0:
    orthogonal_loss = ...
    vpl_loss = vpl_loss + orthogonal_loss
```

---

## 3. Backward Pass 구성

### Gradient Flow

```
vpl_loss.backward()
  ↓
∂vpl_loss/∂reconstruction_loss = 1.0
  ↓
∂reconstruction_loss/∂latent_adjustment
  ↓
∂latent_adjustment/∂z
  ↓
∂z/∂mu, ∂z/∂logvar (reparameterization trick)
  ↓
∂vpl_loss/∂kl_loss = vpl_kl_weight
  ↓
∂kl_loss/∂mu, ∂kl_loss/∂logvar
  ↓
∂mu/∂encoder, ∂logvar/∂encoder
  ↓
encoder.backward()
```

### KL Loss의 Gradient

#### Standard VPL
```python
∂KL/∂mu = -0.5 * (-2 * mu) = mu
∂KL/∂logvar = -0.5 * (1 - exp(logvar))
```

**의미**:
- `mu`가 0에 가까워지도록 유도
- `logvar`가 0에 가까워지도록 유도 (즉, `var`가 1에 가까워지도록)

#### VPL-GP (Mixture Prior)
```python
∂KL/∂mu = ∂(log_q - log_p_mixture)/∂mu
        = ∂log_q/∂mu - ∂log_p_mixture/∂z * ∂z/∂mu

∂KL/∂logvar = ∂(log_q - log_p_mixture)/∂logvar
            = ∂log_q/∂logvar - ∂log_p_mixture/∂z * ∂z/∂logvar
```

**의미**:
- `mu`가 `prior_mus`의 가중 평균에 가까워지도록 유도
- `logvar`가 `prior_logvars`의 가중 평균에 가까워지도록 유도

---

## 4. KL Loss가 Minimize되지 않는 원인 분석

### 가능한 원인들

#### 1. Loss Scale 불균형
```
Total Loss = reconstruction_loss + vpl_kl_weight * kl_loss + orthogonal_loss

만약:
- reconstruction_loss ≈ 0.1
- kl_loss ≈ 10.0
- vpl_kl_weight = 0.1
- orthogonal_loss ≈ 1000.0

그러면:
- vpl_kl_weight * kl_loss = 1.0 (10%)
- orthogonal_loss = 1000.0 (99.9%)
- reconstruction_loss = 0.1 (0.01%)

→ KL loss의 기여도가 너무 작아서 minimize되지 않음
```

**해결책**:
- `vpl_kl_weight`를 증가시킴 (예: 0.1 → 1.0)
- `vpl_orthogonal_weight`를 감소시킴 (예: 1000.0 → 10.0)

#### 2. Prior와 Posterior의 불일치

**VPL-GP의 경우**:
- `prior_mus`가 클라이언트의 실제 `mu` 분포와 다를 수 있음
- `prior_logvars`가 클라이언트의 실제 `logvar` 분포와 다를 수 있음
- KL loss가 minimize되려면 `mu`가 `prior_mus`에 가까워져야 하는데, reconstruction loss가 이를 방해할 수 있음

**해결책**:
- Prior 업데이트 주기를 조정
- Prior weight를 조정

#### 3. Gradient Vanishing/Exploding

**문제**:
- `logvar`가 너무 작으면 `exp(logvar)`가 0에 가까워짐
- `logvar`가 너무 크면 `exp(logvar)`가 무한대로 발산
- `z`의 scale이 너무 크면 gradient가 불안정

**해결책**:
- `logvar`를 clipping: `logvar = torch.clamp(logvar, min=-10, max=10)`
- `z`의 scale을 정규화

#### 4. Reparameterization Trick의 문제

**문제**:
- `eps`는 random noise이므로 매 forward pass마다 다름
- 이로 인해 gradient가 noisy할 수 있음

**해결책**:
- 여러 샘플을 사용하여 Monte Carlo estimation
- 또는 deterministic mode 사용 (평균만 사용)

#### 5. Encoder의 학습 부족

**문제**:
- Encoder가 충분히 학습되지 않아 `mu`와 `logvar`가 적절히 업데이트되지 않음
- Learning rate가 너무 작거나, gradient가 차단됨

**해결책**:
- Encoder의 learning rate를 조정
- Encoder가 optimizer에 포함되어 있는지 확인

---

## 5. 디버깅 방법

### 1. Gradient 확인
```python
# Forward pass 후
kl_loss.backward(retain_graph=True)

# Gradient 확인
print(f"mu grad norm: {mu.grad.norm().item()}")
print(f"logvar grad norm: {logvar.grad.norm().item()}")
print(f"encoder grad norm: {self.variational_encoder.encoder[0].weight.grad.norm().item()}")
```

### 2. Loss 값 확인
```python
print(f"reconstruction_loss: {reconstruction_loss.item()}")
print(f"kl_loss: {kl_loss.item()}")
print(f"vpl_kl_weight * kl_loss: {self.vpl_kl_weight * kl_loss.item()}")
print(f"orthogonal_loss: {orthogonal_loss.item()}")
print(f"total_loss: {vpl_loss.item()}")
```

### 3. mu, logvar 값 확인
```python
print(f"mu mean: {mu.mean().item()}, std: {mu.std().item()}")
print(f"logvar mean: {logvar.mean().item()}, std: {logvar.std().item()}")
print(f"var mean: {torch.exp(logvar).mean().item()}")
```

### 4. Prior 확인 (VPL-GP)
```python
if hasattr(self.variational_encoder, 'prior_mus'):
    print(f"prior_mus mean: {self.variational_encoder.prior_mus.mean().item()}")
    print(f"prior_logvars mean: {self.variational_encoder.prior_logvars.mean().item()}")
    print(f"mu - prior_mus distance: {torch.norm(mu.mean(0) - self.variational_encoder.prior_mus.mean(0)).item()}")
```

---

## 6. 권장 해결책

### 즉시 적용 가능한 해결책

1. **Loss Weight 조정**:
   ```yaml
   llm:
     vpl_kl_weight: 1.0  # 0.1에서 증가
     vpl_orthogonal_weight: 10.0  # 1000.0에서 감소
   ```

2. **Logvar Clipping**:
   ```python
   logvar = torch.clamp(logvar, min=-10, max=10)
   ```

3. **Encoder Learning Rate 조정**:
   - Encoder가 충분히 학습되도록 learning rate 확인

### 장기 해결책

1. **Loss Balance 모니터링**:
   - 각 loss의 기여도를 실시간으로 모니터링
   - 적절한 비율 유지 (예: reconstruction 50%, KL 30%, orthogonal 20%)

2. **Prior 업데이트 전략**:
   - Prior 업데이트 주기 조정
   - Prior weight 계산 방법 개선

3. **Gradient Clipping**:
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

---

## 7. 요약

### Forward Pass
1. Feature extraction → Encoder → mu, logvar
2. Reparameterization: z = mu + eps * std
3. KL divergence 계산
4. Latent conditioning → Reconstruction loss
5. Total loss = reconstruction + kl_weight * kl + orthogonal

### Backward Pass
1. Total loss backward
2. Gradient가 mu, logvar로 전파
3. Encoder로 gradient 전파
4. Encoder 업데이트

### KL Loss가 Minimize되지 않는 주요 원인
1. **Loss scale 불균형** (가장 가능성 높음)
2. Prior와 Posterior의 불일치
3. Gradient vanishing/exploding
4. Encoder의 학습 부족

### 해결책
1. Loss weight 조정 (즉시 적용 가능)
2. Logvar clipping
3. Gradient monitoring 및 clipping
4. Loss balance 모니터링
