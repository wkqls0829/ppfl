# VPL (Variational Preference Learning) 구현 상세 설명

## 1. 전체 아키텍처

### 1.1 기본 VPL 구조
```
Input (preference data) 
  → Feature Extraction (choice_logits)
  → Variational Encoder (q(z|x))
  → Latent z (user-specific preference representation)
  → Latent Conditioning (z → logit adjustment)
  → Conditioned Model Output
```

### 1.2 VPL-GP 확장 (Federated)
```
Client 1: z₁ ~ q₁(z|x₁) → Server: Aggregate {z₁, z₂, ..., zₙ}
  → Mixture Prior: p_mixture(z) = Σᵢ wᵢ N(z; μᵢ, σᵢ²)
  → Broadcast to clients
  → KL(q(z|x) || p_mixture(z))
```

## 2. 핵심 컴포넌트

### 2.1 Feature Extraction (`_extract_preference_features`)

**목적**: 모델의 logits에서 preference 정보를 추출

**과정**:
1. **Input**: 
   - `logits`: (batch, seq_len, vocab_size) - 모델 출력
   - `labels`: (batch, seq_len) - 정답 토큰
   - `choices`: [586, 599] - 'A', 'B' 토큰 ID

2. **Processing**:
   ```python
   # 각 choice 토큰의 logit 값 추출
   choice_logits_per_token = shift_logits[..., choices]  # (batch, seq_len, 2)
   
   # 각 choice 토큰이 나타나는 위치에서 logit 평균 계산
   for choice_idx, choice_token in enumerate(choices):
       choice_positions = (shift_labels == choice_token)
       # 해당 위치의 logit 평균
       avg_logit = choice_logits_per_token[b, choice_positions, choice_idx].mean()
   ```

3. **Output**: 
   - `features`: (batch, 4) = [A_logit, B_logit, A_logit, B_logit]
   - 중복된 이유: input_dim=4로 설정되어 있음 (len(choices) * 2)

### 2.2 Variational Encoder (`VariationalEncoder`)

**구조**:
```
Input (batch, 4) 
  → Encoder Network [Linear(4→128) → ReLU → Dropout → Linear(128→64) → ReLU → Dropout]
  → μ: Linear(64 → 32)  # Mean
  → logvar: Linear(64 → 32)  # Log variance
  → Reparameterization: z = μ + ε * exp(0.5 * logvar)
```

**핵심 메서드**:
- `encode(x)`: μ, logvar 계산
- `reparameterize(μ, logvar)`: z 샘플링 (reparameterization trick)
- `kl_divergence(μ, logvar)`: KL(q(z|x) || N(0, I)) 계산

**KL Divergence 공식**:
```
KL(q(z|x) || p(z)) = -0.5 * Σ(1 + logvar - μ² - exp(logvar))
```

### 2.3 Latent Conditioning

**목적**: Latent z를 모델의 logits에 반영하여 개인화

**과정**:
```python
# 1. z를 choice logit adjustment로 변환
latent_adjustment = self.latent_projection(z)  # (batch, 2)
# latent_projection: Linear(32 → 2)

# 2. Logits에 adjustment 추가
conditioned_logits = new_logits + latent_adjustment_expanded
# new_logits: (batch, seq_len, 2)
# latent_adjustment_expanded: (batch, seq_len, 2)
```

**효과**: 
- z가 사용자의 preference를 인코딩
- z에 따라 모델의 choice 예측이 조정됨
- 같은 입력이라도 사용자별로 다른 선택 가능

## 3. Loss 계산

### 3.1 ELBO (Evidence Lower Bound)

```
ELBO = E_q(z|x)[log p(y|z,x)] - KL(q(z|x) || p(z))
     = Reconstruction Loss - KL Divergence
```

**구성 요소**:

1. **Reconstruction Loss**:
   ```python
   reconstruction_loss = CrossEntropyLoss(
       conditioned_logits.view(-1, num_choices),
       new_labels.view(-1)
   )
   ```
   - z가 조건부로 주어진 상태에서의 예측 정확도
   - `conditioned_logits`: z로 조정된 logits

2. **KL Divergence**:
   ```python
   kl_loss = self.variational_encoder.kl_divergence(mu, logvar)
   ```
   - Posterior q(z|x)가 prior p(z)에서 얼마나 벗어나는지
   - Regularization 역할

3. **Total VPL Loss**:
   ```python
   vpl_loss = reconstruction_loss + self.vpl_kl_weight * kl_loss
   ```
   - `vpl_kl_weight`: KL loss의 가중치 (기본값: 0.1)

## 4. VPL-GP 확장 (Federated Mixture Prior)

### 4.1 VariationalEncoderGP

**차이점**: Standard normal prior 대신 **mixture prior** 사용

**Mixture Prior 구성**:
```python
p_mixture(z) = Σᵢ wᵢ * N(z; μᵢ, σᵢ²)
```
- `μᵢ, σᵢ²`: 다른 클라이언트들의 z 분포
- `wᵢ`: 각 클라이언트의 가중치

### 4.2 Gumbel-Softmax Sampling

**목적**: Mixture prior에서 샘플링 시 discrete selection을 smooth하게

**과정**:
```python
# 1. Gumbel noise 생성
gumbel_noise = -log(-log(U + ε) + ε)  # U ~ Uniform(0,1)

# 2. Component 선택 확률 계산
gumbel_logits = (log_weights + gumbel_noise) / temperature
component_probs = softmax(gumbel_logits)  # (batch, num_components)

# 3. 각 component에서 샘플링 후 가중 평균
for i in range(num_components):
    z_i ~ N(μᵢ, σᵢ²)
    z_samples += component_probs[i] * z_i
```

**Temperature**:
- `temperature → 0`: Hard selection (one-hot)
- `temperature → ∞`: Uniform selection
- 기본값: 1.0

### 4.3 Mixture KL Divergence

**계산**:
```python
# 1. Sample z from q(z|x)
z = μ + ε * exp(0.5 * logvar)

# 2. Compute log q(z|x)
log_q = -0.5 * Σ(log(2π) + logvar + (z-μ)²/var)

# 3. Compute log p_mixture(z) = log(Σᵢ wᵢ * N(z; μᵢ, σᵢ²))
log_p_components = []
for i in range(num_components):
    log_p_i = -0.5 * Σ(log(2π) + logvar_i + (z-μᵢ)²/var_i)
    log_p_i += log(wᵢ)
    log_p_components.append(log_p_i)

# 4. Log-sum-exp trick
log_p_mixture = max(log_p_components) + log(Σ exp(log_p_components - max))

# 5. KL = E_q[log q - log p_mixture]
kl = (log_q - log_p_mixture).mean()
```

**특징**:
- Standard KL보다 계산 비용 높음 (num_components만큼)
- 다른 클라이언트의 preference 정보 활용

## 5. Federated Learning 통합

### 5.1 클라이언트 → 서버

**전송 데이터**:
```python
{
    'client_z_mu': (latent_dim,)  # z 분포의 평균
    'client_z_logvar': (latent_dim,)  # z 분포의 log variance
    'sample_size': int  # 샘플 수 (가중치 계산용)
    'client_z_values': (num_samples, latent_dim)  # 시각화용
    'client_orthogonal_prototypes': (num_prototypes, latent_dim)  # Orthogonal loss용
}
```

### 5.2 서버 → 클라이언트

**브로드캐스트 데이터**:
```python
{
    'vpl_gp_prior': {
        'prior_mus': (num_clients, latent_dim),
        'prior_logvars': (num_clients, latent_dim),
        'prior_weights': (num_clients,)
    },
    'vpl_orthogonal_labels': (num_clients,)  # Manual/K-Means labels
}
```

### 5.3 서버에서의 Prior 업데이트

**과정**:
1. **수집**: 모든 클라이언트의 (μᵢ, logvarᵢ, sample_sizeᵢ)
2. **가중치 계산**: `wᵢ = sample_sizeᵢ / Σⱼ sample_sizeⱼ`
3. **정규화**: `wᵢ = wᵢ / Σⱼ wⱼ`
4. **브로드캐스트**: 모든 클라이언트에 전송

## 6. Orthogonal Loss (선택적)

### 6.1 목적

**Preference 분리**: 서로 다른 preference를 가진 클라이언트들의 z를 orthogonal하게 유지

### 6.2 구현

```python
# Prototypes: (num_prototypes, latent_dim)
# 각 클라이언트는 하나의 prototype에 할당됨

# Orthogonal loss: prototypes 간의 내적 최소화
orthogonal_loss = sum(prototypes[i] @ prototypes[j] for i < j)

# Magnitude regularization: prototypes가 원점으로 수축 방지
magnitude_loss = (||prototypes|| - target_magnitude)²
```

### 6.3 Manual Labeling

**방식**: 클라이언트 ID 기반 수동 할당
- Harmlessness 데이터 클라이언트 (0-4): label = 0
- Helpfulness 데이터 클라이언트 (5-9): label = 1

## 7. 학습 과정

### 7.1 Forward Pass

```
1. Input: (input_ids, labels, attention_mask)
2. Base Model: logits = model(input_ids)
3. Feature Extraction: features = extract(logits, labels, choices)
4. Variational Encoding: z, μ, logvar = encoder(features)
5. Latent Conditioning: conditioned_logits = logits + projection(z)
6. Loss: vpl_loss = reconstruction_loss + kl_weight * kl_loss
```

### 7.2 Backward Pass

```
1. vpl_loss.backward()
2. Gradients flow through:
   - latent_projection
   - variational_encoder
   - base_model (via conditioned_logits)
```

### 7.3 Round End

```
1. Collect z values: z_history → client_z_mu, client_z_logvar
2. Send to server: (mu, logvar, sample_size, z_values)
3. Receive from server: (prior_mus, prior_logvars, prior_weights)
4. Update prior: encoder.update_prior(...)
```

## 8. 하이퍼파라미터

### 8.1 VPL 기본
- `vpl_latent_dim`: 32 (latent z 차원)
- `vpl_kl_weight`: 0.1 (KL loss 가중치)
- `vpl_feature_method`: 'choice_logits' (feature 추출 방법)

### 8.2 VPL-GP
- `vpl_gp_temperature`: 1.0 (Gumbel-Softmax temperature)
- `vpl_use_gp_prior`: True (mixture prior 사용 여부)

### 8.3 Orthogonal Loss
- `vpl_orthogonal_weight`: 10.0 (orthogonal loss 가중치)
- `vpl_use_manual_orthogonal_labels`: True (수동 라벨링 사용)
- `vpl_orthogonal_orthonorm_weight`: 0.1 (prototype magnitude regularization)

## 9. 메모리 최적화

### 9.1 배치 처리
- `z_history`에 z 값 저장 (detached)
- Round end에서만 평균 계산

### 9.2 주기적 정리
```python
if ctx.cur_batch_i % 5 == 0:
    gc.collect()
    torch.cuda.empty_cache()
```

### 9.3 Eval 모드
```python
if ctx.cur_mode != MODE.TRAIN:
    del preference_features, z, mu, logvar, ...
```

## 10. 시각화

### 10.1 t-SNE Plot
- `client_z_values`: 각 클라이언트의 z 샘플
- `orthogonal_prototypes`: Orthogonal basis vectors
- Round별로 `cross_client_z_tsne_round_{round}.png` 생성

### 10.2 WandB Logging
- `vpl_kl_loss`: KL divergence
- `vpl_reconstruction_loss`: Reconstruction loss
- `vpl_elbo_loss`: Total ELBO loss

## 11. 주요 차이점: VPL vs VPL-GP

| 항목 | VPL | VPL-GP |
|------|-----|--------|
| Prior | N(0, I) | Mixture: Σᵢ wᵢ N(μᵢ, σᵢ²) |
| KL Divergence | Standard KL | Mixture KL (log-sum-exp) |
| Federated | X | O (다른 클라이언트 정보 활용) |
| Gumbel-Softmax | X | O (mixture sampling) |
| Temperature | - | 1.0 (Gumbel-Softmax) |

## 12. 수식 요약

### 12.1 ELBO
```
ELBO = E_q(z|x)[log p(y|z,x)] - KL(q(z|x) || p(z))
```

### 12.2 Reparameterization Trick
```
z = μ + ε * σ,  where ε ~ N(0, I), σ = exp(0.5 * logvar)
```

### 12.3 Standard KL
```
KL(q||p) = -0.5 * Σ(1 + logvar - μ² - exp(logvar))
```

### 12.4 Mixture KL
```
KL(q||p_mixture) = E_q[log q(z|x) - log(Σᵢ wᵢ N(z; μᵢ, σᵢ²))]
```

### 12.5 Gumbel-Softmax
```
πᵢ = softmax((log wᵢ + gumbel_noise) / τ)
```

## 13. 디버깅 팁

### 13.1 z 값 확인
- `z_history`: 배치별 z 값
- `client_z_mu`: 클라이언트 평균 z
- `client_z_logvar`: 클라이언트 z 분산

### 13.2 Prior 확인
- `prior_mus`: 다른 클라이언트들의 μ
- `prior_weights`: 각 클라이언트의 가중치
- `avg_mu_distance`: 클라이언트 간 평균 거리

### 13.3 Loss 확인
- `vpl_kl_loss`: KL divergence 값
- `vpl_reconstruction_loss`: Reconstruction loss
- `vpl_total`: 총 샘플 수
