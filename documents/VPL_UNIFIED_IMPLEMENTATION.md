# VPL 통합 구현 설명

## 개요

VPL (Variational Preference Learning)과 VPL-GP (VPL with Gumbel-Softmax Prior)를 하나의 통합된 trainer로 구현했습니다. Config 설정으로 GP prior 사용 여부를 제어할 수 있습니다.

## 통합 구조

### 1. 단일 Trainer 클래스

**이전 구조:**
- `VPLRewardChoiceTrainer`: 기본 VPL (표준 정규 prior)
- `VPLGPRewardChoiceTrainer`: VPL-GP (mixture prior, VPLRewardChoiceTrainer 상속)

**현재 구조:**
- `VPLRewardChoiceTrainer`: 통합 trainer (config로 GP prior 제어)

### 2. Config 설정

```yaml
llm:
  # VPL 기본 설정
  vpl_latent_dim: 32
  vpl_kl_weight: 0.1
  vpl_use_feature_difference: True
  vpl_use_llm_feature_extractor: True
  
  # GP Prior 설정 (선택적)
  vpl_use_gp_prior: True  # True면 GP prior 사용, False면 표준 정규 prior
  vpl_gp_temperature: 1.0  # Gumbel-Softmax temperature

federate:
  client_num: 10  # GP prior에 필요한 클라이언트 수
```

## 구현 세부사항

### 1. 초기화 (`__init__`)

```python
# GP prior 설정 읽기
self.vpl_use_gp_prior = getattr(config.llm, 'vpl_use_gp_prior', False)
self.vpl_gp_temperature = getattr(config.llm, 'vpl_gp_temperature', 1.0)
self.num_clients = getattr(config.federate, 'client_num', 10)

# Variational Encoder 선택
if self.vpl_use_gp_prior:
    # GP prior 사용: VariationalEncoderGP (mixture prior)
    from federatedscope.llm.model.variational_encoder_gp import VariationalEncoderGP
    self.variational_encoder = VariationalEncoderGP(
        input_dim=self.feature_extractor_output_dim,
        latent_dim=self.vpl_latent_dim,
        hidden_dims=[512, 256, 128],
        temperature=self.vpl_gp_temperature,
        num_clients=self.num_clients
    ).to(device)
else:
    # 표준 VPL: VariationalEncoder (표준 정규 prior)
    self.variational_encoder = VariationalEncoder(
        input_dim=self.feature_extractor_output_dim,
        latent_dim=self.vpl_latent_dim,
        hidden_dims=[512, 256, 128]
    ).to(device)

# GP prior 관련 속성 초기화 (GP prior 사용 시에만)
if self.vpl_use_gp_prior:
    self.z_history = []
    self.client_z_mu = None
    self.client_z_logvar = None
```

### 2. Forward Pass (`_hook_on_batch_forward`)

```python
# Feature 추출 (chosen, rejected, difference)
features = self._extract_preference_features(logits, labels, choices, hidden_states)

# Variational encoder로 latent z 추정
mu, logvar = self.variational_encoder.encode(features)
z = self.variational_encoder.reparameterize(mu, logvar)

# GP prior 사용 시 z 값 수집
if self.vpl_use_gp_prior:
    self.z_history.append(z.detach().cpu())

# KL divergence 계산
# - GP prior: KL(q(z|x) || p_mixture(z)) (다른 클라이언트들의 z 분포)
# - 표준 VPL: KL(q(z|x) || N(0, I)) (표준 정규 분포)
kl_loss = self.variational_encoder.kl_divergence(mu, logvar)
```

### 3. Prior 업데이트 (GP prior만)

```python
def update_prior_from_server(self, client_mus, client_logvars, client_weights):
    """
    서버에서 받은 다른 클라이언트들의 z 분포로 mixture prior 업데이트.
    GP prior 사용 시에만 동작.
    """
    if not self.vpl_use_gp_prior:
        return
    if hasattr(self.variational_encoder, 'update_prior'):
        self.variational_encoder.update_prior(client_mus, client_logvars, client_weights)
```

### 4. Z 분포 수집 (`_hook_on_fit_end`)

```python
# GP prior 사용 시 클라이언트의 z 분포 수집
if self.vpl_use_gp_prior and len(self.z_history) > 0:
    z_values = torch.cat(self.z_history, dim=0)
    
    # 클라이언트의 z 분포 계산 (평균과 분산)
    self.client_z_mu = z_values.mean(dim=0)
    self.client_z_logvar = torch.log(z_values.var(dim=0) + 1e-8)
    
    # 시각화용 샘플 저장
    self.client_z_values = z_values[sampled_indices]
    
    # 다음 라운드를 위해 초기화
    self.z_history = []
```

## Prior 비교

### 표준 VPL (vpl_use_gp_prior=False)

```
Prior: p(z) = N(0, I)  (표준 정규 분포)
KL Loss: KL(q(z|x) || N(0, I))

장점:
- 단순하고 빠름
- 독립적인 학습
- 추가 통신 불필요

단점:
- 클라이언트 간 지식 공유 없음
- 유사한 선호도 클라이언트 간 협력 없음
```

### VPL-GP (vpl_use_gp_prior=True)

```
Prior: p(z) = Σ w_i * N(μ_i, σ_i)  (다른 클라이언트들의 mixture)
KL Loss: KL(q(z|x) || p_mixture(z))

장점:
- 클라이언트 간 선호도 지식 공유
- 유사한 선호도 클라이언트 간 협력
- 더 나은 일반화

단점:
- 추가 통신 필요 (z 분포 전송)
- 계산 비용 증가 (Gumbel-Softmax)
```

## 사용 예시

### 표준 VPL 사용

```yaml
trainer:
  type: vplrewardchoicetrainer

llm:
  vpl_use_gp_prior: False  # 표준 정규 prior 사용
```

### VPL-GP 사용

```yaml
trainer:
  type: vplrewardchoicetrainer  # 또는 vplgprewardchoicetrainer (하위 호환)

llm:
  vpl_use_gp_prior: True  # GP prior 사용
  vpl_gp_temperature: 1.0
```

## 하위 호환성

기존 코드에서 `vplgprewardchoicetrainer`를 사용하더라도 자동으로 통합된 trainer를 사용합니다:

```python
# trainer_builder.py에서
elif config.trainer.type.lower() in ['vplrewardchoicetrainer', 'vplgprewardchoicetrainer']:
    dict_path = "federatedscope.llm.trainer.vpl_reward_choice_trainer"
```

## 주요 메서드

### GP Prior 관련 (vpl_use_gp_prior=True일 때만 동작)

1. **`get_client_z_distribution()`**: 클라이언트의 z 분포 (μ, logvar) 반환
2. **`get_client_z_values()`**: 시각화용 z 값 반환
3. **`update_prior_from_server()`**: 서버에서 받은 mixture prior 업데이트
4. **`update_orthogonal_label_from_server()`**: 서버에서 받은 orthogonal label 업데이트
5. **`get_client_orthogonal_prototypes()`**: Orthogonal prototypes 반환

## Feature Extraction

### 원래 VPL 방식

```python
# Input: [chosen_emb, rejected_emb, chosen_emb - rejected_emb]
# Output: 128-dim feature vector

feature_extractor:
  Input: embedding_dim * 3
  → Linear(512) → ReLU → Dropout
  → Linear(256) → ReLU → Dropout
  → Linear(128)
  Output: 128
```

### Variational Encoder

```python
variational_encoder:
  Input: 128 (from feature_extractor)
  → Linear(512) → ReLU → Dropout
  → Linear(256) → ReLU → Dropout
  → Linear(128) → ReLU → Dropout
  → μ: Linear(32), logvar: Linear(32)
  Output: z ~ N(μ, exp(logvar))
```

## 전체 파이프라인

```
1. Forward Pass (1번만)
   ↓
2. Hidden States 추출
   ↓
3. Feature Extraction
   [chosen_emb, rejected_emb, difference] → 128-dim
   ↓
4. Variational Encoder
   128 → 512 → 256 → 128 → μ(32), logvar(32)
   ↓
5. Reparameterization
   z = μ + ε * exp(0.5 * logvar), ε ~ N(0, I)
   ↓
6. KL Divergence
   - 표준 VPL: KL(q(z|x) || N(0, I))
   - VPL-GP: KL(q(z|x) || p_mixture(z))
   ↓
7. Reconstruction Loss
   Choice prediction with z conditioning
   ↓
8. Total Loss
   Loss = Reconstruction + KL_weight * KL_loss
```

## 장점

1. **단일 코드베이스**: 하나의 trainer로 두 가지 모드 지원
2. **Config 제어**: 코드 수정 없이 설정만으로 전환
3. **하위 호환성**: 기존 코드 그대로 사용 가능
4. **유지보수 용이**: 중복 코드 제거
5. **확장성**: 새로운 prior 타입 추가 용이
