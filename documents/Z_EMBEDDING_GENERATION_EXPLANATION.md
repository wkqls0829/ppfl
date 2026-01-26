# Z Embedding 생성 과정 상세 설명

## 문제 상황

현재 t-SNE 플롯에서 클라이언트 0-4와 5-9가 각각 같은 분포로 매핑되길 원하지만, 실제로는 너무 섞여있습니다. 이는 **general information**이 z embedding에 포함되어 있기 때문입니다.

## Z Embedding 생성 파이프라인

### 1단계: Raw Feature Extraction (원시 특징 추출)

#### 1.1 Hidden States 추출
```python
# LLM forward pass
outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
logits = outputs.logits  # (batch, seq_len, vocab_size)
hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
```

**포함된 정보:**
- **General information**: 전체 프롬프트, 대화 맥락, 언어 모델의 일반적인 표현
- **Preference information**: 선택된 응답과 거부된 응답의 차이

#### 1.2 Choice Token 위치 찾기
```python
# Choice token 위치 찾기
A_token, B_token = choices[0], choices[1]  # 예: ['A', 'B']
A_positions = (shift_labels == A_token)  # A 토큰 위치
B_positions = (shift_labels == B_token)  # B 토큰 위치
```

#### 1.3 Embedding 추출
```python
# 각 배치 샘플에 대해
for b in range(batch_size):
    if A_found and B_found:
        chosen_emb = shift_hidden[b, A_pos].mean(dim=0)  # (hidden_dim,)
        rejected_emb = shift_hidden[b, B_pos].mean(dim=0)  # (hidden_dim,)
    # ...
    
    # 차이 계산
    feature_diff = chosen_emb - rejected_emb  # (hidden_dim,)
```

**문제점:**
- `chosen_emb`: 선택된 응답의 embedding → **general + preference 정보 포함**
- `rejected_emb`: 거부된 응답의 embedding → **general + preference 정보 포함**
- `feature_diff`: 차이 → **preference 정보만 포함** (이상적으로)

#### 1.4 Feature Concatenation
```python
if self.vpl_use_llm_feature_extractor:
    # [chosen_emb, rejected_emb, difference] 결합
    feature_combined = torch.cat([chosen_emb, rejected_emb, feature_diff], dim=0)
    # (hidden_dim * 3,) = (2048 * 3,) = (6144,)
else:
    # 차이만 사용
    feature_combined = feature_diff  # (hidden_dim,)
```

**현재 설정:**
- `vpl_use_llm_feature_extractor = True` (기본값)
- 따라서 `[chosen, rejected, difference]` 모두 사용
- **General information이 `chosen_emb`와 `rejected_emb`에 포함됨**

### 2단계: Feature Extractor MLP (특징 추출기)

```python
# Feature extractor: (6144,) -> (128,)
extracted_features = self.feature_extractor(preference_features)
```

**구조:**
```python
self.feature_extractor = nn.Sequential(
    nn.Linear(embedding_dim * 3, 512),  # 6144 -> 512
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 128)
)
```

**문제점:**
- Feature extractor가 `chosen_emb`와 `rejected_emb`의 general information을 학습할 수 있음
- `difference`만 사용하면 general information을 제거할 수 있지만, 현재는 3개를 모두 사용

### 3단계: Variational Encoder (변분 인코더)

```python
# Variational inference: encode to latent z
z, mu, logvar = self.variational_encoder(extracted_features)
```

**과정:**
1. **Encoder Network**: `extracted_features (128,) -> h (128,)`
2. **Mean Network**: `h -> mu (32,)`
3. **Logvar Network**: `h -> logvar (32,)`
4. **Reparameterization**: `z = mu + eps * exp(0.5 * logvar)`

**결과:**
- `z`: (batch, 32) - 최종 latent embedding
- `mu`: (batch, 32) - posterior mean
- `logvar`: (batch, 32) - posterior log variance

**문제점:**
- `extracted_features`에 general information이 포함되어 있으면, `z`에도 포함됨
- Variational encoder는 입력의 모든 정보를 압축하려고 시도하므로, general과 preference 정보가 모두 포함됨

### 4단계: Reconstruction Loss (재구성 손실)

```python
# Condition model on latent z
latent_adjustment = self.latent_projection(z)  # (batch, num_choices)
conditioned_logits = new_logits + latent_adjustment_expanded

# Reconstruction loss
reconstruction_loss = CrossEntropyLoss(conditioned_logits, labels)
```

**문제점:**
- Reconstruction loss는 **전체 sequence를 예측**해야 함
- 따라서 `z`는 전체 sequence를 예측하는 데 필요한 **모든 정보**를 포함해야 함
- 이는 general information (프롬프트, 맥락)과 preference information 모두 포함

### 5단계: KL Loss (KL 손실)

```python
kl_loss = self.variational_encoder.kl_divergence(mu, logvar)
```

**효과:**
- `KL(q(z|x) || p(z))`는 posterior를 prior에 가깝게 만듦
- Prior가 standard normal이면, `z`를 원점 근처로 당김
- **General information과 preference information 모두 정규화됨**

### 6단계: Orthogonal Loss (직교 손실)

```python
if self.vpl_orthogonal_weight > 0.0:
    orthogonal_loss = self._compute_clop_orthogonal_loss(z)
    # Pull loss: z를 prototype에 가깝게
    # Orthonormal constraint: prototype들이 orthonormal 유지
```

**효과:**
- `z`를 특정 prototype에 가깝게 당김
- 하지만 `z`에 general information이 포함되어 있으면, 같은 preference를 가진 클라이언트라도 다른 `z`를 가질 수 있음

## General Information이 포함되는 이유

### 1. Feature Extraction 단계

**현재 구현:**
```python
feature_combined = [chosen_emb, rejected_emb, difference]
```

**문제:**
- `chosen_emb`: 선택된 응답의 embedding → **응답 자체의 general information 포함**
- `rejected_emb`: 거부된 응답의 embedding → **응답 자체의 general information 포함**
- `difference`: 차이 → **preference 정보만 포함** (이상적으로)

**예시:**
- 클라이언트 0: "Which is more helpful? A: I can help you. B: I don't know."
  - `chosen_emb`: "I can help you"의 embedding (helpful한 응답의 일반적인 표현 포함)
  - `rejected_emb`: "I don't know"의 embedding (unhelpful한 응답의 일반적인 표현 포함)
  - `difference`: helpful vs unhelpful의 차이

- 클라이언트 5: "Which is more helpful? A: Let me assist you. B: Sorry, I can't."
  - `chosen_emb`: "Let me assist you"의 embedding (다른 표현이지만 helpful)
  - `rejected_emb`: "Sorry, I can't"의 embedding (다른 표현이지만 unhelpful)
  - `difference`: helpful vs unhelpful의 차이

**결과:**
- `difference`는 비슷하지만 (helpful vs unhelpful)
- `chosen_emb`와 `rejected_emb`는 **응답의 구체적인 표현**에 따라 다름
- 따라서 같은 preference (helpful)를 가진 클라이언트라도 다른 `z`를 가짐

### 2. Reconstruction Loss 단계

**현재 구현:**
```python
reconstruction_loss = CrossEntropyLoss(conditioned_logits, labels)
```

**문제:**
- Reconstruction loss는 **전체 sequence를 예측**해야 함
- `z`는 전체 sequence를 예측하는 데 필요한 정보를 포함해야 함
- 이는 **프롬프트, 맥락, 응답의 구체적인 표현** 등 general information 포함

**예시:**
- 같은 preference를 가진 클라이언트라도:
  - 다른 프롬프트 → 다른 `z` 필요
  - 다른 응답 표현 → 다른 `z` 필요
  - 다른 맥락 → 다른 `z` 필요

### 3. KL Loss 단계

**현재 구현:**
```python
kl_loss = KL(q(z|x) || p(z))
```

**효과:**
- Posterior를 prior에 가깝게 만듦
- 하지만 **general information과 preference information 모두 정규화**
- General information이 많으면, preference information이 상대적으로 작아짐

## 해결 방안

### 방안 1: Difference만 사용

```python
# 현재
feature_combined = [chosen_emb, rejected_emb, difference]  # (6144,)

# 수정
feature_combined = difference  # (2048,)
```

**장점:**
- General information 제거
- Preference information만 capture

**단점:**
- 정보 손실 가능
- Reconstruction 성능 저하 가능

### 방안 2: Adversarial Loss 추가

```python
# General information을 제거하는 adversarial loss
adversarial_loss = -log(D(general_info_extractor(z)))
```

**장점:**
- General information을 명시적으로 제거
- Preference information만 유지

**단점:**
- 구현 복잡도 증가
- 하이퍼파라미터 추가

### 방안 3: Contrastive Loss 추가

```python
# 같은 preference를 가진 클라이언트의 z를 가깝게
contrastive_loss = -log(exp(sim(z_i, z_j)) / sum(exp(sim(z_i, z_k))))
```

**장점:**
- 같은 preference를 가진 클라이언트를 명시적으로 가깝게
- General information의 영향 감소

**단점:**
- 서버에서 클라이언트 간 similarity 계산 필요
- 통신 오버헤드 증가

### 방안 4: Reconstruction Loss 수정

```python
# 전체 sequence가 아닌 choice만 예측
reconstruction_loss = CrossEntropyLoss(choice_logits, choice_labels)
```

**장점:**
- General information 필요성 감소
- Preference information에 집중

**단점:**
- 전체 모델 성능 저하 가능

## 현재 설정 확인

```yaml
llm:
  vpl_use_feature_difference: True  # Embedding difference 사용
  vpl_feature_method: 'choice_logits'  # Choice logits 방법
```

**실제 사용:**
- `vpl_use_feature_difference = True`이면 `_extract_embedding_difference` 사용
- `vpl_use_llm_feature_extractor = True`이면 `[chosen, rejected, difference]` 모두 사용
- 따라서 **general information이 포함됨**

## 권장 사항

1. **Difference만 사용**: `vpl_use_llm_feature_extractor = False`로 설정
2. **Reconstruction loss 수정**: Choice만 예측하도록 수정
3. **Contrastive loss 추가**: 같은 preference를 가진 클라이언트를 가깝게
4. **Adversarial loss 추가**: General information을 명시적으로 제거
