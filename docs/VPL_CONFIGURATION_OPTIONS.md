# VPL Configuration Options 완전 가이드

## 개요

이 문서는 VPL, GP Prior, Orthogonal Loss의 모든 configuration 옵션을 설명합니다.

## 전체 옵션 리스트

### 1. Basic VPL Options

#### `vpl_latent_dim`
- **타입**: `int`
- **기본값**: `32`
- **설명**: Latent space의 차원
- **권장 범위**: 16-64
- **효과**:
  - 작은 값: 더 강한 regularization, 빠른 학습, 표현력 제한
  - 큰 값: 더 풍부한 표현, overfitting 위험, 느린 학습

#### `vpl_kl_weight`
- **타입**: `float`
- **기본값**: `0.1`
- **설명**: KL divergence의 가중치
- **권장 범위**: 0.01-1.0
- **효과**:
  - 작은 값 (0.01-0.1): 더 유연한 posterior, overfitting 위험
  - 큰 값 (0.5-1.0): 강한 regularization, underfitting 위험
- **수식**: `total_loss = reconstruction_loss + vpl_kl_weight * kl_loss`

#### `vpl_feature_method`
- **타입**: `str`
- **기본값**: `'choice_logits'`
- **옵션**: `'choice_logits'` or `'embedding_difference'`
- **설명**: Feature 추출 방법
- **차이**:
  - `'choice_logits'`: Choice token 위치의 logits 사용
  - `'embedding_difference'`: Hidden states에서 embedding 차이 사용

#### `vpl_use_feature_difference`
- **타입**: `bool`
- **기본값**: `False`
- **설명**: Embedding difference 사용 여부
- **효과**:
  - `True`: `chosen_emb - rejected_emb` 계산
  - `False`: Logits-based features 사용
- **권장**: `True` (preference 정보에 집중)

#### `vpl_use_difference_only` ⭐ **NEW**
- **타입**: `bool`
- **기본값**: `False`
- **설명**: Difference embedding만 사용 (general information 제거)
- **효과**:
  - `True`: Feature extractor 입력 = `difference` only (embedding_dim)
  - `False`: Feature extractor 입력 = `[chosen, rejected, difference]` (3 * embedding_dim)
- **권장**: `True` (preference-only 학습 시)
- **장점**: General information 제거, 같은 preference를 가진 클라이언트가 비슷한 z 분포로 매핑
- **단점**: 정보 손실 가능, reconstruction 성능 저하 가능

#### `vpl_use_llm_feature_extractor`
- **타입**: `bool`
- **기본값**: `True`
- **설명**: MLP feature extractor 사용 여부
- **효과**:
  - `True`: Deeper MLP network (512 → 256 → 128)
  - `False`: Simpler feature extraction
- **권장**: `True`

### 2. GP Prior Options

#### `vpl_use_gp_prior`
- **타입**: `bool`
- **기본값**: `False`
- **설명**: Gumbel-Softmax Prior (Mixture Prior) 사용 여부
- **효과**:
  - `True`: 다른 클라이언트들의 z 분포를 mixture prior로 사용
  - `False`: Standard normal prior `N(0, I)` 사용
- **수식**: `p_mixture(z) = Σ_i w_i * N(z; μ_i, σ_i²)`

#### `vpl_gp_temperature`
- **타입**: `float`
- **기본값**: `1.0`
- **설명**: Gumbel-Softmax temperature
- **권장 범위**: 0.5-2.0
- **효과**:
  - 작은 값 (0.5-1.0): 더 discrete한 샘플링, gradient 불안정 가능
  - 큰 값 (1.0-2.0): 부드러운 gradient, 덜 discrete
- **용도**: Prior에서 샘플링 시 사용 (현재 KL 계산에는 직접 사용 안 함)

### 3. Orthogonal Loss Options

#### `vpl_orthogonal_weight`
- **타입**: `float`
- **기본값**: `0.0`
- **설명**: Pull loss의 가중치
- **권장 범위**: 1.0-20.0
- **효과**:
  - 작은 값 (1.0-5.0): 자연스러운 학습, collapse 방지 효과 약함
  - 큰 값 (10.0-20.0): 강한 separation, 학습 불안정 가능
- **수식**: `L_pull = (1/N) Σ ||z - prototype[label]||²`
- **CLOP paper 권장**: 10.0

#### `vpl_orthogonal_orthonorm_weight`
- **타입**: `float`
- **기본값**: `0.1`
- **설명**: Orthonormal constraint의 가중치
- **권장 범위**: 0.01-1.0
- **효과**:
  - 작은 값 (0.01-0.1): 더 유연한 prototype 구조
  - 큰 값 (0.5-1.0): 더 엄격한 orthonormality
- **수식**: `L_orthonorm = ||P^T P - I||²_F`

#### `vpl_use_manual_orthogonal_labels`
- **타입**: `bool`
- **기본값**: `False`
- **설명**: Manual label 사용 여부
- **옵션**:
  - `True`: 서버가 수동으로 라벨 할당 (첫 절반: 0, 두 번째 절반: 1)
  - `False`: 서버가 k-means clustering으로 자동 라벨 할당 (권장)
- **권장**: `False` (k-means가 더 balanced)

#### `vpl_num_prototypes`
- **타입**: `int`
- **기본값**: `num_clients`
- **설명**: Orthogonal prototype의 개수 (k-means의 k)
- **특수 케이스**:
  - `hh-rlhf` 또는 `hrl` 데이터셋: 자동으로 2로 고정
  - 다른 데이터셋: 이 값 사용
- **권장 범위**: 2-10

#### `vpl_prototype_scale` ⭐ **NEW**
- **타입**: `float`
- **기본값**: `5.0`
- **설명**: Prototype의 원점으로부터의 거리
- **권장 범위**: 2.0-10.0
- **효과**:
  - 작은 값 (2.0-3.0): Prototype이 원점에 가까움
  - 큰 값 (5.0-10.0): Prototype이 원점에서 멀리, z embedding과 분리
- **구현**:
  ```python
  # QR decomposition으로 orthonormalize (norm=1)
  Q, R = torch.linalg.qr(prototypes.T)
  # Scale하여 원점에서 거리 조정
  prototypes = Q.T * prototype_scale
  ```
- **권장**: 5.0 (z embedding과 적절한 분리)

### 4. Visualization Options

#### `vpl_tsne_visualize_freq`
- **타입**: `int`
- **기본값**: `10`
- **설명**: t-SNE 시각화 생성 빈도 (라운드 단위)
- **효과**: 매 N 라운드마다 t-SNE 플롯 생성
- **권장**: 10 (너무 자주 생성하면 I/O 오버헤드)

#### Z 값 수집 최적화 (vpl_tsne_visualize_freq 연동)

HHST에서 z 값을 **매 라운드가 아니라**, t-SNE 시각화가 필요한 라운드에만 수집하도록 최적화되어 있습니다.

- **Server**: `vpl_tsne_visualize_freq`를 보고 해당 라운드에만 `_collect_z_values_for_visualization()` 호출 (`federatedscope/llm/llm_local/server.py`).
- **Client**: 같은 로직으로 시각화가 필요한 라운드에만 `get_client_z_values()` 계산·전송 (`federatedscope/llm/llm_local/client.py`).
- **효과**: 50 라운드 기준 z 수집 약 90% 감소 (기본값 10라운드마다 수집).
- **주의**: `vpl_use_gp_prior: True`이면 z 분포(mu, logvar)는 GP prior 업데이트를 위해 매 라운드 수집됩니다. `vpl_tsne_visualize_freq: 1`이면 매 라운드 수집(최적화 없음). 최종 라운드는 항상 z 수집하여 최종 시각화를 생성합니다.

## Feature Extraction 방법 비교

### 방법 1: Choice Logits
```yaml
vpl_feature_method: 'choice_logits'
vpl_use_feature_difference: False
```
- **입력**: `[logit_A_chosen, logit_B_chosen, logit_A_rejected, logit_B_rejected]`
- **장점**: 간단, 빠름
- **단점**: Preference 정보가 덜 명확

### 방법 2: Embedding Difference (Full)
```yaml
vpl_use_feature_difference: True
vpl_use_difference_only: False
```
- **입력**: `[chosen_emb, rejected_emb, difference]` (3 * embedding_dim)
- **장점**: Richer representation
- **단점**: General information 포함 (응답의 구체적 표현)

### 방법 3: Embedding Difference (Difference Only) ⭐ **Recommended**
```yaml
vpl_use_feature_difference: True
vpl_use_difference_only: True
```
- **입력**: `difference` only (embedding_dim)
- **장점**: General information 제거, preference 정보만 capture
- **단점**: 정보 손실 가능
- **사용 사례**: 같은 preference를 가진 클라이언트를 비슷한 z 분포로 매핑하고 싶을 때

## 실험 설정 예시

### Baseline (VPL only, no GP, no Orthogonal)
```yaml
llm:
  vpl_latent_dim: 32
  vpl_kl_weight: 0.1
  vpl_use_feature_difference: True
  vpl_use_difference_only: False
  vpl_use_gp_prior: False
  vpl_orthogonal_weight: 0.0
```

### VPL-GP (with GP Prior, no Orthogonal)
```yaml
llm:
  vpl_latent_dim: 32
  vpl_kl_weight: 0.1
  vpl_use_feature_difference: True
  vpl_use_difference_only: False
  vpl_use_gp_prior: True
  vpl_gp_temperature: 1.0
  vpl_orthogonal_weight: 0.0
```

### VPL-GP + Orthogonal Loss (Full)
```yaml
llm:
  vpl_latent_dim: 32
  vpl_kl_weight: 0.1
  vpl_use_feature_difference: True
  vpl_use_difference_only: False
  vpl_use_gp_prior: True
  vpl_gp_temperature: 1.0
  vpl_orthogonal_weight: 10.0
  vpl_orthogonal_orthonorm_weight: 0.1
  vpl_use_manual_orthogonal_labels: False
  vpl_num_prototypes: 2
  vpl_prototype_scale: 5.0
```

### Preference-Only Learning (Difference Only) ⭐ **NEW**
```yaml
llm:
  vpl_latent_dim: 32
  vpl_kl_weight: 0.1
  vpl_use_feature_difference: True
  vpl_use_difference_only: True  # ⭐ General information 제거
  vpl_use_gp_prior: True
  vpl_gp_temperature: 1.0
  vpl_orthogonal_weight: 10.0
  vpl_orthogonal_orthonorm_weight: 0.1
  vpl_use_manual_orthogonal_labels: False
  vpl_num_prototypes: 2
  vpl_prototype_scale: 5.0
```

## 최근 변경사항 요약

### 1. Difference-Only Embedding (`vpl_use_difference_only`)
- **추가일**: 2025-01-20
- **목적**: General information 제거, preference 정보만 capture
- **효과**: 같은 preference를 가진 클라이언트가 비슷한 z 분포로 매핑
- **사용**: 실험 40100, 50100

### 2. Prototype Scale (`vpl_prototype_scale`)
- **추가일**: 2025-01-20
- **목적**: Prototype을 원점에서 더 멀리 배치하여 z embedding과 분리
- **기본값**: 5.0
- **효과**: t-SNE 시각화에서 prototype과 z embedding의 분리 개선

### 3. 통합 Trainer
- **변경일**: 이전
- **내용**: `VPLGPRewardChoiceTrainer`를 `VPLRewardChoiceTrainer`에 통합
- **효과**: Config로 GP prior 사용 여부 제어 가능

## 옵션 간 상호작용

### Feature Extraction 옵션
```
vpl_use_feature_difference
  ├─ False → vpl_feature_method 사용 (choice_logits)
  └─ True
      ├─ vpl_use_difference_only=False → [chosen, rejected, difference]
      └─ vpl_use_difference_only=True → [difference] only ⭐
```

### GP Prior 옵션
```
vpl_use_gp_prior
  ├─ False → Standard normal prior N(0, I)
  └─ True → Mixture prior (다른 클라이언트들의 분포)
      └─ vpl_gp_temperature: Gumbel-Softmax temperature
```

### Orthogonal Loss 옵션
```
vpl_orthogonal_weight > 0
  ├─ vpl_use_manual_orthogonal_labels
  │   ├─ True → Manual labels (서버가 수동 할당)
  │   └─ False → K-means labels (서버가 자동 할당) ⭐
  ├─ vpl_num_prototypes: k for k-means
  └─ vpl_prototype_scale: 원점으로부터 거리
```

## 권장 설정

### Preference-Only Learning (같은 preference 클라이언트 그룹화)
```yaml
llm:
  vpl_use_difference_only: True  # ⭐ General information 제거
  vpl_use_gp_prior: True
  vpl_orthogonal_weight: 10.0
  vpl_prototype_scale: 5.0
```

### General + Preference Learning (전체 정보 사용)
```yaml
llm:
  vpl_use_difference_only: False  # [chosen, rejected, difference] 모두 사용
  vpl_use_gp_prior: True
  vpl_orthogonal_weight: 10.0
  vpl_prototype_scale: 5.0
```

## 참고

- **VPL Documentation**: `docs/VPL_DOCUMENTATION.md`
- **GP Prior Documentation**: `docs/GP_PRIOR_DOCUMENTATION.md`
- **Orthogonal Loss Documentation**: `docs/ORTHOGONAL_LOSS_DOCUMENTATION.md`
- **Complete Implementation**: `docs/VPL_GP_ORTHOGONAL_COMPLETE_IMPLEMENTATION.md`
- **Z Embedding Generation**: `Z_EMBEDDING_GENERATION_EXPLANATION.md`
