# VPL Components Aggregation 설명

## 개요

**VPL Components Aggregation**은 Federated Learning에서 VPL (Variational Preference Learning) 모델의 구성 요소들을 클라이언트들로부터 수집하여 서버에서 집계(aggregate)하는 기능입니다.

## VPL Components란?

VPL 모델은 사용자별 선호도를 잠재 변수(latent variable) `z`로 모델링합니다. 이를 위해 다음 4가지 주요 구성 요소가 필요합니다:

### 1. **Variational Encoder** (`variational_encoder`)
- **역할**: 사용자의 선호 데이터로부터 잠재 변수 `z`의 분포를 추론
- **입력**: 선호 비교 데이터 (예: RESPONSE A vs RESPONSE B)
- **출력**: `z`의 평균(μ)과 분산(σ²) 파라미터
- **수식**: `q_φ(z | preferences) = N(μ, σ²)`

### 2. **Feature Extractor** (`feature_extractor`)
- **역할**: LLM의 hidden states에서 선호도 관련 특징을 추출
- **입력**: LLM의 hidden representations (예: `h_chosen - h_rejected`)
- **출력**: 선호도 특징 벡터
- **목적**: 일반적인 내용 정보를 제거하고 선호도 특정 정보만 추출

### 3. **Latent Projection** (`latent_projection`)
- **역할**: 잠재 변수 `z`를 모델의 logits에 반영하기 위한 projection
- **입력**: 잠재 변수 `z` (latent_dim 차원)
- **출력**: Logit 조정값
- **수식**: `logits(s_A, s_B | z) = logits_base(s_A, s_B) + f_θ(z)`

### 4. **Z to Embedding** (`z_to_embedding`)
- **역할**: 잠재 변수 `z`를 embedding space로 변환
- **입력**: 잠재 변수 `z`
- **출력**: Embedding 벡터
- **용도**: RLHF에서 conditional generation 등에 사용

## 왜 별도로 Aggregation이 필요한가?

### 문제점
기존 Federated Learning에서는 주로 **모델 파라미터만** 집계합니다. 하지만 VPL의 경우:
1. VPL components는 모델의 일부이지만, 일반 모델 파라미터와는 다른 방식으로 처리되어야 함
2. 클라이언트별로 학습된 VPL components를 서버에서 통합해야 함
3. Checkpoint에 저장하여 나중에 로드할 수 있어야 함

### 해결책: VPL Components Aggregation
1. **수집**: 각 클라이언트로부터 VPL components 파라미터를 별도로 수집
2. **집계**: Sample size 기반 weighted average로 집계
3. **저장**: `aggregator.vpl_components`에 저장하여 checkpoint에 포함

## 구현 세부사항

### 1. 초기화 (`__init__`)
```python
# Initialize VPL components storage in aggregator
if hasattr(self, 'aggregator'):
    if not hasattr(self.aggregator, 'vpl_components'):
        self.aggregator.vpl_components = {}
```

### 2. 수집 (`_perform_federated_aggregation`)
```python
# Extract VPL components from client model parameters
vpl_component_keys = []
for key in model_para.keys():
    if any(comp in key for comp in ['variational_encoder', 'feature_extractor', 
                                     'latent_projection', 'z_to_embedding']):
        vpl_component_keys.append(key)

# Collect VPL components with sample sizes
for key in vpl_component_keys:
    vpl_components_dict[key].append((sample_size, model_para[key]))
```

### 3. 집계 (Weighted Average)
```python
# Weighted average by sample size
total_weight = sum(sample_size for sample_size, _ in client_values)
for sample_size, value in client_values:
    weight = sample_size / total_weight
    weighted_value = value * weight
    avg_value = avg_value + weighted_value
```

### 4. 저장
```python
# Store in aggregator for checkpoint saving
aggregator.vpl_components.update(aggregated_vpl_components)
```

## 사용 예시

### Checkpoint 저장 시
```python
# aggregator.vpl_components에 저장된 VPL components가
# checkpoint에 자동으로 포함됨
ckpt['model']['variational_encoder.weight'] = ...
ckpt['model']['feature_extractor.weight'] = ...
ckpt['model']['latent_projection.weight'] = ...
ckpt['model']['z_to_embedding.weight'] = ...
```

### Checkpoint 로드 시
```python
# VPL components를 로드하여 모델에 적용
variational_encoder.load_state_dict(ckpt['model']['variational_encoder'])
feature_extractor.load_state_dict(ckpt['model']['feature_extractor'])
# ...
```

## 장점

1. **통합 관리**: VPL components를 모델 파라미터와 함께 관리
2. **Checkpoint 지원**: 학습된 VPL components를 저장/로드 가능
3. **Federated Learning**: 클라이언트들의 VPL components를 효과적으로 통합
4. **RLHF 지원**: RLHF에서 VPL components를 사용하여 conditional generation/selection 가능

## 관련 파일

- `federatedscope/llm/llm_local/server.py`: Aggregation 로직
- `federatedscope/llm/trainer/vpl_reward_choice_trainer.py`: VPL trainer
- `federatedscope/llm/model/variational_encoder.py`: Variational encoder 구현
- `federatedscope/core/aggregators/clients_avg_aggregator.py`: Checkpoint 저장 로직
