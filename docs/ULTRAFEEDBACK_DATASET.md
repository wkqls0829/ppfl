# UltraFeedback Dataset Implementation

## Overview

UltraFeedback는 4가지 preference dimension (helpfulness, honesty, instruction_following, truthfulness)에 대한 multi-dimensional preference 데이터셋입니다. 이 구현은 conflicting preference pairs를 추출하여 federated learning 환경에서 각 클라이언트가 서로 다른 preference dimension을 선호하도록 데이터를 분배합니다.

## Dataset Structure

UltraFeedback 데이터셋은 HuggingFace의 `openbmb/UltraFeedback`에서 로드됩니다. 각 샘플은:
- `instruction`: 사용자 프롬프트
- `completions`: 여러 모델의 응답 리스트
  - 각 completion은 `response` 필드와 `annotations` 딕셔너리를 포함
  - `annotations`에는 각 dimension별 점수 (`helpfulness`, `honesty`, `instruction_following`, `truthfulness`)
  - `overall_score` 또는 `fine-grained_score`로 전체 점수 제공

## Data Processing

### Conflicting Pairs Extraction

1. **Threshold Filtering**: 기본적으로 score difference >= 3.0인 conflicting pairs만 사용
   - Config에서 `data.ultrafeedback_threshold`로 조정 가능 (기본값: 3.0)

2. **Conflict Detection**:
   - 각 샘플의 completions를 `overall_score` 기준으로 정렬
   - Best completion과 다른 completion들을 비교
   - Best가 일부 dimension에서 더 좋고, 다른 completion이 다른 dimension에서 더 좋은 경우를 conflicting pair로 식별
   - 가장 큰 차이를 보이는 dimension을 `winning_dim`으로 할당

3. **Data Format**:
   - `prompt`: 사용자 프롬프트
   - `output_A`: Best response (chosen)
   - `output_B`: Conflicting response (rejected)
   - `choice`: " A" (항상 best가 A)
   - `winning_dim`: 가장 큰 차이를 보이는 dimension

## Client Distribution

### Equal Distribution Strategy

UltraFeedback은 균등 분배(equal distribution) 전략을 사용합니다:

#### 10 Clients
- **helpfulness**: 3 clients (client_id: 1-3)
- **honesty**: 3 clients (client_id: 4-6)
- **instruction_following**: 2 clients (client_id: 7-8)
- **truthfulness**: 2 clients (client_id: 9-10)

#### 20 Clients
- **helpfulness**: 5 clients (client_id: 1-5)
- **honesty**: 5 clients (client_id: 6-10)
- **instruction_following**: 5 clients (client_id: 11-15)
- **truthfulness**: 5 clients (client_id: 16-20)

#### Other Client Counts
- `client_num // 4` per dimension (나머지는 순서대로 분배)

### Data Splitting

각 dimension의 데이터는:
1. Train/Test split (80/20, seed=42)
2. 각 dimension 내에서 `shard()` 메서드를 사용하여 클라이언트별로 분배

## Usage

### Selector Training

```yaml
data:
  type: ultrafeedback
  ultrafeedback_threshold: 3.0  # Optional: default is 3.0
  max_train_samples: -1  # Optional: limit training samples
  max_test_samples: -1   # Optional: limit test samples

federate:
  client_num: 10  # or 20 for equal distribution (5,5,5,5)
```

### RLHF Training

```yaml
data:
  type: ultrafeedback
  ultrafeedback_threshold: 3.0
  split_by_client: true

federate:
  client_num: 10  # Must be 10 or 20 for equal distribution
```

## Evaluation Metrics

UltraFeedback 데이터셋에 대해 다음 win rate 메트릭을 사용할 수 있습니다:

- `instruction_following_winrate`: Instruction following dimension에 대한 win rate (클라이언트 7-8 또는 11-15)
- `truthfulness_winrate`: Truthfulness dimension에 대한 win rate (클라이언트 9-10 또는 16-20)

```yaml
eval:
  metrics: ['instruction_following_winrate', 'truthfulness_winrate']
```

## Files

- **Data Loader**: `federatedscope/llm/dataloader/ultrafeedback.py`
  - `load_ultrafeedback_dataset()`: Selector training용
  - `load_ultrafeedback_for_rlhf()`: RLHF training용

- **Federated Data Loader**: `federatedscope/src/data/load_ultrafeedback.py`
  - `load_ultrafeedback_data()`: Federated training용 클라이언트 분배

- **Evaluation Metrics**: `federatedscope/llm/metric/winrate_metrics.py`
  - `eval_instruction_following_winrate()`
  - `eval_truthfulness_winrate()`

## Notes

- UltraFeedback은 HuggingFace에서 자동으로 다운로드됩니다
- Threshold 3.0 이상의 conflicting pairs만 사용하여 preference conflict를 명확히 보여줍니다
- 각 클라이언트는 하나의 preference dimension에 특화된 데이터를 받습니다
- Test evaluation 시 각 클라이언트는 자신의 preference dimension에 맞는 평가를 수행합니다
