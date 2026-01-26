# HRL VPL 테스트 가이드

이 문서는 HRL (Harmlessness-Helpfulness RLHF)에서 VPL (Variational Preference Learning) 기능을 테스트하는 방법을 설명합니다.

## 생성된 파일

1. **`cfg/vpl/hrl.yaml`**: VPL을 사용한 RLHF 학습 설정 파일
2. **`scripts/test_hrl_vpl.sh`**: VPL 테스트 실행 스크립트
3. **`scripts/verify_vpl_setup.sh`**: 설정 검증 스크립트

## 구현된 VPL 기능

### 1. VPL Components 로딩
- **위치**: `federatedscope/llm/rlhf/standalone_training.py::_init_vpl_components()`
- **기능**: 
  - Selector checkpoint에서 `variational_encoder`, `latent_projection`, `z_to_embedding` 로드
  - Selector가 VPL trainer를 사용하는 경우 자동으로 초기화
- **확인 방법**: 로그에서 "Selector uses VPL trainer, initializing VPL components..." 메시지 확인

### 2. Client-specific Latent Z 추론
- **위치**: `federatedscope/llm/rlhf/standalone_training.py::_infer_client_latent_z()`
- **기능**:
  - 각 라운드의 학습 데이터로부터 client-specific latent vector `z` 추론
  - 이전 라운드의 `_last_train_dict`를 사용하여 추론
- **확인 방법**: 로그에서 "Inferred client-specific latent z (Round #N): shape=..." 메시지 확인

### 3. Conditional Generation with Z Embedding
- **위치**: `federatedscope/llm/rlhf/standalone_training.py::_generate_with_z_embedding()`
- **기능**:
  - 추론된 `z`를 embedding dimension으로 projection (`z_to_embedding`)
  - Generation model의 input embeddings에 `z`를 직접 injection
  - Client-specific preference를 반영한 조건부 생성
- **확인 방법**: 로그에서 "Using embedding injection with client z for batch..." 메시지 확인

### 4. VPL-based Preference Selection
- **위치**: `federatedscope/llm/rlhf/standalone_training.py::_choose_better_response()`
- **기능**:
  - VPL forward pass를 사용하여 preference 선택
  - `_extract_preference_features_vpl()`로 preference features 추출
  - Variational encoder로 `z` 추론 후 latent projection으로 logit 조정
  - Conditioned logits로 더 나은 응답 선택
- **확인 방법**: 로그에서 "VPL forward pass" 또는 "Extracted preference features for VPL" 메시지 확인

## 설정 파일 구조

### Selector Config (`cfg/vpl/hhst.yaml`)
```yaml
trainer:
  type: vplrewardchoicetrainer  # VPL trainer 사용
llm:
  vpl_latent_dim: 32
  vpl_kl_weight: 0.1
  vpl_feature_method: 'choice_logits'
```

### RLHF Config (`cfg/vpl/hrl.yaml`)
```yaml
llm:
  rlhf: True
  vpl_latent_dim: 32  # Selector와 동일해야 함
  vpl_feature_method: 'choice_logits'  # Selector와 동일해야 함
trainer:
  type: llmdporewardtrainer  # DPO trainer (policy 학습용)
regenerate_data_freq: 1  # 매 라운드 데이터 재생성 (z 업데이트 반영)
```

## 실행 방법

### 1. 설정 검증
```bash
bash scripts/verify_vpl_setup.sh
```

### 2. 테스트 실행
```bash
# 스크립트 수정 (CUDA_VISIBLE_DEVICES, device 등)
vim scripts/vpl/test_hrl.sh

# 실행
./scripts/vpl/test_hrl.sh

# 로그 모니터링
tail -f outputs/20100.log
```

## 확인해야 할 로그 메시지

### 정상 작동 확인
1. **VPL 초기화**:
   ```
   Selector uses VPL trainer, initializing VPL components for inference...
   Initialized z_to_embedding projection: 32 -> 2048
   Loaded variational_encoder from ...
   VPL components initialized: latent_dim=32, feature_method=choice_logits
   ```

2. **Z 추론**:
   ```
   Inferred client-specific latent z (Round #1): shape=torch.Size([1, 32])
   ```

3. **Conditional Generation**:
   ```
   Using embedding injection with client z for batch 1
   ```

4. **VPL-based Selection**:
   ```
   Using VPL forward pass for preference selection
   Extracted preference features for VPL: shape=torch.Size([batch, 4])
   ```

### 오류 확인
- `variational_encoder`가 None인 경우: Selector checkpoint에 VPL components가 저장되지 않았을 수 있음
- `z_to_embedding`이 None인 경우: Embedding dimension을 찾지 못했을 수 있음
- `client_z`가 None인 경우: 이전 라운드 데이터가 없거나 추론 실패

## 주요 차이점: HRL vs HRL VPL

| 기능 | HRL (기존) | HRL VPL |
|------|-----------|---------|
| Selector Trainer | `llmrewardchoicetrainer` | `vplrewardchoicetrainer` |
| Preference Selection | Standard forward pass | VPL forward pass (z-conditioned) |
| Data Generation | Unconditional | Conditional on client z |
| Client Personalization | 없음 | Latent z 기반 |
| Evaluation | Global metrics | Personalized metrics (optional) |

## 메모리 최적화

VPL 사용 시 추가 메모리가 필요할 수 있습니다:
- `variational_encoder`: 약 1-2MB
- `latent_projection`: 약 1KB
- `z_to_embedding`: 약 100KB

OOM 발생 시:
1. `generation_batch_size` 감소 (현재: 3)
2. `dataloader.batch_size` 감소 (현재: 1)
3. `grad_accum_step` 증가 (현재: 32)

## 문제 해결

### 문제: VPL components가 로드되지 않음
**해결**: Selector checkpoint에 VPL components가 저장되어 있는지 확인
- Selector를 VPL trainer로 학습했는지 확인
- `VPLRewardChoiceTrainer.save_model()`이 제대로 호출되었는지 확인

### 문제: client_z가 None
**해결**: 
- 첫 번째 라운드에서는 `_last_train_dict`가 없어서 정상
- 두 번째 라운드 이후에도 None이면 로그에서 추론 오류 확인

### 문제: Conditional generation이 작동하지 않음
**해결**:
- `z_to_embedding`이 초기화되었는지 확인
- `client_z`가 None이 아닌지 확인
- 로그에서 "Using embedding injection" 메시지 확인
