# a0ef4df 커밋 이후 변경사항 정리

**기준 커밋**: `a0ef4df` (2026-01-22 16:27:35) - "Fix client_average_z_dict loading and t-SNE visualization for RL training"

**최종 업데이트**: 2026-01-23 11:30 (현재 상황 및 최근 버그 수정 포함)

## 📝 수정된 파일 (3개)

### 1. `federatedscope/llm/llm_local/server.py`
**주요 변경사항:**
- **0 라운드 Evaluation 추가**: 학습 전 baseline win rate를 측정하기 위해 0 라운드에서 evaluation 실행
  - `__init__` 메서드에 0 라운드 evaluation 로직 추가
  - win rate 메트릭이 설정된 경우 자동으로 실행
- **VPL Components Aggregation 추가**: 
  - VPL components (variational_encoder, feature_extractor, latent_projection, z_to_embedding)를 별도로 수집 및 집계
  - Weighted average로 aggregation 수행
  - Aggregator에 저장하여 checkpoint에 포함

### 2. `federatedscope/llm/rlhf/fedserver.py`
**주요 변경사항:**
- Import 변경: `standalone_training` → `standalone_training_bon`
  - `standalone_training.py`가 삭제되어 `standalone_training_bon.py` 사용

### 3. `federatedscope/llm/rlhf/main.py`
**주요 변경사항:**
- Import 변경: `standalone_training` → `standalone_training_bon`
  - `standalone_training.py`가 삭제되어 `standalone_training_bon.py` 사용
- **복원됨**: Git 히스토리에서 파일 복원 (원래 있었던 파일)

## ➕ 새로 생성된 파일

### Config 파일
1. **`cfg/fedbiscuit/hrl-21023.yaml`**
   - FedBiscuit RLHF 실험 설정 (21023)
   - 20023 checkpoint 사용
   - Variational 요소 없음
   - GPU 3 사용

2. **`cfg/fedbiscuit/hrl-21024.yaml`**
   - FedBiscuit RLHF 실험 설정 (21024)
   - 20023 checkpoint 사용
   - Variational 요소 없음
   - GPU 3 사용

### Script 파일
1. **`scripts/fedbiscuit/hrl-21023.sh`**
   - 21023 실험 실행 스크립트
   - conda biscuit 환경 사용
   - 0 라운드 evaluation 포함

2. **`scripts/fedbiscuit/hrl-21024.sh`**
   - 21024 실험 실행 스크립트
   - conda biscuit 환경 사용

## 🗑️ 삭제된 파일 (64개)

### RLHF 관련 파일
- `federatedscope/llm/rlhf/load_vpl_components.py` - VPL components 로딩 유틸리티
- `federatedscope/llm/rlhf/standalone_training.py` - standalone_training_bon.py로 대체
- `federatedscope/llm/rlhf/variational_selector.py` - Variational selector

### Config 파일들 (22개)
- `cfg/fedbiscuit/`: hhst-20000.yaml, hhst-20023.yaml, hrl-20000.yaml, hrl.yaml
- `cfg/feddpo/`: hhst-10000.yaml, hrl-10000.yaml
- `cfg/fedvpl/`: hhst-30000.yaml, hrl-30000.yaml
- `cfg/vpl-gp/`: 여러 hhst, hrl config 파일들

### 문서 파일들 (14개)
- `docs/`: VPL, GP Prior, Orthogonal Loss 등 관련 문서들
- `documents/README.md`

### Core 파일들
- `federatedscope/core/aggregators/`: aggregator.py, clients_avg_aggregator.py
- `federatedscope/core/auxiliaries/logging.py`
- `federatedscope/core/monitors/metric_calculator.py`
- `federatedscope/core/workers/`: client.py, server.py

### LLM 관련 파일들
- `federatedscope/llm/dataset/llm_dataset.py`
- `federatedscope/llm/llm_local/`: aggregator.py, client.py
- `federatedscope/llm/metric/`: 여러 metric 파일들
- `federatedscope/llm/model/`: adapter_builder.py, variational_encoder.py
- `federatedscope/llm/reward/`: reward 관련 파일들
- `federatedscope/llm/trainer/`: 여러 trainer 파일들

## 📊 통계

- **수정된 파일**: 3개
- **새로 생성된 파일**: 4개 (config 2개, script 2개)
- **삭제된 파일**: 64개
- **총 변경 라인**: +93, -17,001

## 🔍 주요 작업 내용

1. **0 라운드 Evaluation 기능 추가**
   - Win rate가 100%로 나오는 문제 해결을 위해 baseline 측정
   - 학습 전 모델 성능 확인 가능

2. **VPL Components 저장 기능 추가**
   - VPL components를 checkpoint에 저장
   - Client average z distribution 저장

3. **FedBiscuit RLHF 실험 설정**
   - 21023, 21024 실험을 위한 config 및 script 생성
   - Variational 요소 없는 FedBiscuit 알고리즘 사용

4. **파일 정리**
   - 불필요한 config 파일들 삭제
   - 문서 파일들 정리
   - 사용하지 않는 코드 파일들 삭제

## ⚠️ 주의사항

- 많은 파일들이 삭제되었지만, 이는 작업 디렉토리에서의 변경사항이며 아직 커밋되지 않음
- 일부 파일들은 다른 위치로 이동했거나 다른 이름으로 변경되었을 수 있음
- `main.py`는 복원되었지만, 다른 삭제된 파일들(`standalone_training.py`, `load_vpl_components.py`, `variational_selector.py`)은 아직 복원되지 않음

---

## 📝 최근 추가 변경사항 (2026-01-23)

### 1. Feature Extractor 아키텍처 불일치 수정
**파일**: `federatedscope/llm/rlhf/load_vpl_components.py`
- **문제**: Selector checkpoint와 RL config 간 feature extractor 아키텍처 불일치
  - Selector training: `vpl_use_llm_feature_extractor=True` → `[6144 -> 512 -> 256 -> 128]`
  - RL loading: 항상 `[raw_feature_dim -> 256 -> 512 -> 256 -> 128]` 구조 사용
- **해결**: `vpl_use_llm_feature_extractor` 설정에 따라 아키텍처 선택
  - `True`: `[raw_feature_dim -> 512 -> 256 -> 128]` (selector training과 일치)
  - `False`: `[raw_feature_dim -> 256 -> 512 -> 256 -> 128]` (MLP feature extractor)

### 2. VPL Config 불일치 수정
**파일**: `cfg/vpl-gp/hrl-ortho-51000.yaml`
- **문제**: Selector checkpoint (50000)와 RL config (51000) 간 `vpl_use_difference_only` 설정 불일치
  - 50000: `vpl_use_difference_only` 없음 (기본값 False) → feature extractor 입력: 6144 (2048×3)
  - 51000: `vpl_use_difference_only: True` → feature extractor 입력: 2048
- **해결**: 51000 config를 50000과 일치하도록 수정 (`vpl_use_difference_only: False`)

### 3. OS Import Shadowing 문제 수정
**파일**: `federatedscope/llm/rlhf/standalone_training.py`
- **문제**: `train` 함수 내부에서 `import os`를 다시 선언하여 `os`가 로컬 변수로 shadowing됨
  - 820번째 줄에서 `os.path.split` 사용 시 `UnboundLocalError` 발생
- **해결**: 함수 내부의 중복 `import os` 제거 (파일 상단에서 이미 import됨)

### 4. 모든 HRL 스크립트 설정 통일
**파일들**: 모든 `cfg/*/hrl*.yaml` 파일들
- **변경사항**: 
  - `total_round_num: 30` → `50`
  - `local_update_steps: 10` → `30`
- **영향받는 파일들**:
  - `cfg/vpl-gp/hrl.yaml`
  - `cfg/vpl-gp/hrl-ortho.yaml`
  - `cfg/vpl-gp/hrl-ortho-51000.yaml`
  - `cfg/vpl-gp/hrl-ortho-51022.yaml`
  - `cfg/vpl-gp/hrl-ortho-51024.yaml`
  - `cfg/fedbiscuit/hrl.yaml`
  - `cfg/fedbiscuit/hrl-20000.yaml`
  - `cfg/fedvpl/hrl-30000.yaml`
  - `cfg/feddpo/hrl-10000.yaml`
- **참고**: `cfg/fedbiscuit/hrl-21023.yaml`과 `cfg/fedbiscuit/hrl-21024.yaml`은 이미 50/30으로 설정되어 있음

### 5. Test Evaluation 캐시 문제 수정 (2026-01-23 11:20)
**파일들**: 
- `federatedscope/llm/metric/hhrl_metrics.py`
- `federatedscope/llm/metric/winrate_metrics.py`

**문제**: 
- Test evaluation에서 reward model score가 변하지 않고 winrate가 항상 높게 나오는 문제
- 원인: Test evaluation에서 캐시를 사용하여 매 round마다 새로운 response를 생성하지 않음
- Winrate 계산 시 prompt 매칭이 인덱스 기반으로만 되어 잘못된 비교 발생

**해결**:
- Test split에서는 캐시를 사용하지 않도록 수정 (매 round마다 새로운 response 생성)
- Winrate 계산 시 prompt 텍스트 기반 정확한 매칭으로 변경
- `prompt_to_original_idx` 딕셔너리를 사용하여 정확한 prompt 매칭

### 6. Variational Selector Choice Key 누락 버그 수정 (2026-01-23 11:25)
**파일**: `federatedscope/llm/rlhf/variational_selector.py`

**문제**:
- `use_provided_z=True`일 때 z 값은 사용하지만 실제 선택(choice) 로직이 실행되지 않음
- `predicted_indices`가 비어있어서 `choice` key가 생성되지 않음
- 에러: "Preference data sample missing 'choice' key"

**해결**:
- `use_provided_z=True`일 때도 제공된 z 값을 사용하여 실제 선택을 수행하도록 수정
- 제공된 z 값을 사용한 binary selection 로직 추가

### 7. TOKENIZERS_PARALLELISM 경고 수정 (2026-01-23 11:30)
**파일**: `scripts/fedbiscuit/hrl-21000.sh`

**문제**:
- "The current process just got forked, after parallelism has already been used" 경고가 계속 발생

**해결**:
- `TOKENIZERS_PARALLELISM=false` 환경 변수 추가

### 8. 추가 수정사항
- **Feature extractor 로딩 경고 해결**: Shape mismatch 문제 해결로 feature extractor가 정상적으로 로드됨
- **실험 51000 정상 실행**: 모든 에러 수정 후 실험이 정상적으로 진행 중

---

## 🖥️ 현재 실행 중인 실험 현황 (2026-01-23 11:30)

### GPU 0 (44.3 GB / 49.1 GB 사용, 97%)
- **PID 2340462**: FedBiscuit HHST (TID 20124)
  - Config: `cfg/fedbiscuit/hhst-20124.yaml`
  - 50 clients baseline 비교 실험

### GPU 1 (6.7 GB / 49.1 GB 사용, 0%)
- **PID 2902403**: VPL-GP HHST (TID 50124)
  - Config: `cfg/vpl-gp/hhst-ortho-50124.yaml`
  - 50 clients, orthogonal loss 활성화

### GPU 2 (12.2 GB / 49.1 GB 사용, 42%)
- **PID 2942114**: VPL-GP RL (TID 51000)
  - Config: `cfg/vpl-gp/hrl-ortho-51000.yaml` (selector: `hhst-ortho-50000.yaml`)
  - Test HRL 실험 (50000 selector checkpoint 사용)
  - Variational generation 및 selection 활성화

### GPU 3 (비어있음)
- 사용 가능

### GPU 4 (12.6 GB / 49.1 GB 사용, 39%)
- **PID 2956476 등**: FedBiscuit RL (TID 21000)
  - Config: `cfg/fedbiscuit/hrl-21000.yaml` (selector: `hhst-20023.yaml`)
  - Test RL 실험 (매 round 평가, 10 rounds)
  - TOKENIZERS_PARALLELISM 경고 수정됨

### GPU 5-7
- 다른 프로세스 (VLLM 등) 실행 중

---

## 📝 최근 수정된 파일 요약 (2026-01-23)

1. **`federatedscope/llm/metric/hhrl_metrics.py`**
   - Test evaluation 캐시 비활성화
   - 매 round마다 새로운 response 생성

2. **`federatedscope/llm/metric/winrate_metrics.py`**
   - Test evaluation 캐시 비활성화
   - Prompt 텍스트 기반 정확한 매칭으로 변경
   - GPT API 버전과 internal model 버전 모두 수정

3. **`federatedscope/llm/rlhf/variational_selector.py`**
   - `use_provided_z=True`일 때도 선택 로직 실행하도록 수정
   - Choice key 생성 보장

4. **`scripts/fedbiscuit/hrl-21000.sh`**
   - `TOKENIZERS_PARALLELISM=false` 환경 변수 추가

5. **`cfg/fedbiscuit/hrl-21000.yaml`**
   - Device를 GPU 4로 변경 (1 → 4)
