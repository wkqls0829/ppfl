# Unseen Client Experiment (Qwen 2)

## 개요

이 실험은 VPL-GP가 학습 중에 보지 못한 클라이언트(unseen clients)에 대해 얼마나 잘 적응할 수 있는지 테스트합니다.

## 실험 구조

### 클라이언트 구성
- **총 20개 클라이언트**: 10개 harmless + 10개 helpful
- **Training 클라이언트 (10개)**: Client ID 1-10 (5개 harmless + 5개 helpful)
  - Selector training에만 참여
- **Unseen 클라이언트 (10개)**: Client ID 11-20 (5개 harmless + 5개 helpful)
  - Selector training에는 참여하지 않음
  - RL evaluation에만 참여하여 adaptation 능력 테스트

### 실험 단계

1. **Selector Training (HHST)**
   - Training 클라이언트 (1-10)만 참여
   - Unseen 클라이언트 (11-20)는 데이터는 생성되지만 학습에는 참여하지 않음
   - 각 알고리즘별로 selector checkpoint 생성

2. **RL Training (HRL)**
   - 모든 20개 클라이언트가 evaluation에 참여
   - Seen 클라이언트 (1-10): Selector checkpoint에서 z 값 로드
   - Unseen 클라이언트 (11-20): Training data로부터 z 값 계산
     - Selector의 variational encoder와 feature extractor 사용
     - Binary selector와 동일한 방식으로 z 계산 후 평균

## 알고리즘

다음 4개 알고리즘을 테스트합니다:

1. **FedBiscuit**: Baseline reward model 기반
2. **FedDPO**: Direct Preference Optimization (selector 불필요)
3. **FedVPL**: Variational Preference Learning (standard normal prior)
4. **FedVPL-GP-Ortho**: VPL with Gumbel-Softmax Prior + Orthogonal Loss

## 모델 및 하이퍼파라미터

- **모델**: Qwen/Qwen2-0.5B@huggingface_llm
- **하이퍼파라미터**: Main table 표준 설정 사용
- **클라이언트 수**: 20 (10 harmless + 10 helpful)

## 실험 TID 및 GPU 할당

### Selector Training
- **TID 70000** (FedBiscuit): GPU 6
- **TID 70002** (FedVPL): GPU 6
- **TID 70003** (FedVPL-GP-Ortho): GPU 7

### RL Training
- **TID 70001** (FedDPO): GPU 7 (selector 불필요)
- **TID 70004** (FedBiscuit RL): GPU 6 (selector TID 70000 필요)
- **TID 70005** (FedVPL RL): GPU 6 (selector TID 70002 필요)
- **TID 70006** (FedVPL-GP-Ortho RL): GPU 7 (selector TID 70003 필요)

## 실행 방법

### 1. Selector Training 실행

```bash
# FedBiscuit selector
bash scripts/unseen-qwen/hhst_fedbiscuit.sh

# FedVPL selector
bash scripts/unseen-qwen/hhst_fedvpl.sh

# FedVPL-GP-Ortho selector
bash scripts/unseen-qwen/hhst_fedvplgp_ortho.sh
```

### 2. RL Training 실행

Selector training이 완료된 후 RL training을 실행합니다.

```bash
# FedDPO (selector 불필요)
bash scripts/unseen-qwen/hrl_feddpo.sh

# FedBiscuit RL (selector TID 70000 필요)
bash scripts/unseen-qwen/hrl_fedbiscuit.sh

# FedVPL RL (selector TID 70002 필요)
bash scripts/unseen-qwen/hrl_fedvpl.sh

# FedVPL-GP-Ortho RL (selector TID 70003 필요)
bash scripts/unseen-qwen/hrl_fedvplgp_ortho.sh
```

## Config 파일

각 알고리즘별 config 파일 위치:

- `cfg/fedbiscuit-unseen/hhst.yaml`, `cfg/fedbiscuit-unseen/hrl.yaml`
- `cfg/feddpo-unseen/hrl.yaml` (selector 없음)
- `cfg/fedvpl-unseen/hhst.yaml`, `cfg/fedvpl-unseen/hrl.yaml`
- `cfg/fedvpl-gp-ortho-unseen/hhst.yaml`, `cfg/fedvpl-gp-ortho-unseen/hrl.yaml`

## 주요 설정

### Selector Training Config
- `federate.client_num: 20`: 총 20개 클라이언트 생성
- `federate.unseen_clients_id: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]`: Unseen 클라이언트 ID 명시
- `federate.sample_client_num: 5`: 각 round마다 5개 클라이언트만 샘플링 (training 클라이언트 중에서)

### RL Training Config
- `federate.client_num: 20`: 모든 20개 클라이언트가 evaluation에 참여
- Unseen client z 계산: `standalone_training.py`의 `_compute_unseen_client_z_from_training_data()` 함수 사용

## Unseen Client Z 계산

RL training 초기화 시, unseen 클라이언트의 z 값을 계산합니다:

1. Unseen 클라이언트의 training data 로드
2. Selector checkpoint에서 variational encoder와 feature extractor 로드
3. Training data를 binary selector와 동일한 방식으로 처리:
   - LLMDataset으로 변환
   - Selector model forward pass로 hidden states/logits 추출
   - Feature extractor로 preference features 추출
   - Variational encoder로 z_mu 계산
   - 모든 batch의 z_mu를 평균하여 client-specific z 계산
4. `client_average_z_dict`에 저장하여 RL training에서 사용

## Evaluation

RL evaluation 시:
- **Seen 클라이언트 (1-10)**: 기존 WandB metrics (`test_harmless`, `test_helpful`) 사용
- **Unseen 클라이언트 (11-20)**: 별도로 측정하여 WandB에 로깅
  - `Server_Seen, {metric}`: Seen 클라이언트 성능
  - `Server_Unseen, {metric}`: Unseen 클라이언트 성능

## Checkpoint 위치

- **Local server**: `/hdd/hdd3/kjb/checkpoints/`
- **Cluster**: `$WORK_DIR/checkpoints/`

## WandB 프로젝트

- **Selector Training**: `fvpl-unseen-selector`
- **RL Training**: `fvpl-unseen-rl`

## 주의사항

1. **메모리 관리**: 여러 실험을 동시에 실행할 때는 GPU 메모리 사용량을 고려해야 합니다.
2. **Selector 의존성**: FedBiscuit, FedVPL, FedVPL-GP-Ortho의 RL training은 해당 selector checkpoint가 필요합니다.
3. **FedDPO**: Selector가 필요하지 않으므로 독립적으로 실행 가능합니다.

## 실험 목적

이 실험을 통해 다음을 확인할 수 있습니다:

1. VPL-GP가 학습 중 보지 못한 클라이언트에 대해 얼마나 잘 적응하는지
2. Unseen 클라이언트의 z 분포를 training data로부터 계산하는 방법의 효과
3. 각 알고리즘(FedBiscuit, FedDPO, FedVPL, FedVPL-GP-Ortho)의 unseen client adaptation 성능 비교
