# 실험 실행 가이드 (Experiment Execution Guide)

이 문서는 VPL-GP, FedVPL, FedDPO, FedBiscuit 실험을 실행하는 방법과 프로젝트 구조를 설명합니다.

## 목차

1. [프로젝트 구조](#프로젝트-구조)
2. [실험 ID 관리 체계](#실험-id-관리-체계)
3. [실험 실행 방법](#실험-실행-방법)
4. [스크립트 구조](#스크립트-구조)
5. [Configuration 파일 구조](#configuration-파일-구조)
6. [GPU 할당 방법](#gpu-할당-방법)
7. [로그 확인 및 모니터링](#로그-확인-및-모니터링)
8. [알고리즘별 실험 설정](#알고리즘별-실험-설정)

---

## 프로젝트 구조

```
/home/kjb/ppfl/
├── scripts/                    # 실행 스크립트
│   ├── vpl-gp/                # VPL-GP 실험 스크립트
│   │   ├── hhst-*.sh         # HHST (selector) 실험
│   │   └── hrl.sh            # HRL (RLHF) 실험
│   ├── fedvpl/                # FedVPL (naive VPL) 실험 스크립트
│   │   ├── hhst.sh
│   │   └── hrl.sh
│   ├── feddpo/                # FedDPO (DPO baseline) 실험 스크립트
│   │   ├── hhst.sh
│   │   └── hrl.sh
│   └── fedbiscuit/            # FedBiscuit (Multi-LoRA baseline) 실험 스크립트
│       ├── hhst.sh
│       └── hrl.sh
│
├── cfg/                       # Configuration 파일
│   ├── vpl-gp/                # VPL-GP 설정
│   │   ├── hhst-*.yaml       # HHST 실험 설정
│   │   └── hrl.yaml          # HRL 실험 설정
│   ├── fedvpl/                # FedVPL 설정
│   ├── feddpo/                # FedDPO 설정
│   └── fedbiscuit/            # FedBiscuit 설정
│
├── outputs/                   # 로그 파일
│   └── {tid}.log             # 실험 ID별 로그
│
├── exp/                       # 실험 결과 디렉토리
│   └── {expname}/            # 실험 이름별 결과
│       ├── cross_client_z_tsne_round_*.png  # t-SNE 시각화
│       └── sub_exp_*/        # 서브 실험 결과
│
├── /hdd/hdd3/kjb/checkpoints/ # 체크포인트 저장 경로
│   └── *.ckpt                # 모델 체크포인트
│
└── docs/                      # 문서
    ├── EXPERIMENT_GUIDE.md   # 이 문서
    ├── VPL_DOCUMENTATION.md  # VPL 설명
    ├── GP_PRIOR_DOCUMENTATION.md  # Gumbel-Softmax Prior 설명
    └── ORTHOGONAL_LOSS_DOCUMENTATION.md  # Orthogonal Loss 설명
```

---

## 실험 ID 관리 체계

실험 ID (Task ID, `tid`)는 각 실험을 고유하게 식별하는 숫자입니다.

### ID 범위

| 범위 | 알고리즘 | 설명 |
|------|----------|------|
| 10000-19999 | FedDPO | Direct Preference Optimization baseline |
| 20000-29999 | FedBiscuit | Multi-LoRA baseline |
| 30000-39999 | FedVPL | Naive VPL (no GP prior, no orthogonal loss) |
| 40000-49999 | VPL-GP (baseline) | VPL-GP without orthogonal loss |
| 50000-59999 | VPL-GP (orthogonal) | VPL-GP with orthogonal loss |

### 현재 실행 중인 실험 예시

| 실험 ID | GPU | KL Weight | Orthogonal Weight | Difference Only | 설명 |
|---------|-----|-----------|-------------------|------------------|------|
| 40001 | 0 | 1.0 | 0.0 | False | VPL-GP baseline, full embedding |
| 40100 | 0 | 0.1 | 0.0 | True | VPL-GP baseline, difference-only |
| 40101 | 2 | 1.0 | 0.0 | True | VPL-GP baseline, difference-only, 10x KL |
| 50001 | 1 | 1.0 | 100.0 | False | VPL-GP orthogonal, full embedding |
| 50002 | 2 | 10.0 | 1000.0 | False | VPL-GP orthogonal, 100x coefficients |
| 50100 | 1 | 0.1 | 10.0 | True | VPL-GP orthogonal, difference-only |
| 50101 | 3 | 1.0 | 100.0 | True | VPL-GP orthogonal, difference-only, 10x coefficients |

---

## 실험 실행 방법

### 1. 기본 실행 방법

```bash
# 스크립트 실행 권한 부여
chmod +x scripts/{algorithm}/{script_name}.sh

# 실험 실행
bash scripts/{algorithm}/{script_name}.sh
```

### 2. 예시: VPL-GP HHST 실험 실행

```bash
# 실험 40101 실행 (GPU 2)
bash scripts/vpl-gp/hhst-40101.sh
```

### 3. 실험 종료

```bash
# 특정 실험 ID의 프로세스 종료
ps aux | grep "python.*main.py.*{tid}" | grep -v grep | awk '{print $2}' | xargs kill -9

# GPU 2, 3에서 실행 중인 모든 실험 종료
ps aux | grep -E "python.*main.py.*(50002|30000|20000)" | grep -v grep | awk '{print $2}' | xargs kill -9
```

### 4. 실험 상태 확인

```bash
# 실행 중인 실험 확인
ps aux | grep "python.*main.py" | grep -v grep

# 특정 실험 ID 확인
ps aux | grep "python.*main.py.*40101" | grep -v grep
```

---

## 스크립트 구조

### 스크립트 템플릿

모든 실험 스크립트는 다음 구조를 따릅니다:

```bash
#!/bin/bash

# 실험 설명 주석
# 알고리즘, 설정, GPU 정보 등

tid={experiment_id}  # 실험 ID
# GPU는 config 파일에서 지정 (device: X)
# CUDA_VISIBLE_DEVICES는 설정하지 않음

# 환경 변수 설정
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

# PYTHONPATH 설정 (현재 프로젝트의 federatedscope 사용)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 실험 실행 (백그라운드)
nohup python -u federatedscope/main.py \
    --cfg cfg/{algorithm}/{config_file}.yaml \
    federate.save_to /hdd/hdd3/kjb/checkpoints/{checkpoint_name}_${tid}.ckpt \
    expname "{expname}_t${tid}" \
    > outputs/${tid}.log 2>&1 &

# 실행 정보 출력
echo "Training started (task ID: ${tid})"
echo "Config: cfg/{algorithm}/{config_file}.yaml"
echo "GPU: X (specified in config file: device: X)"
echo "Log file: outputs/${tid}.log"
echo "Monitor with: tail -f outputs/${tid}.log"
```

### 주요 스크립트 파일

#### VPL-GP 스크립트 (`scripts/vpl-gp/`)

- `hhst-{tid}.sh`: HHST (selector) 실험
  - `hhst-40001.sh`: Baseline, full embedding, KL=1.0
  - `hhst-40100.sh`: Baseline, difference-only, KL=0.1
  - `hhst-40101.sh`: Baseline, difference-only, KL=1.0
- `hhst-ortho-{tid}.sh`: Orthogonal loss 실험
  - `hhst-ortho-50001.sh`: Orthogonal, full embedding, KL=1.0, Orth=100.0
  - `hhst-ortho-50002.sh`: Orthogonal, full embedding, KL=10.0, Orth=1000.0
  - `hhst-ortho-50100.sh`: Orthogonal, difference-only, KL=0.1, Orth=10.0
  - `hhst-ortho-50101.sh`: Orthogonal, difference-only, KL=1.0, Orth=100.0
- `hrl.sh`: HRL (RLHF) 실험

#### FedVPL 스크립트 (`scripts/fedvpl/`)

- `hhst.sh`: HHST 실험 (tid: 30000)
- `hrl.sh`: HRL 실험 (tid: 30000)

#### FedDPO 스크립트 (`scripts/feddpo/`)

- `hhst.sh`: HHST 실험 (tid: 10000)
- `hrl.sh`: HRL 실험 (tid: 10000)

#### FedBiscuit 스크립트 (`scripts/fedbiscuit/`)

- `hhst.sh`: HHST 실험 (tid: 20000)
- `hrl.sh`: HRL 실험 (tid: 20000)

---

## Configuration 파일 구조

### Configuration 파일 위치

- `cfg/{algorithm}/{experiment_name}.yaml`

### 주요 설정 섹션

#### 1. 기본 설정

```yaml
use_gpu: True
device: 0  # GPU 번호 (0, 1, 2, 3)
backend: 'torch'
```

#### 2. Federated Learning 설정

```yaml
federate:
  mode: standalone
  client_num: 10              # 전체 클라이언트 수
  sample_client_num: 5        # 매 라운드 샘플링할 클라이언트 수
  total_round_num: 50         # 전체 라운드 수
  save_to: "/path/to/checkpoint.ckpt"
  save_freq: 20               # 체크포인트 저장 빈도
  share_local_model: True     # Single GPU 모드
  online_aggr: False
```

#### 3. 데이터 설정

```yaml
data:
  root: /hdd/hdd3/kjb/
  type: 'hh-rlhf@llm'         # 데이터셋 타입
  splits: [0.9,0.09,0.01]     # train/val/test 비율
  splitter: 'meta'            # 데이터 분할 방법
```

#### 4. VPL-GP 특화 설정

```yaml
llm:
  # VPL-GP 기본 설정
  vpl_use_gp_prior: True      # Gumbel-Softmax Prior 사용 여부
  vpl_latent_dim: 32          # Latent dimension
  vpl_kl_weight: 0.1          # KL divergence weight
  vpl_gp_temperature: 1.0     # Gumbel-Softmax temperature
  
  # Feature extraction 설정
  vpl_feature_method: 'choice_logits'  # 'choice_logits' 또는 embedding-based
  vpl_use_feature_difference: True      # Embedding difference 사용 여부
  vpl_use_difference_only: True         # Difference만 사용 (general information 제거)
  
  # Orthogonal loss 설정
  vpl_orthogonal_weight: 10.0          # Pull loss weight (0.0 = disabled)
  vpl_orthogonal_orthonorm_weight: 0.1  # Orthonormal constraint weight
  vpl_use_manual_orthogonal_labels: False  # Manual vs k-means labeling
  vpl_num_prototypes: 2                 # Orthogonal prototype 수
  vpl_prototype_scale: 5.0             # Prototype 거리 (원점으로부터)
  vpl_tsne_visualize_freq: 10          # t-SNE 시각화 빈도
```

#### 5. 모델 설정

```yaml
model:
  type: 'google/gemma-2b@huggingface_llm'
```

#### 6. 학습 설정

```yaml
train:
  local_update_steps: 30      # 클라이언트별 로컬 업데이트 스텝
  batch_or_epoch: batch
  optimizer:
    type: AdamW
    betas: (0.9, 0.95)
    lr: 0.00001
  is_enable_half: True        # Mixed precision training
```

#### 7. 평가 설정

```yaml
eval:
  freq: 5                      # 평가 빈도
  metrics: ['loss', 'acc', 'vpl_kl_loss', 'vpl_reconstruction_loss']
  best_res_update_round_wise_key: train_avg_loss
```

#### 8. WandB 설정

```yaml
wandb:
  use: True
  name_user: ''
  name_project: 'vpl-gp-selector'  # 또는 'vpl-gp-rl' (HRL의 경우)
  online_track: True
  client_train_info: True
```

### Configuration 파일 예시

#### VPL-GP Baseline (40101)

```yaml
# cfg/vpl-gp/hhst-40101.yaml
use_gpu: True
device: 2
llm:
  vpl_use_gp_prior: True
  vpl_kl_weight: 1.0
  vpl_orthogonal_weight: 0.0  # Disabled
  vpl_use_difference_only: True
```

#### VPL-GP with Orthogonal Loss (50101)

```yaml
# cfg/vpl-gp/hhst-ortho-50101.yaml
use_gpu: True
device: 3
llm:
  vpl_use_gp_prior: True
  vpl_kl_weight: 1.0
  vpl_orthogonal_weight: 100.0  # Enabled
  vpl_use_difference_only: True
```

---

## GPU 할당 방법

### 1. Configuration 파일에서 지정 (권장)

```yaml
# cfg/vpl-gp/hhst-40101.yaml
use_gpu: True
device: 2  # GPU 2 사용
```

### 2. 스크립트에서 환경 변수 사용 (비권장)

```bash
# 사용하지 않음 - config 파일이 우선
export CUDA_VISIBLE_DEVICES=2
```

### 3. GPU 사용 현황 확인

```bash
# nvidia-smi로 GPU 사용 현황 확인
nvidia-smi

# 특정 GPU에서 실행 중인 프로세스 확인
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv -i 2
```

### 4. GPU 할당 전략

- **Single GPU 모드**: `federate.share_local_model: True`
- **Multi-GPU 모드**: `federate.share_local_model: False`, `federate.process_num: {num_gpus}`

---

## 로그 확인 및 모니터링

### 1. 로그 파일 위치

- `outputs/{tid}.log`: 실험 ID별 로그 파일

### 2. 로그 실시간 확인

```bash
# 실시간 로그 확인
tail -f outputs/40101.log

# 마지막 100줄 확인
tail -n 100 outputs/40101.log

# 에러 검색
grep -i error outputs/40101.log
```

### 3. WandB 모니터링

#### 프로젝트별 WandB 프로젝트

- **Selector 실험**: `vpl-gp-selector`
- **RLHF 실험**: `vpl-gp-rl`

#### WandB에서 확인 가능한 메트릭

- **Client-level metrics**:
  - `vpl_total_loss`: 전체 VPL loss
  - `vpl_reconstruction_loss`: Reconstruction loss
  - `vpl_kl_loss`: KL divergence loss
  - `vpl_orthogonal_loss`: Orthogonal loss (if enabled)
  - `train_avg_loss`: 평균 학습 loss
  - `acc`: 정확도

- **Server-level metrics**:
  - `avg_vpl_total_loss`: 모든 클라이언트 평균
  - `avg_vpl_reconstruction_loss`
  - `avg_vpl_kl_loss`
  - `avg_vpl_orthogonal_loss`
  - `client_0_vpl_total_loss`: 개별 클라이언트 (0, 1, 2)

### 4. t-SNE 시각화

- 위치: `exp/{expname}/cross_client_z_tsne_round_{round_num}.png`
- 빈도: `vpl_tsne_visualize_freq` (기본: 10 라운드마다)
- 내용:
  - 클라이언트별 z embedding 분포
  - Orthogonal prototypes (orthogonal loss 사용 시)

---

## 알고리즘별 실험 설정

### 1. VPL-GP (Variational Preference Learning with Gumbel-Softmax Prior)

**설명**: VPL with Gumbel-Softmax mixture prior and optional orthogonal loss

**Config 위치**: `cfg/vpl-gp/`

**주요 설정**:
- `vpl_use_gp_prior: True`
- `vpl_kl_weight`: KL divergence weight
- `vpl_orthogonal_weight`: Orthogonal loss weight (0.0 = disabled)
- `vpl_use_difference_only`: Use only difference embedding

**실험 ID 범위**: 40000-59999

### 2. FedVPL (Federated Variational Preference Learning)

**설명**: Naive VPL for federated learning (baseline, no GP prior, no orthogonal loss)

**Config 위치**: `cfg/fedvpl/`

**주요 설정**:
- `vpl_use_gp_prior: False`
- `vpl_orthogonal_weight: 0.0`
- `vpl_use_feature_difference: False` (follows original VPL)

**실험 ID 범위**: 30000-39999

### 3. FedDPO (Federated Direct Preference Optimization)

**설명**: Pure DPO baseline for federated learning

**Config 위치**: `cfg/feddpo/`

**주요 설정**:
- `trainer.type: llmdporewardtrainer`
- `llm.reward_coeff: 0.1` (DPO beta parameter)

**실험 ID 범위**: 10000-19999

### 4. FedBiscuit

**설명**: Multi-LoRA baseline (U=3 adapters)

**Config 위치**: `cfg/fedbiscuit/`

**주요 설정**:
- `trainer.type: llmrewardchoicetrainer`
- `llm.adapter.count: 3`

**실험 ID 범위**: 20000-29999

---

## 실험 생성 가이드

### 새 실험 생성 절차

1. **Configuration 파일 생성**
   ```bash
   cp cfg/vpl-gp/hhst-40100.yaml cfg/vpl-gp/hhst-{new_tid}.yaml
   ```

2. **Configuration 수정**
   - `device`: GPU 번호
   - `tid`: 실험 ID
   - `expname`: 실험 이름
   - 하이퍼파라미터 조정

3. **스크립트 생성**
   ```bash
   cp scripts/vpl-gp/hhst-40100.sh scripts/vpl-gp/hhst-{new_tid}.sh
   ```

4. **스크립트 수정**
   - `tid={new_tid}`
   - `--cfg cfg/vpl-gp/hhst-{new_tid}.yaml`
   - `expname "vplgp_hhst_t{new_tid}"`

5. **실험 실행**
   ```bash
   chmod +x scripts/vpl-gp/hhst-{new_tid}.sh
   bash scripts/vpl-gp/hhst-{new_tid}.sh
   ```

### 하이퍼파라미터 조정 예시

#### KL weight 증가 (10배)

```yaml
# Before
vpl_kl_weight: 0.1

# After
vpl_kl_weight: 1.0
```

#### Orthogonal weight 증가 (10배)

```yaml
# Before
vpl_orthogonal_weight: 10.0

# After
vpl_orthogonal_weight: 100.0
```

#### Difference-only embedding 활성화

```yaml
vpl_use_feature_difference: True
vpl_use_difference_only: True  # 추가
```

---

## 문제 해결 (Troubleshooting)

### 1. GPU 메모리 부족 (OOM)

- `batch_size` 감소
- `grad_accum_step` 증가 (effective batch size 유지)
- `tok_len`, `max_new_token` 감소
- `is_enable_half: True` 확인

### 2. 실험이 시작되지 않음

- 로그 확인: `tail -f outputs/{tid}.log`
- 프로세스 확인: `ps aux | grep {tid}`
- GPU 사용 확인: `nvidia-smi`

### 3. 에러 발생 시

- 로그에서 에러 메시지 확인
- Python 캐시 삭제: `find . -type d -name __pycache__ -exec rm -r {} +`
- 체크포인트 경로 확인

### 4. WandB 로그가 안 보임

- `wandb.use: True` 확인
- WandB 로그인 확인: `wandb login`
- 프로젝트 이름 확인: `wandb.name_project`

---

## 참고 문서

- [VPL Documentation](VPL_DOCUMENTATION.md): VPL 구현 상세 설명
- [GP Prior Documentation](GP_PRIOR_DOCUMENTATION.md): Gumbel-Softmax Prior 설명
- [Orthogonal Loss Documentation](ORTHOGONAL_LOSS_DOCUMENTATION.md): Orthogonal Loss 설명
- [VPL Configuration Options](VPL_CONFIGURATION_OPTIONS.md): 모든 VPL 설정 옵션

---

## 업데이트 이력

- 2025-01-20: 초기 문서 작성
- 실험 ID 체계, 스크립트 구조, configuration 구조 문서화
