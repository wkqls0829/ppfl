# Hyperparameter Search Scripts

SLURM 클러스터에서 Hyperparameter Search 실험을 실행하기 위한 스크립트입니다.

## 실험 구조

- **Selector Training (Binary Classification)**: TID 54000-54038
- **RL Training (RLHF)**: TID 55000-55038

## Phase별 실험 범위

### Phase 1: Orthogonal Loss Parameters
- **Selector**: 54000-54006 (7개)
- **RL**: 55000-55006 (7개)

### Phase 2: VPL Core Parameters
- **Selector**: 54007-54013 (7개)
- **RL**: 55007-55013 (7개)

### Phase 3: Learning Rate
- **Selector**: 54014-54016 (3개)
- **RL**: 55014-55016 (3개)

### Phase 4: Combined Best Parameters
- **Selector**: 54017 (1개)
- **RL**: 55017 (1개)

### Phase 5: Fine-grained Hyperparameter Search
- **Selector**: 54018-54038 (21개)
- **RL**: 55018-55038 (21개)

## 파일 구조

```
scripts/hpsearch/
├── README.md (이 파일)
├── run_selector_hpsearch.sh      # Selector 실행 스크립트
├── run_rl_hpsearch.sh            # RL 실행 스크립트
├── submit_selector.sh            # Selector 제출 (phase 인자 사용)
└── submit_rl.sh                  # RL 제출 (phase 인자 사용)
```

## 사용 방법

### 1. Selector 실험 실행

#### Phase별로 실행
```bash
# Phase 1
bash scripts/hpsearch/submit_selector.sh 1

# Phase 2
bash scripts/hpsearch/submit_selector.sh 2

# Phase 3
bash scripts/hpsearch/submit_selector.sh 3

# Phase 4
bash scripts/hpsearch/submit_selector.sh 4

# Phase 5
bash scripts/hpsearch/submit_selector.sh 5
```

#### 개별 실험 실행
```bash
# 개별 selector 실험
sbatch scripts/hpsearch/run_selector_hpsearch.sh 54000
```

### 2. RL 실험 실행

**주의**: RL 실험은 해당 selector checkpoint가 완료된 후에 실행해야 합니다.

#### Phase별로 실행
```bash
# Phase 1 (selector 54000-54006 완료 후)
bash scripts/hpsearch/submit_rl.sh 1

# Phase 2 (selector 54007-54013 완료 후)
bash scripts/hpsearch/submit_rl.sh 2

# Phase 3 (selector 54014-54016 완료 후)
bash scripts/hpsearch/submit_rl.sh 3

# Phase 4 (selector 54017 완료 후)
bash scripts/hpsearch/submit_rl.sh 4

# Phase 5 (selector 54018-54038 완료 후)
bash scripts/hpsearch/submit_rl.sh 5
```

#### 개별 RL 실험 실행
```bash
# 개별 RL 실험 (selector checkpoint 필요)
sbatch scripts/hpsearch/run_rl_hpsearch.sh 55000 54000
```

## 하이퍼파라미터 설정

각 Phase별로 탐색하는 하이퍼파라미터는 `run_selector_hpsearch.sh` 스크립트 내에서 TID에 따라 자동으로 설정됩니다.

### Phase 1: Orthogonal Loss Parameters
- `vpl_orthogonal_weight`: [0.2, 1.0, 5.0]
- `vpl_orthogonal_orthonorm_weight`: [0.0, 0.1, 0.5]
- `vpl_prototype_scale`: [2.0, 5.0, 10.0]

### Phase 2: VPL Core Parameters
- `vpl_kl_weight`: [0.02, 0.05, 0.1, 0.2]
- `vpl_gp_temperature`: [0.5, 1.0, 2.0, 5.0]

### Phase 3: Learning Rate
- `lr`: [0.00005, 0.0001, 0.0002]

### Phase 4: Combined Best Parameters
- Phase 1-3의 최적값 조합

### Phase 5: Fine-grained Search
- `vpl_orthogonal_weight`: [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
- `vpl_orthogonal_orthonorm_weight`: [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
- `vpl_kl_weight`: [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

## 설정 변경사항

### 체크포인트 경로
- `$WORK_DIR/checkpoints/` (로컬 repo)

### 데이터 경로
- `$WORK_DIR/data/` (로컬 repo)

### 공통 설정
- Model: `google/gemma-2b@huggingface_llm`
- Client Num: 10
- Sample Client Num: 5
- Total Rounds: 50
- Batch Size: 8
- Grad Accum Step: 4
- Local Update Steps: 30
- VPL Latent Dim: 32

## 모니터링

### SLURM 작업 상태 확인
```bash
squeue -u $USER
```

### 로그 확인
```bash
# 실시간 로그 확인
tail -f outputs/54000.log

# SLURM 출력 확인
tail -f /home2/jbkoo/slurm/logs/slurm-*.out
```

### 체크포인트 확인
```bash
ls -lh checkpoints/*54000*.ckpt
```

## 주의사항

1. **Selector 완료 후 RL 실행**: RL 실험은 해당 Selector checkpoint가 필요합니다.
2. **Phase별 순차 실행**: Phase 2는 Phase 1의 최적값을 사용하므로, Phase 1 완료 후 실행하는 것을 권장합니다.
3. **Phase 5 순차 실행**: Phase 5의 Sub-phase 5.2는 5.1의 최적값을, 5.3는 5.1-5.2의 최적값을 사용합니다.
4. **체크포인트 경로**: 로컬 repo의 `checkpoints/` 디렉토리에 저장됩니다.
5. **데이터 경로**: 데이터가 `$WORK_DIR/data/`에 있어야 합니다.
6. **GPU 메모리**: RL 실험은 메모리 사용량이 크므로 GPU당 하나씩만 실행됩니다.

## Phase 5 세부 사항

Phase 5는 4개의 Sub-phase로 구성됩니다:

1. **Sub-phase 5.1** (54018-54024): Orthogonal Weight 탐색
2. **Sub-phase 5.2** (54025-54030): Orthonorm Weight 탐색 (5.1 최적값 사용)
3. **Sub-phase 5.3** (54031-54037): KL Weight 탐색 (5.1-5.2 최적값 사용)
4. **Sub-phase 5.4** (54038): 최적 조합 검증

**참고**: Sub-phase 5.2와 5.3는 이전 sub-phase의 최적값을 사용해야 하므로, 현재 스크립트는 기본값을 사용합니다. 최적값이 결정되면 스크립트를 수정하여 반영해야 합니다.
