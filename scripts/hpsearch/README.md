# Hyperparameter Search Scripts

SLURM 클러스터에서 Hyperparameter Search 실험을 실행하기 위한 스크립트입니다.

## 실험 구조

- **Gemma**: Selector 54000-54047, RL 55000-55047
- **Qwen**: Selector 54100-54138, RL 55100-55138

## Phase별 실험 범위 (Gemma)

### Phase 1: Orthogonal Loss Parameters
- **Selector**: 54000-54006 (7개)
- **RL**: 55000-55006 (7개)

### Phase 2: VPL Core Parameters
- **Selector**: 54007-54013 (7개)
- **RL**: 55007-55013 (7개)

### Phase 3: Refinement (54005/54008/54013 기반)
- **Selector**: 54014-54022 (9개)
- **RL**: 55014-55022 (9개)
- prototype_scale, kl_weight, gp_temperature 세밀 탐색

### Phase 4: Learning Rate
- **Selector**: 54023-54025 (3개)
- **RL**: 55023-55025 (3개)

### Phase 5: Combined Best Parameters
- **Selector**: 54026 (1개)
- **RL**: 55026 (1개)

### Phase 6: Fine-grained Hyperparameter Search
- **Selector**: 54027-54047 (21개)
- **RL**: 55027-55047 (21개)

## 파일 구조

```
scripts/hpsearch/
├── README.md (이 파일)
├── run_selector_hpsearch.sh      # Selector 실행 (Gemma, SLURM)
├── run_rl_hpsearch.sh            # RL 실행 (Gemma, SLURM, selector 필요)
├── submit_selector.sh            # Selector 제출 Gemma (phase 인자)
├── submit_rl.sh                  # RL 제출 Gemma (phase 인자)
├── run_selector_hpsearch_qwen.sh # Selector 실행 (Qwen, SLURM)
├── run_rl_hpsearch_qwen.sh      # RL 실행 (Qwen, SLURM, selector 필요)
├── submit_selector_qwen.sh       # Selector 제출 Qwen (phase 인자)
├── submit_rl_qwen.sh             # RL 제출 Qwen (phase 인자)
├── run_rl_local_only.sh         # Local RL only (로컬, selector 불필요)
├── vpl-gp/                      # Phase 스크립트 등
└── vpl-gp-rl/
```

## 사용 방법

### 1. Selector 실험 실행

**중요**: `submit_selector.sh`와 `submit_rl.sh`는 bash 스크립트이므로 `bash`로 실행해야 합니다. `sbatch`로 실행하면 안 됩니다!

#### Phase별로 실행
```bash
# Phase 1 (7개)
bash scripts/hpsearch/submit_selector.sh 1
# Phase 2 (7개)
bash scripts/hpsearch/submit_selector.sh 2
# Phase 3 (9개, refinement)
bash scripts/hpsearch/submit_selector.sh 3
# Phase 4 (3개, LR)
bash scripts/hpsearch/submit_selector.sh 4
# Phase 5 (1개, combined best)
bash scripts/hpsearch/submit_selector.sh 5
# Phase 6 (21개, fine-grained)
bash scripts/hpsearch/submit_selector.sh 6
```

#### 개별 실험 실행
```bash
# 개별 selector 실험 (sbatch 사용)
sbatch scripts/hpsearch/run_selector_hpsearch.sh 54000
```

### 2. RL 실험 실행

**주의**: 
- RL 실험은 해당 selector checkpoint가 완료된 후에 실행해야 합니다.
- `submit_rl.sh`는 bash 스크립트이므로 `bash`로 실행해야 합니다. `sbatch`로 실행하면 안 됩니다!

#### Phase별로 실행
```bash
# Phase 1–2 완료 후 RL 제출
bash scripts/hpsearch/submit_rl.sh 1
bash scripts/hpsearch/submit_rl.sh 2
# Phase 3 (selector 54014-54022 완료 후, 9개)
bash scripts/hpsearch/submit_rl.sh 3
# Phase 4–6
bash scripts/hpsearch/submit_rl.sh 4
bash scripts/hpsearch/submit_rl.sh 5
bash scripts/hpsearch/submit_rl.sh 6
```

#### 개별 RL 실험 실행
```bash
# 개별 RL 실험 (sbatch 사용, selector checkpoint 필요)
sbatch scripts/hpsearch/run_rl_hpsearch.sh 55000 54000
```

### 3. Local RL Training Only (Baseline)

**목적**: Selector checkpoint 없이 RL training만 실행하여 baseline 성능 측정

**특징**:
- Selector checkpoint 불필요
- 로컬 서버에서 실행 (SLURM cluster 아님)
- `rlhf_use_variational_selection: False`

**실행 방법**:
```bash
# 로컬 서버에서 실행 (sbatch 사용 안 함, bash로 직접 실행)
bash scripts/hpsearch/run_rl_local_only.sh 56000
```

**TID 범위**: 56000-560XX (baseline 실험용)

## 하이퍼파라미터 설정

각 Phase별로 탐색하는 하이퍼파라미터는 `run_selector_hpsearch.sh` 스크립트 내에서 TID에 따라 자동으로 설정됩니다.

### Phase 1: Orthogonal Loss Parameters
- `vpl_orthogonal_weight`: [0.2, 1.0, 5.0]
- `vpl_orthogonal_orthonorm_weight`: [0.0, 0.1, 0.5]
- `vpl_prototype_scale`: [2.0, 5.0, 10.0]

### Phase 2: VPL Core Parameters
- `vpl_kl_weight`: [0.02, 0.05, 0.1, 0.2]
- `vpl_gp_temperature`: [0.5, 1.0, 2.0, 5.0]

### Phase 3: Refinement (54005/54008/54013 기반)
- `vpl_prototype_scale`: [1.0, 2.0, 3.0] (54014-54016)
- `vpl_kl_weight`: [0.03, 0.05, 0.08] (54017-54019)
- `vpl_gp_temperature`: [3.0, 5.0, 7.0] (54020-54022)

### Phase 4: Learning Rate
- `lr`: [0.00005, 0.0001, 0.0002]

### Phase 5: Combined Best Parameters
- Phase 1-4 최적값 조합

### Phase 6: Fine-grained Search
- `vpl_orthogonal_weight`: [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
- `vpl_orthogonal_orthonorm_weight`: [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
- `vpl_kl_weight`: [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

## TID별 하이퍼파라미터 (Phase 1·2)

Phase 1·2만 TID와 파라미터 값을 매핑한 표입니다. 공통 기본값: `lr=0.0001`, Phase 2 기준값은 Phase 1 최적(orthogonal=1.0, orthonorm=0.1, prototype_scale=5.0).

### Gemma (google/gemma-2b)

**Phase 1** — Selector 54000–54006 / RL 55000–55006

| TID (Sel/RL) | vpl_orthogonal_weight | vpl_orthogonal_orthonorm_weight | vpl_prototype_scale | vpl_kl_weight | vpl_gp_temperature |
|--------------|-----------------------|----------------------------------|----------------------|---------------|--------------------|
| 54000/55000 | **0.2** | 0.1 | 5.0 | 0.1 | 1.0 |
| 54001/55001 | **1.0** | 0.1 | 5.0 | 0.1 | 1.0 |
| 54002/55002 | **5.0** | 0.1 | 5.0 | 0.1 | 1.0 |
| 54003/55003 | 1.0 | **0.0** | 5.0 | 0.1 | 1.0 |
| 54004/55004 | 1.0 | **0.5** | 5.0 | 0.1 | 1.0 |
| 54005/55005 | 1.0 | 0.1 | **2.0** | 0.1 | 1.0 |
| 54006/55006 | 1.0 | 0.1 | **10.0** | 0.1 | 1.0 |

**Phase 2** — Selector 54007–54013 / RL 55007–55013

| TID (Sel/RL) | vpl_orthogonal_weight | vpl_orthogonal_orthonorm_weight | vpl_prototype_scale | vpl_kl_weight | vpl_gp_temperature |
|--------------|-----------------------|----------------------------------|----------------------|---------------|--------------------|
| 54007/55007 | 1.0 | 0.1 | 5.0 | **0.02** | 1.0 |
| 54008/55008 | 1.0 | 0.1 | 5.0 | **0.05** | 1.0 |
| 54009/55009 | 1.0 | 0.1 | 5.0 | **0.1** | 1.0 |
| 54010/55010 | 1.0 | 0.1 | 5.0 | **0.2** | 1.0 |
| 54011/55011 | 1.0 | 0.1 | 5.0 | 0.1 | **0.5** |
| 54012/55012 | 1.0 | 0.1 | 5.0 | 0.1 | **2.0** |
| 54013/55013 | 1.0 | 0.1 | 5.0 | 0.1 | **5.0** |

**Phase 3** — Refinement (54005/54008/54013 기반) — Selector 54014–54022 / RL 55014–55022

| TID (Sel/RL) | vpl_orthogonal_weight | vpl_orthogonal_orthonorm_weight | vpl_prototype_scale | vpl_kl_weight | vpl_gp_temperature |
|--------------|-----------------------|----------------------------------|----------------------|---------------|--------------------|
| 54014/55014 | 1.0 | 0.1 | **1.0** | 0.1 | 1.0 |
| 54015/55015 | 1.0 | 0.1 | **2.0** | 0.1 | 1.0 |
| 54016/55016 | 1.0 | 0.1 | **3.0** | 0.1 | 1.0 |
| 54017/55017 | 1.0 | 0.1 | 2.0 | **0.03** | 1.0 |
| 54018/55018 | 1.0 | 0.1 | 2.0 | **0.05** | 1.0 |
| 54019/55019 | 1.0 | 0.1 | 2.0 | **0.08** | 1.0 |
| 54020/55020 | 1.0 | 0.1 | 2.0 | 0.05 | **3.0** |
| 54021/55021 | 1.0 | 0.1 | 2.0 | 0.05 | **5.0** |
| 54022/55022 | 1.0 | 0.1 | 2.0 | 0.05 | **7.0** |

- 실행: `submit_selector.sh` / `submit_rl.sh`, 스크립트: `run_selector_hpsearch.sh`, `run_rl_hpsearch.sh`
- Phase 3에서 54015(prototype_scale=2.0), 54018(kl=0.05), 54021(gp_temp=5.0), Phase 4에서 54024(lr=0.0001) 반영.
- 상세: `PHASE3_EXECUTION_GUIDE.md` 참고

**Phase 5 (Combined Best)** — Selector 54026 / RL 55026

| TID (Sel/RL) | vpl_orthogonal_weight | vpl_orthogonal_orthonorm_weight | vpl_prototype_scale | vpl_kl_weight | vpl_gp_temperature | lr |
|--------------|-----------------------|----------------------------------|----------------------|---------------|--------------------|-----|
| 54026/55026 | 1.0 | 0.1 | **2.0** | **0.05** | **5.0** | **0.0001** |

- Phase 1–4 최적/중간값 조합. 실행: `submit_selector.sh 5`, `submit_rl.sh 5`. 상세: `PHASE5_COMBINED_BEST_GUIDE.md`

### Qwen (Qwen2-0.5B)

**Phase 1** — Selector 54100–54106 / RL 55100–55106

| TID (Sel/RL) | vpl_orthogonal_weight | vpl_orthogonal_orthonorm_weight | vpl_prototype_scale | vpl_kl_weight | vpl_gp_temperature |
|--------------|-----------------------|----------------------------------|----------------------|---------------|--------------------|
| 54100/55100 | **0.2** | 0.1 | 5.0 | 0.1 | 1.0 |
| 54101/55101 | **1.0** | 0.1 | 5.0 | 0.1 | 1.0 |
| 54102/55102 | **5.0** | 0.1 | 5.0 | 0.1 | 1.0 |
| 54103/55103 | 1.0 | **0.0** | 5.0 | 0.1 | 1.0 |
| 54104/55104 | 1.0 | **0.5** | 5.0 | 0.1 | 1.0 |
| 54105/55105 | 1.0 | 0.1 | **2.0** | 0.1 | 1.0 |
| 54106/55106 | 1.0 | 0.1 | **10.0** | 0.1 | 1.0 |

**Phase 2** — Selector 54107–54113 / RL 55107–55113 **(54105 기반: prototype_scale=2.0)**

| TID (Sel/RL) | vpl_orthogonal_weight | vpl_orthogonal_orthonorm_weight | vpl_prototype_scale | vpl_kl_weight | vpl_gp_temperature |
|--------------|-----------------------|----------------------------------|----------------------|---------------|--------------------|
| 54107/55107 | 1.0 | 0.1 | **2.0** | **0.02** | 1.0 |
| 54108/55108 | 1.0 | 0.1 | **2.0** | **0.05** | 1.0 |
| 54109/55109 | 1.0 | 0.1 | **2.0** | **0.1** | 1.0 |
| 54110/55110 | 1.0 | 0.1 | **2.0** | **0.2** | 1.0 |
| 54111/55111 | 1.0 | 0.1 | **2.0** | 0.1 | **0.5** |
| 54112/55112 | 1.0 | 0.1 | **2.0** | 0.1 | **2.0** |
| 54113/55113 | 1.0 | 0.1 | **2.0** | 0.1 | **5.0** |

- Phase 1에서 54105(prototype_scale=2.0)를 기준으로 Phase 2에서 kl_weight / gp_temperature만 탐색.
- 실행: `submit_selector_qwen.sh` / `submit_rl_qwen.sh`, 스크립트: `run_selector_hpsearch_qwen.sh`, `run_rl_hpsearch_qwen.sh`

**Phase 4 (Combined Best)** — Selector 54117 / RL 55117

| TID (Sel/RL) | vpl_orthogonal_weight | vpl_orthogonal_orthonorm_weight | vpl_prototype_scale | vpl_kl_weight | vpl_gp_temperature | lr |
|--------------|-----------------------|----------------------------------|----------------------|---------------|--------------------|-----|
| 54117/55117 | 1.0 | 0.1 | **2.0** | **0.05** | **1.0** | **0.0001** |

- Phase 1·2 최적/중간값 조합. 실행: `submit_selector_qwen.sh 4`, `submit_rl_qwen.sh 4`. 상세: `PHASE5_COMBINED_BEST_GUIDE.md`

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
2. **Phase별 순차 실행**: Phase 2는 Phase 1, Phase 3는 54005/54008/54013 기반 refinement이므로 Phase 1·2 완료 후 실행 권장.
3. **Phase 6 순차 실행**: Phase 6의 Sub-phase 6.2는 6.1의, 6.3는 6.1-6.2의 최적값을 사용합니다.
4. **체크포인트 경로**: 로컬 repo의 `checkpoints/` 디렉토리에 저장됩니다.
5. **데이터 경로**: 데이터가 `$WORK_DIR/data/`에 있어야 합니다.
6. **GPU 메모리**: RL 실험은 메모리 사용량이 크므로 GPU당 하나씩만 실행됩니다.

## Phase 6 세부 사항 (Gemma)

Phase 6는 4개의 Sub-phase로 구성됩니다:

1. **Sub-phase 6.1** (54027-54033): Orthogonal Weight 탐색
2. **Sub-phase 6.2** (54034-54039): Orthonorm Weight 탐색
3. **Sub-phase 6.3** (54040-54046): KL Weight 탐색
4. **Sub-phase 6.4** (54047): 최적 조합 검증

**참고**: Sub-phase 6.2와 6.3는 이전 sub-phase의 최적값을 사용해야 하므로, 현재 스크립트는 기본값을 사용합니다. 최적값이 결정되면 스크립트를 수정하여 반영해야 합니다.
