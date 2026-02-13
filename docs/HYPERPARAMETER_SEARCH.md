# VPL-GP Hyperparameter Search Documentation

## 개요

이 문서는 **Variational Preference Learning with Gumbel-Softmax Prior (VPL-GP)** 모델의 하이퍼파라미터 탐색 실험에 대한 전체 계획, 상세 설정, 진행 상황, 그리고 결과 분석 가이드를 제공합니다.

## 실험 구조

하이퍼파라미터 서치는 **2단계 구조**로 진행됩니다:

1. **Selector Training (Binary Classification)**: 각 하이퍼파라미터 조합에 대해 selector 모델 학습
   - Task ID: 54000-54038
   - WandB Project: `fvpl-selector`
   - Config: `cfg/hpsearch/vpl-gp/phase_*.yaml`
   - Scripts: `scripts/hpsearch/run_selector_hpsearch.sh <tid>`
   - Submit: `scripts/hpsearch/submit_selector.sh <phase>`

2. **RL Training (RLHF)**: 완료된 selector checkpoint를 사용하여 RLHF 학습
   - Task ID: 55000-55038
   - WandB Project: `fvpl-rl`
   - Config: `cfg/hpsearch/vpl-gp-rl/hrl_*.yaml`
   - Scripts: `scripts/hpsearch/run_rl_hpsearch.sh <rl_tid> <selector_tid>`
   - Submit: `scripts/hpsearch/submit_rl.sh <phase>`

### Qwen 하이퍼파라미터 서치 (54100 / 55100)

동일한 Phase 구조로 **Qwen2-0.5B** 모델에 대한 하이퍼파라미터 서치를 별도 TID 대역으로 진행합니다.

1. **Selector Training (Qwen)**  
   - Task ID: **54100–54138**  
   - WandB Project: `fvpl-selector`  
   - Config: `cfg/hpsearch/vpl-gp-qwen/phase_${TID}.yaml`  
   - Script: `scripts/hpsearch/run_selector_hpsearch_qwen.sh <tid>`  
   - Submit: `bash scripts/hpsearch/submit_selector_qwen.sh <phase>`  
   - Checkpoint: `hhrl_choice_qwen2_fedbiscuit_u3_vplgp_ortho_t${TID}.ckpt` (또는 `final_*`, `40_*`)

2. **RL Training (Qwen)**  
   - Task ID: **55100–55138**  
   - WandB Project: `fvpl-rl`  
   - Config: `cfg/hpsearch/vpl-gp-rl-qwen/hrl_${RL_TID}.yaml`  
   - Script: `scripts/hpsearch/run_rl_hpsearch_qwen.sh <rl_tid> <selector_tid>`  
   - Submit: `bash scripts/hpsearch/submit_rl_qwen.sh <phase>`  
   - Selector TID: 54100–54138 (phase별 54100–54106, 54107–54113, 54114–54116, 54117, 54118–54138)  
   - Checkpoint: `hhrl_rlhf_qwen2_choice_vplgp_t${RL_TID}.ckpt`  
   - Model: `Qwen/Qwen2-0.5B@huggingface_llm`

Phase 범위는 Gemma와 동일: Phase 1 (54100–54106), Phase 2 (54107–54113), Phase 3 (54114–54116), Phase 4 (54117), Phase 5 (54118–54138).

## 공통 설정 (모든 실험)

| 파라미터 | 값 |
|---------|-----|
| Model | `google/gemma-2b@huggingface_llm` |
| Client Num | 10 |
| Sample Client Num | 5 |
| Total Rounds | 50 |
| Batch Size | 8 |
| Grad Accum Step | 4 |
| Local Update Steps | 30 |
| VPL Latent Dim | 32 |
| VPL Feature Method | `choice_logits` |
| VPL Use Feature Difference | `True` |
| VPL Use Difference Only | `True` |
| VPL Max Logvar | -3.0 |
| VPL Use Manual Orthogonal Labels | `True` |
| VPL Num Prototypes | 2 |

## Phase별 실험 계획 및 상세 설정

### Phase 1: Orthogonal Loss Parameters (54000-54006)

**목적**: Orthogonal loss의 weight와 prototype scale을 탐색하여 클라이언트 간 preference disentanglement를 최적화합니다.

**탐색 파라미터**:
- `vpl_orthogonal_weight`: [0.2, 1.0, 5.0]
- `vpl_orthogonal_orthonorm_weight`: [0.0, 0.1, 0.5]
- `vpl_prototype_scale`: [2.0, 5.0, 10.0]

**고정 파라미터**:
- `vpl_kl_weight`: 0.1
- `vpl_gp_temperature`: 1.0
- `lr`: 0.0001

**실험별 상세 설정**:

| TID | orthogonal_weight | orthonorm_weight | prototype_scale | kl_weight | gp_temperature | lr | Status |
|-----|------------------|------------------|-----------------|-----------|----------------|-----|--------|
| 54000 | 0.2 | 0.1 | 5.0 | 0.1 | 1.0 | 0.0001 | Running |
| 54001 | 1.0 | 0.1 | 5.0 | 0.1 | 1.0 | 0.0001 | Completed |
| 54002 | 5.0 | 0.1 | 5.0 | 0.1 | 1.0 | 0.0001 | Completed |
| 54003 | 1.0 | 0.0 | 5.0 | 0.1 | 1.0 | 0.0001 | Completed |
| 54004 | 1.0 | 0.5 | 5.0 | 0.1 | 1.0 | 0.0001 | Completed |
| 54005 | 1.0 | 0.1 | 2.0 | 0.1 | 1.0 | 0.0001 | Completed |
| 54006 | 1.0 | 0.1 | 10.0 | 0.1 | 1.0 | 0.0001 | Completed |

**설명**:
- **54000-54002**: `orthogonal_weight` 탐색 (0.2 → 1.0 → 5.0)
- **54003-54004**: `orthonorm_weight` 탐색 (0.0 → 0.5, baseline 0.1은 54001)
- **54005-54006**: `prototype_scale` 탐색 (2.0 → 10.0, baseline 5.0은 54001)

**최적값 (Phase 1 결과 기반)**:
- `vpl_orthogonal_weight`: 1.0
- `vpl_orthogonal_orthonorm_weight`: 0.1
- `vpl_prototype_scale`: 5.0

**평가 지표**:
- `train_avg_loss`: 주요 최적화 목표
- `vpl_orthogonal_loss`: Orthogonal loss 값
- `vpl_kl_loss`: KL divergence
- `acc`: Binary classification accuracy
- t-SNE visualization: 클라이언트별 z 분포 시각화

**분석 포인트**:
1. Orthogonal loss weight가 클라이언트 간 z 분리에 미치는 영향
2. Prototype scale이 preference disentanglement에 미치는 영향
3. Loss 값과 실제 z 분포의 상관관계

---

### Phase 2: VPL Core Parameters (54007-54013)

**목적**: VPL-GP의 핵심 하이퍼파라미터인 KL weight와 Gumbel-Softmax temperature를 탐색합니다.

**탐색 파라미터**:
- `vpl_kl_weight`: [0.02, 0.05, 0.1, 0.2]
- `vpl_gp_temperature`: [0.5, 1.0, 2.0, 5.0]

**Phase 1 최적값 사용**:
- `vpl_orthogonal_weight`: 1.0
- `vpl_orthogonal_orthonorm_weight`: 0.1
- `vpl_prototype_scale`: 5.0
- `lr`: 0.0001

**실험별 상세 설정**:

| TID | kl_weight | gp_temperature | orthogonal_weight | orthonorm_weight | prototype_scale | lr | Status |
|-----|-----------|----------------|-------------------|------------------|-----------------|-----|--------|
| 54007 | 0.02 | 1.0 | 1.0 | 0.1 | 5.0 | 0.0001 | Completed |
| 54008 | 0.05 | 1.0 | 1.0 | 0.1 | 5.0 | 0.0001 | Completed |
| 54009 | 0.1 | 1.0 | 1.0 | 0.1 | 5.0 | 0.0001 | Completed |
| 54010 | 0.2 | 1.0 | 1.0 | 0.1 | 5.0 | 0.0001 | Completed |
| 54011 | 0.1 | 0.5 | 1.0 | 0.1 | 5.0 | 0.0001 | Completed |
| 54012 | 0.1 | 2.0 | 1.0 | 0.1 | 5.0 | 0.0001 | Completed |
| 54013 | 0.1 | 5.0 | 1.0 | 0.1 | 5.0 | 0.0001 | Completed |

**설명**:
- **54007-54010**: `kl_weight` 탐색 (0.02 → 0.05 → 0.1 → 0.2)
- **54011-54013**: `gp_temperature` 탐색 (0.5 → 2.0 → 5.0, baseline 1.0은 54009)

**최적값 (Phase 2 결과 기반)**:
- `vpl_kl_weight`: 0.1
- `vpl_gp_temperature`: 1.0

**평가 지표**:
- `train_avg_loss`: 주요 최적화 목표
- `vpl_kl_loss`: KL divergence (mixture prior vs standard prior)
- `vpl_reconstruction_loss`: Reconstruction loss
- t-SNE visualization: Gumbel-Softmax prior의 효과 확인

**분석 포인트**:
1. KL weight가 prior regularization에 미치는 영향
2. Temperature가 Gumbel-Softmax sampling에 미치는 영향
3. Mixture prior의 효과 (다른 클라이언트들의 분포 활용)

---

### Phase 3: Learning Rate (54014-54016)

**목적**: 최적 learning rate를 탐색합니다. Phase 1-2의 최적 하이퍼파라미터를 사용합니다.

**탐색 파라미터**:
- `lr`: [0.00005, 0.0001, 0.0002]

**Phase 1-2 최적값 사용**:
- `vpl_orthogonal_weight`: 1.0
- `vpl_orthogonal_orthonorm_weight`: 0.1
- `vpl_prototype_scale`: 5.0
- `vpl_kl_weight`: 0.1
- `vpl_gp_temperature`: 1.0

**실험별 상세 설정**:

| TID | lr | kl_weight | gp_temperature | orthogonal_weight | orthonorm_weight | prototype_scale | Status |
|-----|----|-----------|----------------|------------------|-----------------|-----------------|--------|
| 54014 | 0.00005 | 0.1 | 1.0 | 1.0 | 0.1 | 5.0 | Completed |
| 54015 | 0.0001 | 0.1 | 1.0 | 1.0 | 0.1 | 5.0 | Completed |
| 54016 | 0.0002 | 0.1 | 1.0 | 1.0 | 0.1 | 5.0 | Completed |

**최적값 (Phase 3 결과 기반)**:
- `lr`: 0.0001

**평가 지표**:
- `train_avg_loss`: 수렴 속도와 최종 loss
- Training stability
- Convergence speed

**분석 포인트**:
1. Learning rate가 수렴 속도에 미치는 영향
2. 최적 learning rate 결정

---

### Phase 4: Combined Best Parameters (54017)

**목적**: Phase 1-3에서 찾은 최적 하이퍼파라미터 조합으로 최종 검증을 수행합니다.

**최종 하이퍼파라미터**:

| 파라미터 | 값 | 출처 |
|---------|-----|------|
| `vpl_orthogonal_weight` | 1.0 | Phase 1 최적값 |
| `vpl_orthogonal_orthonorm_weight` | 0.1 | Phase 1 최적값 |
| `vpl_prototype_scale` | 5.0 | Phase 1 최적값 |
| `vpl_kl_weight` | 0.1 | Phase 2 최적값 |
| `vpl_gp_temperature` | 1.0 | Phase 2 최적값 |
| `lr` | 0.0001 | Phase 3 최적값 |

| TID | orthogonal_weight | prototype_scale | kl_weight | temperature | lr | Status |
|-----|-------------------|-----------------|-----------|------------|-----|--------|
| 54017 | 1.0 | 5.0 | 0.1 | 1.0 | 0.0001 | Completed |

---

## RL Experiments (55000-55038)

각 selector 실험(54000-54038)에 대응하는 RL 실험이 있습니다. RL 실험은 해당 selector의 checkpoint를 사용하여 RLHF 학습을 수행합니다.

### RL 공통 설정

| 파라미터 | 값 |
|---------|-----|
| `rlhf_use_variational_selection` | `True` |
| `rlhf_use_variational_generation` | `False` |
| `reward_coeff` | 0.1 |
| `grad_accum_step` | 4 |
| `max_prompts_for_generation` | 50 |
| `generation_batch_size` | 3 |
| `max_samples_for_reward` | 30 |
| `use_gpt_api_for_winrate` | `True` |
| `use_baseline_model_for_winrate` | `True` |
| `openai_model` | `gpt-4o-mini` |

### RL 실험 매핑

| RL TID | Selector TID | Selector Phase | Selector 하이퍼파라미터 | Status |
|--------|-------------|----------------|----------------------|--------|
| 55000 | 54000 | Phase 1 | orthogonal_weight=0.2, prototype_scale=5.0 | Completed |
| 55001 | 54001 | Phase 1 | orthogonal_weight=1.0, prototype_scale=5.0 | Completed |
| 55002 | 54002 | Phase 1 | orthogonal_weight=5.0, prototype_scale=5.0 | Completed |
| 55003 | 54003 | Phase 1 | orthogonal_weight=1.0, orthonorm_weight=0.0 | Completed |
| 55004 | 54004 | Phase 1 | orthogonal_weight=1.0, orthonorm_weight=0.5 | Completed |
| 55005 | 54005 | Phase 1 | orthogonal_weight=1.0, prototype_scale=2.0 | Completed |
| 55006 | 54006 | Phase 1 | orthogonal_weight=1.0, prototype_scale=10.0 | Completed |
| 55007 | 54007 | Phase 2 | kl_weight=0.02, gp_temperature=1.0 | Completed |
| 55008 | 54008 | Phase 2 | kl_weight=0.05, gp_temperature=1.0 | Completed |
| 55009 | 54009 | Phase 2 | kl_weight=0.1, gp_temperature=1.0 | Completed |
| 55010 | 54010 | Phase 2 | kl_weight=0.2, gp_temperature=1.0 | Completed |
| 55011 | 54011 | Phase 2 | kl_weight=0.1, gp_temperature=0.5 | Completed |
| 55012 | 54012 | Phase 2 | kl_weight=0.1, gp_temperature=2.0 | Completed |
| 55013 | 54013 | Phase 2 | kl_weight=0.1, gp_temperature=5.0 | Completed |
| 55014 | 54014 | Phase 3 | lr=0.00005 | Completed |
| 55015 | 54015 | Phase 3 | lr=0.0001 | Completed |
| 55016 | 54016 | Phase 3 | lr=0.0002 | Completed |
| 55017 | 54017 | Phase 4 | 최적 조합 (모든 파라미터) | Completed |
| 55018 | 54018 | Phase 5.1 | orthogonal_weight=0.1 | Not started |
| 55019 | 54019 | Phase 5.1 | orthogonal_weight=0.2 | Not started |
| 55020 | 54020 | Phase 5.1 | orthogonal_weight=0.5 | Not started |
| 55021 | 54021 | Phase 5.1 | orthogonal_weight=1.0 | Not started |
| 55022 | 54022 | Phase 5.1 | orthogonal_weight=2.0 | Not started |
| 55023 | 54023 | Phase 5.1 | orthogonal_weight=5.0 | Not started |
| 55024 | 54024 | Phase 5.1 | orthogonal_weight=10.0 | Not started |
| 55025 | 54025 | Phase 5.2 | orthonorm_weight=0.0 | Not started |
| 55026 | 54026 | Phase 5.2 | orthonorm_weight=0.05 | Not started |
| 55027 | 54027 | Phase 5.2 | orthonorm_weight=0.1 | Not started |
| 55028 | 54028 | Phase 5.2 | orthonorm_weight=0.2 | Not started |
| 55029 | 54029 | Phase 5.2 | orthonorm_weight=0.5 | Not started |
| 55030 | 54030 | Phase 5.2 | orthonorm_weight=1.0 | Not started |
| 55031 | 54031 | Phase 5.3 | kl_weight=0.01 | Not started |
| 55032 | 54032 | Phase 5.3 | kl_weight=0.02 | Not started |
| 55033 | 54033 | Phase 5.3 | kl_weight=0.05 | Not started |
| 55034 | 54034 | Phase 5.3 | kl_weight=0.1 | Not started |
| 55035 | 54035 | Phase 5.3 | kl_weight=0.2 | Not started |
| 55036 | 54036 | Phase 5.3 | kl_weight=0.5 | Not started |
| 55037 | 54037 | Phase 5.3 | kl_weight=1.0 | Not started |
| 55038 | 54038 | Phase 5.4 | 최적 조합 (Phase 5.1-5.3) | Not started |

**평가 지표**:
- `avg_helpfulness`: Helpfulness score
- `avg_harmlessness`: Harmlessness score
- `helpfulness_winrate`: Helpful response win rate
- `harmlessness_winrate`: Harmless response win rate
- `avg_winlose_rate`: Overall win-lose rate

---

## 최종 최적 하이퍼파라미터

Phase 1-4 결과를 종합한 최종 하이퍼파라미터:

```yaml
llm:
  vpl_orthogonal_weight: 1.0
  vpl_orthogonal_orthonorm_weight: 0.1
  vpl_prototype_scale: 5.0
  vpl_kl_weight: 0.1
  vpl_gp_temperature: 1.0
train:
  optimizer:
    lr: 0.0001
```

이 값들이 main table 실험에서 사용됩니다.

---

## 결과 분석 계획

### 1. Selector 실험 결과 분석

#### Phase 1 분석
- **목표**: Orthogonal loss weight와 prototype scale의 최적값 결정
- **분석 방법**:
  1. WandB에서 `train_avg_loss` 비교
  2. t-SNE visualization으로 클라이언트별 z 분포 확인
  3. `vpl_orthogonal_loss` 값 비교
  4. 클라이언트 간 z 분리 정도 정량화

#### Phase 2 분석
- **목표**: KL weight와 temperature의 최적값 결정
- **분석 방법**:
  1. `vpl_kl_loss` 비교 (mixture prior 효과)
  2. `train_avg_loss` 비교
  3. Training stability 확인
  4. t-SNE visualization으로 prior 효과 확인

#### Phase 3 분석
- **목표**: 최적 learning rate 결정
- **분석 방법**:
  1. Convergence speed 비교
  2. Final loss 비교
  3. Training stability 확인

### 2. RL 실험 결과 분석

#### RL 성능 평가
- **목표**: 각 selector checkpoint로 학습한 RL 모델의 성능 비교
- **분석 방법**:
  1. `avg_helpfulness`와 `avg_harmlessness` 비교
  2. `helpfulness_winrate`와 `harmlessness_winrate` 비교
  3. `avg_winlose_rate` 비교
  4. Selector 하이퍼파라미터와 RL 성능의 상관관계 분석

#### 최적 하이퍼파라미터 결정
- Selector 성능과 RL 성능을 종합하여 최적 하이퍼파라미터 조합 결정
- Trade-off 분석 (selector accuracy vs RL performance)

### 3. 시각화 및 정량화

#### t-SNE Visualization
- 각 실험의 t-SNE plot 비교
- 클라이언트별 z 분포의 분리 정도 정량화
- Orthogonal loss의 효과 시각화

#### Loss Curves
- Training loss curves 비교
- KL loss, orthogonal loss, reconstruction loss 추이 분석

#### Performance Metrics
- WandB에서 모든 실험의 metrics 비교
- Best experiment 식별

---

## 파일 구조

```
cfg/hpsearch/vpl-gp/
├── phase1_orthogonal_5400.yaml
├── phase1_orthogonal_5401.yaml
├── ...
├── phase2_vpl_core_5407.yaml
├── ...
├── phase3_lr_5414.yaml
├── ...
└── phase4_combined_5417.yaml

cfg/hpsearch/vpl-gp-rl/
├── hrl_55000.yaml
├── hrl_55001.yaml
└── ...

scripts/hpsearch/vpl-gp/
├── phase1_orthogonal_5400.sh
├── ...

scripts/hpsearch/vpl-gp-rl/
├── hrl_55000.sh
├── ...

outputs/
├── 54000.log
├── 54001.log
├── ...
├── 55000.log
└── ...
```

---

## 체크포인트 위치

### Selector Checkpoints
- 위치: `/hdd/hdd3/kjb/checkpoints/`
- 파일명: `hhrl_choice_gemma_fedbiscuit_u3_vplgp_ortho_{TID}.ckpt`
- Final checkpoint: `final_hhrl_choice_gemma_fedbiscuit_u3_vplgp_ortho_{TID}.ckpt`

### RL Checkpoints
- 위치: `/hdd/hdd3/kjb/checkpoints/`
- 파일명: `hhrl_rlhf_gemma_choice_vplgp_ortho_{TID}.ckpt`

---

## WandB 프로젝트

- **Selector 실험**: `fvpl-selector`
- **RL 실험**: `fvpl-rl`

각 실험의 이름은 `vplgp_hhst_ortho_t{TID}` (selector) 또는 `vplgp_hrl_ortho_t{TID}` (RL) 형식입니다.

---

## 모니터링 및 디버깅

### 로그 확인
```bash
# 특정 실험 로그 확인
tail -f outputs/54000.log

# 여러 실험 로그 동시 확인
tail -f outputs/5200*.log
```

### 프로세스 확인
```bash
# 실행 중인 실험 확인
ps aux | grep -E "520[0-9]{2}|530[0-9]{2}" | grep python
```

### GPU 사용량 확인
```bash
nvidia-smi
```

### WandB 실시간 모니터링
- WandB 웹 인터페이스에서 실시간으로 metrics 확인
- 실험 간 비교 및 시각화

---

## 참고사항

- **OOM 방지**: RL 실험은 메모리 사용량이 크므로 GPU당 하나씩만 실행
- **체크포인트 확인**: RL 실험 실행 전 해당 selector checkpoint가 존재하는지 확인
- **로그 백업**: 중요한 실험의 로그는 별도로 백업 권장
- **WandB 동기화**: 실험 결과는 WandB에 자동으로 업로드되지만, 네트워크 문제 시 수동 동기화 필요

---

## Local RL Training Only Experiments (56000)

**목적**: Selector checkpoint 없이 RL training만 실행하여 baseline 성능을 측정합니다. 이는 selector training의 효과를 평가하기 위한 비교 실험입니다.

**특징**:
- Selector checkpoint 불필요 (FedDPO 방식)
- `rlhf_use_variational_selection: False` - Variational selection 사용 안 함
- FedDPO와 유사한 방식으로 직접 preference data에서 학습
- 로컬 서버에서 실행 (SLURM cluster 아님, nohup으로 백그라운드 실행)
- `scripts/feddpo/hrl-10000.sh` 스크립트를 참고하여 작성

**실험 설정**:

| TID | 설명 | Status |
|-----|------|--------|
| 56000 | Local RL only (baseline, no selector) | Not started |

**공통 설정**:

| 파라미터 | 값 |
|---------|-----|
| Model | `google/gemma-2b@huggingface_llm` |
| Trainer | `llmdporewardtrainer` (DPO trainer) |
| `rlhf_use_variational_selection` | `False` |
| `rlhf_use_variational_generation` | `False` |
| `reward_coeff` | 0.1 |
| `grad_accum_step` | 4 |
| `max_prompts_for_generation` | 50 |
| `generation_batch_size` | 3 |
| `use_gpt_api_for_winrate` | `True` |
| `use_baseline_model_for_winrate` | `True` |
| `openai_model` | `gpt-4o-mini` |
| Learning rate | 0.0001 |
| Total rounds | 50 |
| Local update steps | 30 |
| Batch size | 1 |

**실행 방법**:

```bash
# 로컬 서버에서 실행 (bash로 실행, nohup으로 백그라운드 실행됨)
bash scripts/hpsearch/run_rl_local_only.sh
```

**참고**: 
- TID는 스크립트 내부에서 56000으로 고정되어 있습니다.
- 스크립트는 `nohup`으로 백그라운드 실행되므로 터미널을 닫아도 계속 실행됩니다.
- 로그는 `outputs/56000.log`에 저장됩니다.

**평가 지표**:
- `avg_helpfulness`: Helpfulness score
- `avg_harmlessness`: Harmlessness score
- `helpfulness_winrate`: Helpful response win rate
- `harmlessness_winrate`: Harmless response win rate
- `avg_winlose_rate`: Overall win-lose rate

**비교 목적**:
- Selector training이 있는 경우 (55000-55038)와 없는 경우 (56000)의 성능 비교
- Selector의 기여도 정량화
- FedDPO와 동일한 방식으로 직접 preference learning 수행

---

## 업데이트 이력

- **2026-01-26**: 초기 문서 작성, 현재 진행 상황 기록
- **2026-01-XX**: 문서 통합 및 정리 완료