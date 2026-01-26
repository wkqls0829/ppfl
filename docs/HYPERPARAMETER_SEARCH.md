# VPL-GP Hyperparameter Search Documentation

## 개요

이 문서는 **Variational Preference Learning with Gumbel-Softmax Prior (VPL-GP)** 모델의 하이퍼파라미터 탐색 실험에 대한 전체 계획, 진행 상황, 그리고 결과 분석 가이드를 제공합니다.

## 실험 구조

하이퍼파라미터 서치는 **2단계 구조**로 진행됩니다:

1. **Selector Training (Binary Classification)**: 각 하이퍼파라미터 조합에 대해 selector 모델 학습
   - Task ID: 52000-52017
   - WandB Project: `fvpl-selector`
   - Config: `cfg/hpsearch/vpl-gp/phase*_*.yaml`
   - Scripts: `scripts/hpsearch/vpl-gp/phase*_*.sh`

2. **RL Training (RLHF)**: 완료된 selector checkpoint를 사용하여 RLHF 학습
   - Task ID: 53000-53017
   - WandB Project: `fvpl-rl`
   - Config: `cfg/hpsearch/vpl-gp-rl/hrl_*.yaml`
   - Scripts: `scripts/hpsearch/vpl-gp-rl/hrl_*.sh`

## Phase별 실험 계획

### Phase 1: Orthogonal Loss Weight 탐색 (52000-52006)

**목적**: Orthogonal loss의 weight와 prototype scale을 탐색하여 클라이언트 간 preference disentanglement를 최적화합니다.

**하이퍼파라미터 공간**:
- `vpl_orthogonal_weight`: [0.2, 1.0, 5.0]
- `vpl_prototype_scale`: [2.0, 5.0, 10.0]
- 고정값: `vpl_kl_weight=0.1`, `vpl_gp_temperature=1.0`, `lr=0.0001`

| TID | orthogonal_weight | prototype_scale | kl_weight | temperature | lr | Status |
|-----|-------------------|-----------------|-----------|------------|-----|--------|
| 52000 | 0.2 | 5.0 | 0.1 | 1.0 | 0.0001 | Running |
| 52001 | 1.0 | 5.0 | 0.1 | 1.0 | 0.0001 | Completed |
| 52002 | 5.0 | 5.0 | 0.1 | 1.0 | 0.0001 | Completed |
| 52003 | 1.0 | 5.0 | 0.1 | 1.0 | 0.0001 | Completed |
| 52004 | 1.0 | 5.0 | 0.1 | 1.0 | 0.0001 | Completed |
| 52005 | 1.0 | 2.0 | 0.1 | 1.0 | 0.0001 | Completed |
| 52006 | 1.0 | 10.0 | 0.1 | 1.0 | 0.0001 | Completed |

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

### Phase 2: VPL Core Parameters 탐색 (52007-52013)

**목적**: VPL-GP의 핵심 하이퍼파라미터인 KL weight와 Gumbel-Softmax temperature를 탐색합니다.

**하이퍼파라미터 공간**:
- `vpl_kl_weight`: [0.02, 0.1, 0.5]
- `vpl_gp_temperature`: [0.5, 1.0, 2.0]
- Phase 1 최적값 사용: `vpl_orthogonal_weight=1.0`, `vpl_prototype_scale=5.0`

| TID | orthogonal_weight | prototype_scale | kl_weight | temperature | lr | Status |
|-----|-------------------|-----------------|-----------|------------|-----|--------|
| 52007 | 1.0 | 5.0 | 0.02 | 1.0 | 0.0001 | Running |
| 52008 | 1.0 | 5.0 | 0.1 | 1.0 | 0.0001 | Completed |
| 52009 | 1.0 | 5.0 | 0.5 | 1.0 | 0.0001 | Completed |
| 52010 | 1.0 | 5.0 | 0.1 | 0.5 | 0.0001 | Completed |
| 52011 | 1.0 | 5.0 | 0.1 | 2.0 | 0.0001 | Completed |
| 52012 | 1.0 | 5.0 | 0.1 | 1.0 | 0.0001 | Completed |
| 52013 | 1.0 | 5.0 | 0.1 | 1.0 | 0.0001 | Completed |

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

### Phase 3: Learning Rate 탐색 (52014-52016)

**목적**: 최적 learning rate를 탐색합니다. Phase 1-2의 최적 하이퍼파라미터를 사용합니다.

**하이퍼파라미터 공간**:
- `lr`: [0.00005, 0.0001, 0.0002]
- Phase 1-2 최적값 사용

| TID | orthogonal_weight | prototype_scale | kl_weight | temperature | lr | Status |
|-----|-------------------|-----------------|-----------|------------|-----|--------|
| 52014 | 1.0 | 5.0 | 0.1 | 1.0 | 0.00005 | Completed |
| 52015 | 1.0 | 5.0 | 0.1 | 1.0 | 0.0001 | Running |
| 52016 | 1.0 | 5.0 | 0.1 | 1.0 | 0.0002 | Completed |

**평가 지표**:
- `train_avg_loss`: 수렴 속도와 최종 loss
- Training stability
- Convergence speed

**분석 포인트**:
1. Learning rate가 수렴 속도에 미치는 영향
2. 최적 learning rate 결정

---

### Phase 4: Best Parameters Combination (52017)

**목적**: Phase 1-3에서 찾은 최적 하이퍼파라미터 조합으로 최종 실험을 수행합니다.

| TID | orthogonal_weight | prototype_scale | kl_weight | temperature | lr | Status |
|-----|-------------------|-----------------|-----------|------------|-----|--------|
| 52017 | 1.0 | 5.0 | 0.1 | 1.0 | 0.0001 | Running |

**참고**: Phase 1-3 결과를 바탕으로 하이퍼파라미터를 업데이트해야 합니다.

---

## RL Experiments (53000-53017)

각 selector 실험(52000-52017)에 대응하는 RL 실험이 있습니다. RL 실험은 해당 selector의 checkpoint를 사용하여 RLHF 학습을 수행합니다.

### RL 실험 설정

**공통 설정**:
- `rlhf_use_variational_selection: true`: Selector에서 학습한 z를 사용하여 conditional selection
- `rlhf_use_variational_generation: false`: Generation은 z를 사용하지 않음
- `reward_coeff: 0.1`: Reward coefficient
- `grad_accum_step: 4`: Gradient accumulation steps
- `max_samples_for_reward: 30`: Reward 평가 샘플 수 (성능 최적화)
- `use_gpt_api_for_winrate: true`: GPT API를 사용한 winrate 평가 (성능 최적화)
- `use_baseline_model_for_winrate: true`: Baseline 모델과 비교
- `openai_model: gpt-4o-mini`: Winrate 평가용 모델

**평가 지표**:
- `avg_helpfulness`: Helpfulness score
- `avg_harmlessness`: Harmlessness score
- `helpfulness_winrate`: Helpful response win rate
- `harmlessness_winrate`: Harmless response win rate
- `avg_winlose_rate`: Overall win-lose rate

### RL 실험 매핑

| RL TID | Selector TID | Selector Phase | Status |
|--------|-------------|----------------|--------|
| 53000 | 52000 | Phase 1 | Running |
| 53001 | 52001 | Phase 1 | Running |
| 53002 | 52002 | Phase 1 | Running |
| 53003 | 52003 | Phase 1 | Running |
| 53004 | 52004 | Phase 1 | Running |
| 53005 | 52005 | Phase 1 | Running |
| 53006 | 52006 | Phase 1 | Not started |
| 53007 | 52007 | Phase 2 | Running |
| 53008 | 52008 | Phase 2 | Not started |
| 53009 | 52009 | Phase 2 | Not started |
| 53010 | 52010 | Phase 2 | Not started |
| 53011 | 52011 | Phase 2 | Not started |
| 53012 | 52012 | Phase 2 | Not started |
| 53013 | 52013 | Phase 2 | Not started |
| 53014 | 52014 | Phase 3 | Not started |
| 53015 | 52015 | Phase 3 | Not started (selector running) |
| 53016 | 52016 | Phase 3 | Not started |
| 53017 | 52017 | Phase 4 | Not started (selector running) |

---

## 현재 진행 상황 (2026-01-26)

### Selector Experiments

- **완료**: 15개 (52001-52006, 52008-52014, 52016)
- **실행 중**: 3개 (52000, 52015, 52017)
- **대기 중**: 0개

### RL Experiments

- **완료**: 0개
- **실행 중**: 7개 (53000-53005, 53007)
- **대기 중**: 11개 (53006, 53008-53014, 53016-53017)

### GPU 할당

- **GPU 0**: RL 53000 (거의 완료), Selector 52017 (실행 중)
- **GPU 1**: Selector 52015 (실행 중)
- **GPU 2**: RL 53001
- **GPU 3**: RL 53002
- **GPU 4**: RL 53007
- **GPU 5**: RL 53003
- **GPU 6**: RL 53004
- **GPU 7**: RL 53005

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

## 추가 실험 계획

### 1. Fine-grained 탐색

Phase 1-3 결과를 바탕으로 더 세밀한 탐색이 필요할 수 있습니다:

- **Phase 1 후보**: `orthogonal_weight` [0.5, 1.5, 2.0] 또는 `prototype_scale` [3.0, 7.0]
- **Phase 2 후보**: `kl_weight` [0.05, 0.2] 또는 `temperature` [0.8, 1.2]
- **Phase 3 후보**: `lr` [0.000075, 0.00015]

### 2. Ablation Studies

최적 하이퍼파라미터를 찾은 후, 각 컴포넌트의 기여도를 확인:

- **Orthogonal loss ablation**: `orthogonal_weight=0` (no orthogonal loss)
- **Gumbel-Softmax prior ablation**: Standard normal prior vs mixture prior
- **Feature difference ablation**: `vpl_use_feature_difference=false`

### 3. Cross-validation

최적 하이퍼파라미터의 일반화 성능 확인:

- 다른 데이터셋에서 테스트
- 다른 모델 크기에서 테스트
- 다른 클라이언트 수에서 테스트

### 4. RL-specific 하이퍼파라미터 탐색

Selector 하이퍼파라미터가 결정된 후, RL 학습의 하이퍼파라미터도 탐색:

- `reward_coeff`: [0.05, 0.1, 0.2]
- `grad_accum_step`: [2, 4, 8]
- RL learning rate

---

## 파일 구조

```
cfg/hpsearch/vpl-gp/
├── phase1_orthogonal_52000.yaml
├── phase1_orthogonal_52001.yaml
├── ...
├── phase2_vpl_core_52007.yaml
├── ...
├── phase3_lr_52014.yaml
├── ...
└── phase4_combined_52017.yaml

cfg/hpsearch/vpl-gp-rl/
├── hrl_53000.yaml
├── hrl_53001.yaml
└── ...

scripts/hpsearch/vpl-gp/
├── phase1_orthogonal_52000.sh
├── ...

scripts/hpsearch/vpl-gp-rl/
├── hrl_53000.sh
├── ...

outputs/
├── 52000.log
├── 52001.log
├── ...
├── 53000.log
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
tail -f outputs/52000.log

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

## 다음 단계

1. **Selector 실험 완료 대기**: 52000, 52015, 52017 완료 대기
2. **RL 실험 실행**: 남은 RL 실험들 순차 실행 (OOM 방지를 위해 하나씩)
3. **결과 분석**: 모든 실험이 완료되면 WandB에서 결과 분석
4. **최적 하이퍼파라미터 결정**: 분석 결과를 바탕으로 최적값 결정
5. **추가 실험 계획**: 필요시 fine-grained 탐색 또는 ablation study 수행

---

## 참고사항

- **OOM 방지**: RL 실험은 메모리 사용량이 크므로 GPU당 하나씩만 실행
- **체크포인트 확인**: RL 실험 실행 전 해당 selector checkpoint가 존재하는지 확인
- **로그 백업**: 중요한 실험의 로그는 별도로 백업 권장
- **WandB 동기화**: 실험 결과는 WandB에 자동으로 업로드되지만, 네트워크 문제 시 수동 동기화 필요

---

## 업데이트 이력

- **2026-01-26**: 초기 문서 작성, 현재 진행 상황 기록
