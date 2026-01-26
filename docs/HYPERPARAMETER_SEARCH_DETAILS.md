# Hyperparameter Search - Detailed Configurations

이 문서는 하이퍼파라미터 서치의 각 실험에 사용된 정확한 하이퍼파라미터 값을 상세히 기록합니다.

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

---

## Phase 1: Orthogonal Loss Parameters (52000-52006)

**목적**: Orthogonal loss의 weight와 prototype scale을 탐색하여 클라이언트 간 preference disentanglement를 최적화

**탐색 파라미터**:
- `vpl_orthogonal_weight`: [0.2, 1.0, 5.0]
- `vpl_orthogonal_orthonorm_weight`: [0.0, 0.1, 0.5]
- `vpl_prototype_scale`: [2.0, 5.0, 10.0]

**고정 파라미터**:
- `vpl_kl_weight`: 0.1
- `vpl_gp_temperature`: 1.0
- `lr`: 0.0001

### 실험별 상세 설정

| TID | orthogonal_weight | orthonorm_weight | prototype_scale | kl_weight | gp_temperature | lr |
|-----|------------------|------------------|-----------------|-----------|----------------|-----|
| 52000 | 0.2 | 0.1 | 5.0 | 0.1 | 1.0 | 0.0001 |
| 52001 | 1.0 | 0.1 | 5.0 | 0.1 | 1.0 | 0.0001 |
| 52002 | 5.0 | 0.1 | 5.0 | 0.1 | 1.0 | 0.0001 |
| 52003 | 1.0 | 0.0 | 5.0 | 0.1 | 1.0 | 0.0001 |
| 52004 | 1.0 | 0.5 | 5.0 | 0.1 | 1.0 | 0.0001 |
| 52005 | 1.0 | 0.1 | 2.0 | 0.1 | 1.0 | 0.0001 |
| 52006 | 1.0 | 0.1 | 10.0 | 0.1 | 1.0 | 0.0001 |

**설명**:
- **52000-52002**: `orthogonal_weight` 탐색 (0.2 → 1.0 → 5.0)
- **52003-52004**: `orthonorm_weight` 탐색 (0.0 → 0.5, baseline 0.1은 52001)
- **52005-52006**: `prototype_scale` 탐색 (2.0 → 10.0, baseline 5.0은 52001)

**최적값 (Phase 1 결과 기반)**:
- `vpl_orthogonal_weight`: 1.0
- `vpl_orthogonal_orthonorm_weight`: 0.1
- `vpl_prototype_scale`: 5.0

---

## Phase 2: VPL Core Parameters (52007-52013)

**목적**: VPL-GP의 핵심 하이퍼파라미터인 KL weight와 Gumbel-Softmax temperature를 탐색

**탐색 파라미터**:
- `vpl_kl_weight`: [0.02, 0.05, 0.1, 0.2]
- `vpl_gp_temperature`: [0.5, 1.0, 2.0, 5.0]

**Phase 1 최적값 사용**:
- `vpl_orthogonal_weight`: 1.0
- `vpl_orthogonal_orthonorm_weight`: 0.1
- `vpl_prototype_scale`: 5.0
- `lr`: 0.0001

### 실험별 상세 설정

| TID | kl_weight | gp_temperature | orthogonal_weight | orthonorm_weight | prototype_scale | lr |
|-----|-----------|----------------|-------------------|------------------|-----------------|-----|
| 52007 | 0.02 | 1.0 | 1.0 | 0.1 | 5.0 | 0.0001 |
| 52008 | 0.05 | 1.0 | 1.0 | 0.1 | 5.0 | 0.0001 |
| 52009 | 0.1 | 1.0 | 1.0 | 0.1 | 5.0 | 0.0001 |
| 52010 | 0.2 | 1.0 | 1.0 | 0.1 | 5.0 | 0.0001 |
| 52011 | 0.1 | 0.5 | 1.0 | 0.1 | 5.0 | 0.0001 |
| 52012 | 0.1 | 2.0 | 1.0 | 0.1 | 5.0 | 0.0001 |
| 52013 | 0.1 | 5.0 | 1.0 | 0.1 | 5.0 | 0.0001 |

**설명**:
- **52007-52010**: `kl_weight` 탐색 (0.02 → 0.05 → 0.1 → 0.2)
- **52011-52013**: `gp_temperature` 탐색 (0.5 → 2.0 → 5.0, baseline 1.0은 52009)

**최적값 (Phase 2 결과 기반)**:
- `vpl_kl_weight`: 0.1
- `vpl_gp_temperature`: 1.0

---

## Phase 3: Learning Rate (52014-52016)

**목적**: 최적 learning rate를 탐색

**탐색 파라미터**:
- `lr`: [0.00005, 0.0001, 0.0002]

**Phase 1-2 최적값 사용**:
- `vpl_orthogonal_weight`: 1.0
- `vpl_orthogonal_orthonorm_weight`: 0.1
- `vpl_prototype_scale`: 5.0
- `vpl_kl_weight`: 0.1
- `vpl_gp_temperature`: 1.0

### 실험별 상세 설정

| TID | lr | kl_weight | gp_temperature | orthogonal_weight | orthonorm_weight | prototype_scale |
|-----|----|-----------|----------------|------------------|-----------------|-----------------|
| 52014 | 0.00005 | 0.1 | 1.0 | 1.0 | 0.1 | 5.0 |
| 52015 | 0.0001 | 0.1 | 1.0 | 1.0 | 0.1 | 5.0 |
| 52016 | 0.0002 | 0.1 | 1.0 | 1.0 | 0.1 | 5.0 |

**최적값 (Phase 3 결과 기반)**:
- `lr`: 0.0001

---

## Phase 4: Combined Best Parameters (52017)

**목적**: Phase 1-3에서 찾은 최적 하이퍼파라미터 조합으로 최종 검증

### 최종 하이퍼파라미터

| 파라미터 | 값 | 출처 |
|---------|-----|------|
| `vpl_orthogonal_weight` | 1.0 | Phase 1 최적값 |
| `vpl_orthogonal_orthonorm_weight` | 0.1 | Phase 1 최적값 |
| `vpl_prototype_scale` | 5.0 | Phase 1 최적값 |
| `vpl_kl_weight` | 0.1 | Phase 2 최적값 |
| `vpl_gp_temperature` | 1.0 | Phase 2 최적값 |
| `lr` | 0.0001 | Phase 3 최적값 |

---

## RL Experiments (53000-53017)

각 selector 실험(52000-52017)에 대응하는 RL 실험이 있습니다.

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

| RL TID | Selector TID | Selector Phase | Selector 하이퍼파라미터 |
|--------|-------------|----------------|----------------------|
| 53000 | 52000 | Phase 1 | orthogonal_weight=0.2, prototype_scale=5.0 |
| 53001 | 52001 | Phase 1 | orthogonal_weight=1.0, prototype_scale=5.0 |
| 53002 | 52002 | Phase 1 | orthogonal_weight=5.0, prototype_scale=5.0 |
| 53003 | 52003 | Phase 1 | orthogonal_weight=1.0, orthonorm_weight=0.0 |
| 53004 | 52004 | Phase 1 | orthogonal_weight=1.0, orthonorm_weight=0.5 |
| 53005 | 52005 | Phase 1 | orthogonal_weight=1.0, prototype_scale=2.0 |
| 53006 | 52006 | Phase 1 | orthogonal_weight=1.0, prototype_scale=10.0 |
| 53007 | 52007 | Phase 2 | kl_weight=0.02, gp_temperature=1.0 |
| 53008 | 52008 | Phase 2 | kl_weight=0.05, gp_temperature=1.0 |
| 53009 | 52009 | Phase 2 | kl_weight=0.1, gp_temperature=1.0 |
| 53010 | 52010 | Phase 2 | kl_weight=0.2, gp_temperature=1.0 |
| 53011 | 52011 | Phase 2 | kl_weight=0.1, gp_temperature=0.5 |
| 53012 | 52012 | Phase 2 | kl_weight=0.1, gp_temperature=2.0 |
| 53013 | 52013 | Phase 2 | kl_weight=0.1, gp_temperature=5.0 |
| 53014 | 52014 | Phase 3 | lr=0.00005 |
| 53015 | 52015 | Phase 3 | lr=0.0001 |
| 53016 | 52016 | Phase 3 | lr=0.0002 |
| 53017 | 52017 | Phase 4 | 최적 조합 (모든 파라미터) |

---

## 최종 최적 하이퍼파라미터 (Main Table 실험용)

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
