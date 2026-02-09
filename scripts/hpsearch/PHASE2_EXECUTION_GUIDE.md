# Phase 2 실행 가이드

Phase 1 최적값(55001/54001)을 기반으로 Phase 2를 실행합니다.

## Phase 2 설정 요약

### Phase 1 최적값 (기본값으로 사용)
- `vpl_orthogonal_weight`: **1.0** ✅
- `vpl_orthogonal_orthonorm_weight`: **0.1** ✅
- `vpl_prototype_scale`: **5.0** ✅
- `lr`: **0.0001** ✅

### Phase 2 탐색 파라미터
- `vpl_kl_weight`: [0.02, 0.05, 0.1, 0.2] (54007-54010)
- `vpl_gp_temperature`: [0.5, 1.0, 2.0, 5.0] (54011-54013, baseline 1.0은 54009)

## 실험 범위

| TID | kl_weight | gp_temperature | 설명 |
|-----|-----------|----------------|------|
| 54007 | 0.02 | 1.0 | KL weight 탐색 (낮음) |
| 54008 | 0.05 | 1.0 | KL weight 탐색 |
| 54009 | 0.1 | 1.0 | KL weight 탐색 (baseline) |
| 54010 | 0.2 | 1.0 | KL weight 탐색 (높음) |
| 54011 | 0.1 | 0.5 | Temperature 탐색 (낮음) |
| 54012 | 0.1 | 2.0 | Temperature 탐색 (높음) |
| 54013 | 0.1 | 5.0 | Temperature 탐색 (매우 높음) |

## 실행 방법

### 1. Selector Training 실행

```bash
# Cluster에서 Phase 2 Selector 실험 제출 (7개)
bash scripts/hpsearch/submit_selector.sh 2
```

**실행 범위**: Selector TID 54007-54013 (7개 실험)

### 2. Selector 완료 확인

```bash
# Checkpoint 확인
ls -lh /home2/jbkoo/ppfl/checkpoints/*54007*.ckpt
ls -lh /home2/jbkoo/ppfl/checkpoints/*54008*.ckpt
# ... (54009-54013도 확인)

# 또는 패턴으로 확인
ls -lh /home2/jbkoo/ppfl/checkpoints/*vplgp_ortho_t5400[7-9]*.ckpt
ls -lh /home2/jbkoo/ppfl/checkpoints/*vplgp_ortho_t5401[0-3]*.ckpt
```

**예상 checkpoint 이름**:
- `hhrl_choice_gemma-2b_fedbiscuit_u3_vplgp_ortho_t54007.ckpt`
- `final_hhrl_choice_gemma-2b_fedbiscuit_u3_vplgp_ortho_t54007.ckpt` (최종)
- `40_hhrl_choice_gemma-2b_fedbiscuit_u3_vplgp_ortho_t54007.ckpt` (중간)

### 3. RL Training 실행

Selector checkpoint가 모두 준비되면 RL 실험을 실행합니다:

```bash
# Cluster에서 Phase 2 RL 실험 제출 (7개)
bash scripts/hpsearch/submit_rl.sh 2
```

**실행 범위**: RL TID 55007-55013 (7개 실험)

**매칭 관계**:
- RL 55007 → Selector 54007
- RL 55008 → Selector 54008
- ... (55013 → 54013)

## 모니터링

### SLURM 작업 상태
```bash
squeue -u $USER
```

### 로그 확인
```bash
# Selector 로그
tail -f outputs/54007.log
tail -f outputs/54008.log
# ... (54009-54013)

# RL 로그
tail -f outputs/55007.log
tail -f outputs/55008.log
# ... (55009-55013)

# SLURM 출력
tail -f /home2/jbkoo/slurm/logs/slurm-*.out
```

### WandB 모니터링
- **Selector**: `fvpl-selector` 프로젝트
- **RL**: `fvpl-rl` 프로젝트

## 예상 결과

Phase 2 완료 후 다음을 확인합니다:

1. **최적 KL weight**: 54007-54010 중 가장 낮은 loss
2. **최적 Temperature**: 54011-54013 중 가장 낮은 loss (baseline 54009와 비교)
3. **최종 최적값**: Phase 1 + Phase 2 최적값 조합

## 다음 단계

Phase 2 완료 후:
- Phase 3 (Learning Rate 탐색) 실행
- 또는 Phase 4 (Combined Best Parameters)로 바로 진행 가능

## 주의사항

1. **Phase 1 완료 확인**: Phase 2는 Phase 1 최적값을 사용하므로, Phase 1이 완료되어야 합니다.
2. **Selector 완료 후 RL**: RL 실험은 해당 selector checkpoint가 필요합니다.
3. **Checkpoint 경로**: `/home2/jbkoo/ppfl/checkpoints/`에 저장됩니다.
