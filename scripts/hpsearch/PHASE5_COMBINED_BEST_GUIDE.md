# Combined Best (최종 스테이지) 실행 가이드 — Gemma & Qwen

Phase 1~4(Gemma) / Phase 1~2(Qwen) 결과에서 **중간값이 가장 좋았을 때**를 기준으로 한 **Combined Best** 파라미터로 실험합니다.

## 설정 요약

### Gemma — Phase 5 (TID 54026 / 55026)

| 파라미터 | 값 | 비고 |
|----------|-----|------|
| vpl_orthogonal_weight | 1.0 | Phase 1 기준 |
| vpl_orthogonal_orthonorm_weight | 0.1 | Phase 1 기준 |
| vpl_prototype_scale | **2.0** | 54005, 54015 |
| vpl_kl_weight | **0.05** | 54008, 54018 |
| vpl_gp_temperature | **5.0** | 54013, 54021 |
| lr | **0.0001** | Phase 4 중간(54024) |

### Qwen — Phase 4 (TID 54117 / 55117)

| 파라미터 | 값 | 비고 |
|----------|-----|------|
| vpl_orthogonal_weight | 1.0 | Phase 1 기준 |
| vpl_orthogonal_orthonorm_weight | 0.1 | Phase 1 기준 |
| vpl_prototype_scale | **2.0** | 54105 기준 |
| vpl_kl_weight | **0.05** | Phase 2 중간 |
| vpl_gp_temperature | **1.0** | Phase 2 기준 |
| lr | **0.0001** | 기본값 |

---

## 실행 방법

### 1. Gemma Combined Best (Phase 5)

**Selector 1개 (54026)** 완료 후 **RL 1개 (55026)** 실행.

```bash
# 작업 디렉토리로 이동 (SLURM 클러스터에서는 /home2/jbkoo/ppfl)
cd $WORK_DIR   # 또는 cd /home/kjb/ppfl (로컬)

# 1) Selector 제출
bash scripts/hpsearch/submit_selector.sh 5

# 2) Selector 완료 확인 후 RL 제출
#    체크포인트: final_hhrl_choice_gemma-2b_fedbiscuit_u3_vplgp_ortho_t54026.ckpt
bash scripts/hpsearch/submit_rl.sh 5
```

**수동 실행 (sbatch 직접 사용):**
```bash
sbatch scripts/hpsearch/run_selector_hpsearch.sh 54026
# 완료 후
sbatch scripts/hpsearch/run_rl_hpsearch.sh 55026 54026
```

### 2. Qwen Combined Best (Phase 4)

**Selector 1개 (54117)** 완료 후 **RL 1개 (55117)** 실행.

```bash
cd $WORK_DIR

# 1) Selector 제출
bash scripts/hpsearch/submit_selector_qwen.sh 4

# 2) Selector 완료 확인 후 RL 제출
#    체크포인트: final_hhrl_choice_qwen2_fedbiscuit_u3_vplgp_ortho_t54117.ckpt
bash scripts/hpsearch/submit_rl_qwen.sh 4
```

**수동 실행:**
```bash
sbatch scripts/hpsearch/run_selector_hpsearch_qwen.sh 54117
# 완료 후
sbatch scripts/hpsearch/run_rl_hpsearch_qwen.sh 55117 54117
```

---

## 체크포인트 경로

- **Gemma**: `$CHECKPOINT_DIR/hhrl_choice_gemma-2b_fedbiscuit_u3_vplgp_ortho_t54026.ckpt` (final_ 접두사로 최종 저장)
- **Qwen**: `$CHECKPOINT_DIR/hhrl_choice_qwen2_fedbiscuit_u3_vplgp_ortho_t54117.ckpt`

`$CHECKPOINT_DIR`는 `run_*_hpsearch*.sh`에서 `/hdd/hdd3/kjb/checkpoints` 또는 `$WORK_DIR/checkpoints`로 설정됩니다.

---

## 로그 및 모니터링

```bash
# Gemma
tail -f outputs/54026.log   # Selector
tail -f outputs/55026.log   # RL

# Qwen
tail -f outputs/54117.log   # Selector
tail -f outputs/55117.log   # RL

# SLURM 작업 목록
squeue -u $USER
```

---

## 참고

- **WORK_DIR**: `submit_*.sh`는 `WORK_DIR="/home2/jbkoo/ppfl"`로 되어 있습니다. 다른 경로에서 실행할 경우 스크립트 내 `WORK_DIR`를 수정하거나, 수동으로 `sbatch scripts/hpsearch/run_selector_hpsearch.sh 54026` 형태로 실행하세요.
- Combined Best 파라미터는 `run_selector_hpsearch.sh`(Gemma 54026), `run_selector_hpsearch_qwen.sh`(Qwen 54117) 안에 반영되어 있습니다.
