# VPL-GP Hyperparameter Search — 통합 문서

VPL-GP 하이퍼파라미터 탐색의 **Phase 구간, TID–파라미터 매핑, 실행 방법**을 한 문서로 정리합니다.

---

## 1. 요약

| 모델 | Selector TID | RL TID | Phase |
|------|--------------|--------|--------|
| **Gemma-2B** | 54000–54047 | 55000–55047 | 6 (Phase 1–6) |
| **Qwen2-0.5B** | 54100–54138 | 55100–55138 | 5 (Phase 1–5) |

**실행**: Gemma `bash scripts/hpsearch/submit_selector.sh <phase>`, 완료 후 `bash scripts/hpsearch/submit_rl.sh <phase>`  
Qwen `bash scripts/hpsearch/submit_selector_qwen.sh <phase>`, 완료 후 `bash scripts/hpsearch/submit_rl_qwen.sh <phase>`

**상세 스크립트·파일 구조**: `scripts/hpsearch/README.md`

---

## 2. 공통 설정

| 항목 | 값 |
|------|-----|
| Client Num | 10 |
| Sample Client Num | 5 |
| Total Rounds | 50 |
| Batch Size | 8 |
| Grad Accum Step | 4 |
| Local Update Steps | 30 |
| VPL Latent Dim | 32 |
| VPL Feature Method | choice_logits |
| VPL Use Difference Only | True |
| VPL Num Prototypes | 2 |

**체크포인트**: `$CHECKPOINT_DIR` (스크립트에서 `/hdd/hdd3/kjb/checkpoints` 또는 `$WORK_DIR/checkpoints`)  
Selector: `hhrl_choice_<model>_fedbiscuit_u3_vplgp_ortho_t{TID}.ckpt` (최종: `final_*`)  
RL: `hhrl_rlhf_<model>_choice_vplgp_t{RL_TID}.ckpt`

**WandB**: Selector `fvpl-selector`, RL `fvpl-rl`

---

## 3. Gemma-2B (54000 / 55000)

### Phase 1: Orthogonal Loss — 54000–54006 / 55000–55006 (7개)

| TID (Sel/RL) | vpl_orthogonal_weight | vpl_orthogonal_orthonorm_weight | vpl_prototype_scale | vpl_kl_weight | vpl_gp_temperature |
|--------------|------------------------|----------------------------------|----------------------|---------------|--------------------|
| 54000/55000 | **0.2** | 0.1 | 5.0 | 0.1 | 1.0 |
| 54001/55001 | **1.0** | 0.1 | 5.0 | 0.1 | 1.0 |
| 54002/55002 | **5.0** | 0.1 | 5.0 | 0.1 | 1.0 |
| 54003/55003 | 1.0 | **0.0** | 5.0 | 0.1 | 1.0 |
| 54004/55004 | 1.0 | **0.5** | 5.0 | 0.1 | 1.0 |
| 54005/55005 | 1.0 | 0.1 | **2.0** | 0.1 | 1.0 |
| 54006/55006 | 1.0 | 0.1 | **10.0** | 0.1 | 1.0 |

### Phase 2: VPL Core — 54007–54013 / 55007–55013 (7개)

| TID (Sel/RL) | vpl_orthogonal_weight | vpl_orthogonal_orthonorm_weight | vpl_prototype_scale | vpl_kl_weight | vpl_gp_temperature |
|--------------|------------------------|----------------------------------|----------------------|---------------|--------------------|
| 54007/55007 | 1.0 | 0.1 | 5.0 | **0.02** | 1.0 |
| 54008/55008 | 1.0 | 0.1 | 5.0 | **0.05** | 1.0 |
| 54009/55009 | 1.0 | 0.1 | 5.0 | **0.1** | 1.0 |
| 54010/55010 | 1.0 | 0.1 | 5.0 | **0.2** | 1.0 |
| 54011/55011 | 1.0 | 0.1 | 5.0 | 0.1 | **0.5** |
| 54012/55012 | 1.0 | 0.1 | 5.0 | 0.1 | **2.0** |
| 54013/55013 | 1.0 | 0.1 | 5.0 | 0.1 | **5.0** |

### Phase 3: Refinement (54005/54008/54013 기반) — 54014–54022 / 55014–55022 (9개)

| TID (Sel/RL) | vpl_orthogonal_weight | vpl_orthogonal_orthonorm_weight | vpl_prototype_scale | vpl_kl_weight | vpl_gp_temperature |
|--------------|------------------------|----------------------------------|----------------------|---------------|--------------------|
| 54014/55014 | 1.0 | 0.1 | **1.0** | 0.1 | 1.0 |
| 54015/55015 | 1.0 | 0.1 | **2.0** | 0.1 | 1.0 |
| 54016/55016 | 1.0 | 0.1 | **3.0** | 0.1 | 1.0 |
| 54017/55017 | 1.0 | 0.1 | 2.0 | **0.03** | 1.0 |
| 54018/55018 | 1.0 | 0.1 | 2.0 | **0.05** | 1.0 |
| 54019/55019 | 1.0 | 0.1 | 2.0 | **0.08** | 1.0 |
| 54020/55020 | 1.0 | 0.1 | 2.0 | 0.05 | **3.0** |
| 54021/55021 | 1.0 | 0.1 | 2.0 | 0.05 | **5.0** |
| 54022/55022 | 1.0 | 0.1 | 2.0 | 0.05 | **7.0** |

### Phase 4: Learning Rate — 54023–54025 / 55023–55025 (3개)

| TID (Sel/RL) | vpl_prototype_scale | vpl_kl_weight | vpl_gp_temperature | lr |
|--------------|----------------------|---------------|--------------------|--------|
| 54023/55023 | 2.0 | 0.1 | 1.0 | **0.00005** |
| 54024/55024 | 2.0 | 0.1 | 1.0 | **0.0001** |
| 54025/55025 | 2.0 | 0.1 | 1.0 | **0.0002** |

### Phase 5: Combined Best — 54026 / 55026 (1개)

Phase 1–4 최적/중간값 조합.

| TID (Sel/RL) | vpl_orthogonal_weight | vpl_orthogonal_orthonorm_weight | vpl_prototype_scale | vpl_kl_weight | vpl_gp_temperature | lr |
|--------------|------------------------|----------------------------------|----------------------|---------------|--------------------|--------|
| 54026/55026 | 1.0 | 0.1 | **2.0** | **0.05** | **5.0** | **0.0001** |

**실행**: `bash scripts/hpsearch/submit_selector.sh 5` → 완료 후 `bash scripts/hpsearch/submit_rl.sh 5`

### Phase 6: Fine-grained — 54027–54047 / 55027–55047 (21개)

| Sub-phase | TID (Sel/RL) | 탐색 파라미터 | 값 |
|-----------|--------------|----------------|-----|
| 6.1 Orthogonal Weight | 54027–54033 / 55027–55033 | vpl_orthogonal_weight | 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0 |
| 6.2 Orthonorm Weight | 54034–54039 / 55034–55039 | vpl_orthogonal_orthonorm_weight | 0.0, 0.05, 0.1, 0.2, 0.5, 1.0 |
| 6.3 KL Weight | 54040–54046 / 55040–55046 | vpl_kl_weight | 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0 |
| 6.4 최적 조합 검증 | 54047 / 55047 | (기본값) | 1개 |

---

## 4. Qwen2-0.5B (54100 / 55100)

Phase 2부터 **54105(prototype_scale=2.0)** 기준 사용.

### Phase 1: Orthogonal Loss — 54100–54106 / 55100–55106 (7개)

| TID (Sel/RL) | vpl_orthogonal_weight | vpl_orthogonal_orthonorm_weight | vpl_prototype_scale | vpl_kl_weight | vpl_gp_temperature |
|--------------|------------------------|----------------------------------|----------------------|---------------|--------------------|
| 54100/55100 | **0.2** | 0.1 | 5.0 | 0.1 | 1.0 |
| 54101/55101 | **1.0** | 0.1 | 5.0 | 0.1 | 1.0 |
| 54102/55102 | **5.0** | 0.1 | 5.0 | 0.1 | 1.0 |
| 54103/55103 | 1.0 | **0.0** | 5.0 | 0.1 | 1.0 |
| 54104/55104 | 1.0 | **0.5** | 5.0 | 0.1 | 1.0 |
| 54105/55105 | 1.0 | 0.1 | **2.0** | 0.1 | 1.0 |
| 54106/55106 | 1.0 | 0.1 | **10.0** | 0.1 | 1.0 |

### Phase 2: VPL Core (prototype_scale=2.0 고정) — 54107–54113 / 55107–55113 (7개)

| TID (Sel/RL) | vpl_prototype_scale | vpl_kl_weight | vpl_gp_temperature |
|--------------|----------------------|---------------|--------------------|
| 54107/55107 | **2.0** | **0.02** | 1.0 |
| 54108/55108 | 2.0 | **0.05** | 1.0 |
| 54109/55109 | 2.0 | **0.1** | 1.0 |
| 54110/55110 | 2.0 | **0.2** | 1.0 |
| 54111/55111 | 2.0 | 0.1 | **0.5** |
| 54112/55112 | 2.0 | 0.1 | **2.0** |
| 54113/55113 | 2.0 | 0.1 | **5.0** |

### Phase 3: Learning Rate — 54114–54116 / 55114–55116 (3개)

| TID (Sel/RL) | lr |
|--------------|--------|
| 54114/55114 | **0.00005** |
| 54115/55115 | **0.0001** |
| 54116/55116 | **0.0002** |

### Phase 4: Combined Best — 54117 / 55117 (1개)

Phase 1·2 최적/중간값 조합.

| TID (Sel/RL) | vpl_orthogonal_weight | vpl_orthogonal_orthonorm_weight | vpl_prototype_scale | vpl_kl_weight | vpl_gp_temperature | lr |
|--------------|------------------------|----------------------------------|----------------------|---------------|--------------------|--------|
| 54117/55117 | 1.0 | 0.1 | **2.0** | **0.05** | **1.0** | **0.0001** |

**실행**: `bash scripts/hpsearch/submit_selector_qwen.sh 4` → 완료 후 `bash scripts/hpsearch/submit_rl_qwen.sh 4`

### Phase 5: Fine-grained — 54118–54138 / 55118–55138 (21개)

| Sub-phase | TID (Sel/RL) | 탐색 파라미터 | 값 |
|-----------|--------------|----------------|-----|
| Orthogonal Weight | 54118–54124 / 55118–55124 | vpl_orthogonal_weight | 0.1 … 10.0 |
| Orthonorm Weight | 54125–54130 / 55125–55130 | vpl_orthogonal_orthonorm_weight | 0.0 … 1.0 |
| KL Weight | 54131–54137 / 55131–55137 | vpl_kl_weight | 0.01 … 1.0 |
| 최적 조합 검증 | 54138 / 55138 | (기본값) | 1개 |

---

## 5. 실행 요약

| 모델 | Selector 제출 | RL 제출 (Selector 완료 후) |
|------|----------------|----------------------------|
| Gemma | `bash scripts/hpsearch/submit_selector.sh <1–6>` | `bash scripts/hpsearch/submit_rl.sh <1–6>` |
| Qwen | `bash scripts/hpsearch/submit_selector_qwen.sh <1–5>` | `bash scripts/hpsearch/submit_rl_qwen.sh <1–5>` |

**개별 실험 (sbatch)**:
```bash
sbatch scripts/hpsearch/run_selector_hpsearch.sh 54000
sbatch scripts/hpsearch/run_rl_hpsearch.sh 55000 54000

sbatch scripts/hpsearch/run_selector_hpsearch_qwen.sh 54100
sbatch scripts/hpsearch/run_rl_hpsearch_qwen.sh 55100 54100
```

**WORK_DIR**: `submit_*.sh`는 `WORK_DIR="/home2/jbkoo/ppfl"` 사용. 다른 경로는 스크립트 내 수정 또는 위 sbatch로 직접 실행.

---

## 6. 모니터링

```bash
squeue -u $USER
tail -f outputs/54026.log
tail -f outputs/55026.log
```

진행 상황·결과: WandB `fvpl-selector`, `fvpl-rl` 및 `outputs/<TID>.log` 참고.

---

## 7. 기타

- **Local RL only (baseline)**: TID 56000 등. Selector 없이 RL만 실행. `scripts/hpsearch/run_rl_local_only.sh` 참고.
- **Config 생성**: `run_selector_hpsearch.sh` / `run_selector_hpsearch_qwen.sh`가 TID에 따라 `cfg/hpsearch/vpl-gp/phase_<TID>.yaml` 또는 `cfg/hpsearch/vpl-gp-qwen/phase_<TID>.yaml` 생성.
