# Hyperparameter Search: Phase 및 TID 정리

FedVPA-GP 하이퍼파라미터 서치의 Phase 구간과 TID–파라미터 매핑입니다.

---

## 1. 요약 표

| 모델 | Selector TID 범위 | RL TID 범위 | Phase 수 |
|------|-------------------|-------------|----------|
| **Gemma-2B** | 54000–54047 | 55000–55047 | 6 |
| **Qwen 2**   | 54100–54138 | 55100–55138 | 5 |

**실행**: Gemma `bash scripts/hpsearch/submit_selector.sh <phase>`, RL `bash scripts/hpsearch/submit_rl.sh <phase>`  
Qwen: `submit_selector_qwen.sh` / `submit_rl_qwen.sh` (phase 1–5)

---

## 2. Gemma-2B (54000 / 55000 대)

### Phase 1: Orthogonal Loss Parameters  
**Selector 54000–54006 / RL 55000–55006 (7개)**

| TID (Sel/RL) | vpl_orthogonal_weight | vpl_orthogonal_orthonorm_weight | vpl_prototype_scale | vpl_kl_weight | vpl_gp_temperature |
|--------------|-----------------------|----------------------------------|----------------------|---------------|--------------------|
| 54000/55000 | **0.2**  | 0.1 | 5.0 | 0.1 | 1.0 |
| 54001/55001 | **1.0**  | 0.1 | 5.0 | 0.1 | 1.0 |
| 54002/55002 | **5.0**  | 0.1 | 5.0 | 0.1 | 1.0 |
| 54003/55003 | 1.0 | **0.0**  | 5.0 | 0.1 | 1.0 |
| 54004/55004 | 1.0 | **0.5**  | 5.0 | 0.1 | 1.0 |
| 54005/55005 | 1.0 | 0.1 | **2.0**  | 0.1 | 1.0 |
| 54006/55006 | 1.0 | 0.1 | **10.0** | 0.1 | 1.0 |

---

### Phase 2: VPL Core Parameters  
**Selector 54007–54013 / RL 55007–55013 (7개)**

| TID (Sel/RL) | vpl_orthogonal_weight | vpl_orthogonal_orthonorm_weight | vpl_prototype_scale | vpl_kl_weight | vpl_gp_temperature |
|--------------|-----------------------|----------------------------------|----------------------|---------------|--------------------|
| 54007/55007 | 1.0 | 0.1 | 5.0 | **0.02** | 1.0 |
| 54008/55008 | 1.0 | 0.1 | 5.0 | **0.05** | 1.0 |
| 54009/55009 | 1.0 | 0.1 | 5.0 | **0.1**  | 1.0 |
| 54010/55010 | 1.0 | 0.1 | 5.0 | **0.2**  | 1.0 |
| 54011/55011 | 1.0 | 0.1 | 5.0 | 0.1 | **0.5** |
| 54012/55012 | 1.0 | 0.1 | 5.0 | 0.1 | **2.0** |
| 54013/55013 | 1.0 | 0.1 | 5.0 | 0.1 | **5.0** |

---

### Phase 3: Refinement (54005/54008/54013 기반)  
**Selector 54014–54022 / RL 55014–55022 (9개)**

| TID (Sel/RL) | vpl_orthogonal_weight | vpl_orthogonal_orthonorm_weight | vpl_prototype_scale | vpl_kl_weight | vpl_gp_temperature |
|--------------|-----------------------|----------------------------------|----------------------|---------------|--------------------|
| 54014/55014 | 1.0 | 0.1 | **1.0**  | 0.1 | 1.0 |
| 54015/55015 | 1.0 | 0.1 | **2.0**  | 0.1 | 1.0 |
| 54016/55016 | 1.0 | 0.1 | **3.0**  | 0.1 | 1.0 |
| 54017/55017 | 1.0 | 0.1 | 2.0 | **0.03** | 1.0 |
| 54018/55018 | 1.0 | 0.1 | 2.0 | **0.05** | 1.0 |
| 54019/55019 | 1.0 | 0.1 | 2.0 | **0.08** | 1.0 |
| 54020/55020 | 1.0 | 0.1 | 2.0 | 0.05 | **3.0** |
| 54021/55021 | 1.0 | 0.1 | 2.0 | 0.05 | **5.0** |
| 54022/55022 | 1.0 | 0.1 | 2.0 | 0.05 | **7.0** |

---

### Phase 4: Learning Rate  
**Selector 54023–54025 / RL 55023–55025 (3개)**

| TID (Sel/RL) | lr |
|--------------|-----|
| 54023/55023 | **0.00005** |
| 54024/55024 | **0.0001** |
| 54025/55025 | **0.0002** |

(그 외 VPL-GP 파라미터는 기본값 유지)

---

### Phase 5: Combined Best Parameters  
**Selector 54026 / RL 55026 (1개)**

- Phase 1–4 기본/최적값 조합 (스크립트 기본값 사용).

---

### Phase 6: Fine-grained Search  
**Selector 54027–54047 / RL 55027–55047 (21개)**

| Sub-phase | TID (Sel/RL) | 탐색 파라미터 | 값 |
|-----------|--------------|----------------|-----|
| **6.1** Orthogonal Weight | 54027–54033 / 55027–55033 | vpl_orthogonal_weight | 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0 |
| **6.2** Orthonorm Weight  | 54034–54039 / 55034–55039 | vpl_orthogonal_orthonorm_weight | 0.0, 0.05, 0.1, 0.2, 0.5, 1.0 |
| **6.3** KL Weight         | 54040–54046 / 55040–55046 | vpl_kl_weight | 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0 |
| **6.4** 최적 조합 검증   | 54047 / 55047 | (기본값) | 1개 |

---

## 3. Qwen 2 (54100 / 55100 대)

Qwen은 **Phase 5개**이며, Phase 2부터는 **54105(prototype_scale=2.0)** 기준을 사용합니다.

### Phase 1: Orthogonal Loss Parameters  
**Selector 54100–54106 / RL 55100–55106 (7개)**

| TID (Sel/RL) | vpl_orthogonal_weight | vpl_orthogonal_orthonorm_weight | vpl_prototype_scale | vpl_kl_weight | vpl_gp_temperature |
|--------------|-----------------------|----------------------------------|----------------------|---------------|--------------------|
| 54100/55100 | **0.2**  | 0.1 | 5.0 | 0.1 | 1.0 |
| 54101/55101 | **1.0**  | 0.1 | 5.0 | 0.1 | 1.0 |
| 54102/55102 | **5.0**  | 0.1 | 5.0 | 0.1 | 1.0 |
| 54103/55103 | 1.0 | **0.0**  | 5.0 | 0.1 | 1.0 |
| 54104/55104 | 1.0 | **0.5**  | 5.0 | 0.1 | 1.0 |
| 54105/55105 | 1.0 | 0.1 | **2.0**  | 0.1 | 1.0 |
| 54106/55106 | 1.0 | 0.1 | **10.0** | 0.1 | 1.0 |

---

### Phase 2: VPL Core (54105 기반, prototype_scale=2.0 고정)  
**Selector 54107–54113 / RL 55107–55113 (7개)**

| TID (Sel/RL) | vpl_prototype_scale | vpl_kl_weight | vpl_gp_temperature |
|--------------|---------------------|---------------|--------------------|
| 54107/55107 | **2.0** | **0.02** | 1.0 |
| 54108/55108 | 2.0 | **0.05** | 1.0 |
| 54109/55109 | 2.0 | **0.1**  | 1.0 |
| 54110/55110 | 2.0 | **0.2**  | 1.0 |
| 54111/55111 | 2.0 | 0.1 | **0.5** |
| 54112/55112 | 2.0 | 0.1 | **2.0** |
| 54113/55113 | 2.0 | 0.1 | **5.0** |

(orthogonal_weight=1.0, orthonorm_weight=0.1 고정)

---

### Phase 3: Learning Rate (Qwen)  
**Selector 54114–54116 / RL 55114–55116 (3개)**

| TID (Sel/RL) | lr |
|--------------|-----|
| 54114/55114 | **0.00005** |
| 54115/55115 | **0.0001** |
| 54116/55116 | **0.0002** |

---

### Phase 4: Combined Best (Qwen)  
**Selector 54117 / RL 55117 (1개)**

- Phase 1–3 기준 최적 조합(스크립트 기본값).

---

### Phase 5: Fine-grained (Qwen)  
**Selector 54118–54138 / RL 55118–55138 (21개)**

| Sub-phase | TID (Sel/RL) | 탐색 파라미터 | 값 |
|-----------|--------------|----------------|-----|
| Orthogonal Weight | 54118–54124 / 55118–55124 | vpl_orthogonal_weight | 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0 |
| Orthonorm Weight  | 54125–54130 / 55125–55130 | vpl_orthogonal_orthonorm_weight | 0.0, 0.05, 0.1, 0.2, 0.5, 1.0 |
| KL Weight         | 54131–54137 / 55131–55137 | vpl_kl_weight | 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0 |
| 최적 조합 검증   | 54138 / 55138 | (기본값) | 1개 |

---

## 4. 실행 요약

| 모델 | Selector 제출 | RL 제출 (Selector 완료 후) |
|------|----------------|-----------------------------|
| Gemma | `bash scripts/hpsearch/submit_selector.sh <1~6>` | `bash scripts/hpsearch/submit_rl.sh <1~6>` |
| Qwen  | `bash scripts/hpsearch/submit_selector_qwen.sh <1~5>` | `bash scripts/hpsearch/submit_rl_qwen.sh <1~5>` |

개별 실험 예:

```bash
# Gemma Selector
sbatch scripts/hpsearch/run_selector_hpsearch.sh 54000

# Gemma RL (Selector 54000 사용)
sbatch scripts/hpsearch/run_rl_hpsearch.sh 55000 54000

# Qwen Selector
sbatch scripts/hpsearch/run_selector_hpsearch_qwen.sh 54100

# Qwen RL
sbatch scripts/hpsearch/run_rl_hpsearch_qwen.sh 55100 54100
```

---

## 5. 기타 TID

| 용도 | TID 범위 | 비고 |
|------|----------|------|
| Local RL only (baseline) | 56000–560XX | Selector 없음, `run_rl_local_only.sh` |

---

## 6. 참고

- **공통 기본값**: `lr=0.0001`, `client_num=10`, `sample_client_num=5`, `vpl_latent_dim=32`.
- **RL 매핑**: RL TID = Selector TID + 1000 (같은 phase 내 동일 오프셋).
- **Config 생성**: `run_selector_hpsearch.sh` / `run_selector_hpsearch_qwen.sh`가 TID에 따라 `cfg/hpsearch/vpl-gp/phase_${TID}.yaml` 또는 `cfg/hpsearch/vpl-gp-qwen/phase_${TID}.yaml` 생성.
