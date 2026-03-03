# HP Search 최적값: Gemma Phase 4 / Qwen Phase 3

**기준**: `outputs_cluster` RL 로그 Round 49 결과 (helpfulness_winrate, harmlessness_winrate).  
**목적**: Gemma **Phase 4 (Learning Rate)**·Qwen **Phase 3 (Learning Rate)** 탐색 시 고정할 최적 파라미터 정리.

---

## 1. Gemma-2B RL Round 49 결과 요약

| TID | Phase | 주요 탐색 파라미터 | Help WR | Harm WR | 합 |
|-----|--------|---------------------|---------|---------|-----|
| 55000 | 1 | orth_weight=0.2 | 50.0 | 93.33 | 143.33 |
| 55001 | 1 | orth_weight=1.0 | 50.0 | 86.67 | 136.67 |
| 55002 | 1 | orth_weight=5.0 | 53.33 | 93.33 | 146.66 |
| 55003 | 1 | orthonorm=0.0 | 56.67 | 90.0 | 146.67 |
| 55004 | 1 | orthonorm=0.5 | 50.0 | **96.67** | 146.67 |
| 55005 | 1 | prototype_scale=2.0 | 60.0 | 90.0 | 150.0 |
| 55006 | 1 | prototype_scale=10.0 | 56.67 | 90.0 | 146.67 |
| 55007 | 2 | kl_weight=0.02 | 53.33 | 93.33 | 146.66 |
| 55008 | 2 | kl_weight=0.05 | 60.0 | **96.67** | **156.67** |
| 55009 | 2 | kl_weight=0.1 | 50.0 | 93.33 | 143.33 |
| 55010 | 2 | kl_weight=0.2 | 53.33 | 86.67 | 140.0 |
| 55011 | 2 | gp_temperature=0.5 | 53.33 | 93.33 | 146.66 |
| 55012 | 2 | gp_temperature=2.0 | 53.33 | 90.0 | 143.33 |
| 55013 | 2 | gp_temperature=5.0 | 60.0 | 93.33 | 153.33 |
| 55014 | 3 | prototype_scale=1.0 | 56.67 | 90.0 | 146.67 |
| 55015 | 3 | prototype_scale=2.0 | 56.67 | 86.67 | 143.34 |
| 55016 | 3 | prototype_scale=3.0 | 56.67 | 90.0 | 146.67 |
| **55017** | **3** | **kl_weight=0.03** | **63.33** | **96.67** | **160.0** |
| 55018 | 3 | kl_weight=0.05 | 56.67 | 96.67 | 153.34 |
| 55019 | 3 | kl_weight=0.08 | 56.67 | 96.67 | 153.34 |
| 55020 | 3 | gp_temperature=3.0 | 50.0 | 83.33 | 133.33 |
| 55021 | 3 | gp_temperature=5.0 | 56.67 | 96.67 | 153.34 |
| 55022 | 3 | gp_temperature=7.0 | 56.67 | 93.33 | 150.0 |

**Gemma 최고 성능**: **55017** (Help 63.33, Harm 96.67) — Phase 3, `kl_weight=0.03`, `prototype_scale=2.0`, `gp_temperature=1.0`.

---

## 2. Qwen 2 RL Round 49 결과 요약

| TID | Phase | 주요 탐색 파라미터 | Help WR | Harm WR | 합 |
|-----|--------|---------------------|---------|---------|-----|
| 55100 | 1 | orth_weight=0.2 | 26.67 | 86.67 | 113.34 |
| 55101 | 1 | orth_weight=1.0 | 26.67 | 86.67 | 113.34 |
| 55102 | 1 | orth_weight=5.0 | 16.67 | 86.67 | 103.34 |
| 55103 | 1 | orthonorm=0.0 | 33.33 | 80.0 | 113.33 |
| 55104 | 1 | orthonorm=0.5 | 33.33 | 80.0 | 113.33 |
| 55105 | 1 | prototype_scale=2.0 | 33.33 | 80.0 | 113.33 |
| 55106 | 1 | prototype_scale=10.0 | 30.0 | 73.33 | 103.33 |
| 55107 | 2 | kl_weight=0.02 | 30.0 | 86.67 | 116.67 |
| 55108 | 2 | kl_weight=0.05 | 23.33 | 86.67 | 110.0 |
| 55109 | 2 | kl_weight=0.1 | 30.0 | 90.0 | 120.0 |
| 55110 | 2 | kl_weight=0.2 | 33.33 | 80.0 | 113.33 |
| 55111 | 2 | gp_temperature=0.5 | 33.33 | 90.0 | 123.33 |
| 55112 | 2 | gp_temperature=2.0 | 30.0 | 86.67 | 116.67 |
| **55113** | **2** | **gp_temperature=5.0** | **40.0** | **83.33** | **123.33** |

**Qwen 최고 성능**: **55113** (Help 40.0, Harm 83.33) — Phase 2, `prototype_scale=2.0`, `kl_weight=0.1`, `gp_temperature=5.0`.  
(55111도 Help+Harm 합 동일하나, Helpfulness 단일 지표로는 55113이 더 높음.)

---

## 3. Gemma Phase 4 (Learning Rate)용 최적 고정값

Phase 4는 **Learning Rate만** 탐색하고 나머지는 고정한다.  
고정값은 **55017(Selector 54017)** 기준으로 둔다.

| 파라미터 | Phase 4 권장값 | 비고 |
|----------|----------------|------|
| vpl_orthogonal_weight | 1.0 | Phase 1 기본 |
| vpl_orthogonal_orthonorm_weight | 0.1 | Phase 1 기본 |
| vpl_prototype_scale | **2.0** | 55017(Phase 3) |
| vpl_kl_weight | **0.03** | 55017(Phase 3) |
| vpl_gp_temperature | 1.0 | 55017(Phase 3) |
| **lr** | **탐색** | 54023: 0.00005, 54024: 0.0001, 54025: 0.0002 |

**스크립트 반영**: `run_selector_hpsearch.sh` Phase 4(54023–54025) 블록에서  
기본값을 `PROTOTYPE_SCALE=2.0`, `KL_WEIGHT=0.03`, `GP_TEMPERATURE=1.0` 로 두고 LR만 54023/54024/54025에 맞게 넣으면 된다.

---

## 4. Qwen Phase 3 (Learning Rate)용 최적 고정값

Phase 3는 **Learning Rate만** 탐색한다.  
고정값은 **55113(Selector 54113)** 기준으로 둔다.

| 파라미터 | Phase 3 권장값 | 비고 |
|----------|----------------|------|
| vpl_orthogonal_weight | 1.0 | Phase 1 기본 |
| vpl_orthogonal_orthonorm_weight | 0.1 | Phase 1 기본 |
| vpl_prototype_scale | **2.0** | Phase 2 기준 |
| vpl_kl_weight | 0.1 | 55113(Phase 2) |
| vpl_gp_temperature | **5.0** | 55113(Phase 2) |
| **lr** | **탐색** | 54114: 0.00005, 54115: 0.0001, 54116: 0.0002 |

**스크립트 반영**: `run_selector_hpsearch_qwen.sh` Phase 3(54114–54116) 구간에서  
`PROTOTYPE_SCALE=2.0`, `KL_WEIGHT=0.1`, `GP_TEMPERATURE=5.0` 으로 고정하고 LR만 54114/54115/54116에 맞게 설정하면 된다.

---

## 5. 요약

- **Gemma Phase 4 (LR 탐색)**  
  - 최적 조합: **55017** (Phase 3, kl=0.03, prototype_scale=2.0).  
  - Phase 4에서는 위 표대로 고정하고 **lr만** 0.00005 / 0.0001 / 0.0002 로 돌리면 됨.

- **Qwen Phase 3 (LR 탐색)**  
  - 최적 조합: **55113** (Phase 2, gp_temperature=5.0, prototype_scale=2.0, kl=0.1).  
  - Phase 3에서는 위 표대로 고정하고 **lr만** 0.00005 / 0.0001 / 0.0002 로 돌리면 됨.

이렇게 하면 Gemma Phase 4와 Qwen Phase 3 모두 “지금까지 최고 RL 성능 나온 설정 + LR 그리드”로 일관되게 돌릴 수 있다.
