# Hyperparameter Search 진행 상황 보고서

**기준**: `outputs_cluster/` 로그 파일 (클러스터 실험 결과)  
**확인 일시**: 2026-02-25 기준

---

## 1. 요약

| 구분 | Gemma-2B Selector | Gemma-2B RL | Qwen 2 Selector | Qwen 2 RL |
|------|-------------------|-------------|-----------------|-----------|
| **Phase 1** | ⚠️ 미완료 (중단) | ✅ 완료 | ✅ 완료 | ✅ 완료 |
| **Phase 2** | ✅ 완료 | ✅ 완료 | ✅ 완료 | ✅ 완료 |
| **Phase 3** | ✅ 완료 | ✅ 완료 | ❌ 미실행 | ❌ 미실행 |
| **Phase 4** | ❌ 미실행 | ❌ 미실행 | ❌ 미실행 | ❌ 미실행 |
| **Phase 5** | ❌ 미실행 | ❌ 미실행 | ❌ 미실행 | ❌ 미실행 |
| **Phase 6 (Gemma)** | ❌ 미실행 | ❌ 미실행 | — | — |

---

## 2. Gemma-2B 상세

### Phase 1: Orthogonal Loss (TID 54000–54006 / 55000–55006)

- **Selector (54000–54006)**  
  - 로그 파일 있음. **50 round 미완료**, 중간에 중단된 상태.  
  - 54000: round 10 수준, 54001: round 10, 54002: round 12, 54003–54006: round 2–3 수준에서 종료.  
  - **조치**: 재실행 필요 (`bash scripts/hpsearch/submit_selector.sh 1`).

- **RL (55000–55006)**  
  - 7개 로그 모두 **Round 49**까지 기록됨 → **정상 완료**.  
  - (RL은 round 0~49로 50 rounds 구성.)

### Phase 2: VPL Core (TID 54007–54013 / 55007–55013)

- **Selector**: 54007–54013 로그 모두 **round 50** 및 "Final evaluation" 로그 있음 → **완료**.
- **RL**: 55007–55013 로그 모두 **Round 49**까지 기록 → **완료**.

### Phase 3: Refinement (TID 54014–54022 / 55014–55022)

- **Selector**: 54014–54022 로그 모두 **round 50** 및 "Final evaluation" 로그 있음 → **완료**.
- **RL**: 55014–55022 로그 모두 **Round 49**까지 기록 → **완료**.

### Phase 4–6 (TID 54023–54047 / 55023–55047)

- **로그 파일 없음** → **미실행**.  
- 실행 시: `submit_selector.sh 4` → 완료 후 `submit_rl.sh 4`, 동일하게 Phase 5, 6 순서로 진행.

---

## 3. Qwen 2 상세

### Phase 1: Orthogonal Loss (TID 54100–54106 / 55100–55106)

- **Selector**: 54100–54106 로그에 **round 50** 브로드캐스트 로그 있음 → **완료**.
- **RL**: 55100–55106 로그에 **Round 49** 로그 있음 → **완료**.

### Phase 2: VPL Core (TID 54107–54113 / 55107–55113)

- **Selector**: 54107–54113 로그에 **round 50** 및 "Final evaluation" 로그 있음 → **완료**.
- **RL**: 55107–55113 로그에 **Round 49** 로그 있음 → **완료**.

### Phase 3–5 (TID 54114–54138 / 55114–55138)

- **로그 파일 없음** → **미실행**.  
- Qwen은 Phase 5개만 정의됨 (README 기준).  
- 실행 시: `submit_selector_qwen.sh 3`, `submit_rl_qwen.sh 3` 등으로 진행.

---

## 4. 로그 파일 현황 (outputs_cluster)

### Gemma Selector (54xxx)

- **있음**: 54000–54022 (23개)
- **없음**: 54023–54047 (Phase 4–6)

### Gemma RL (55xxx)

- **있음**: 55000–55022 (23개)
- **없음**: 55023–55047

### Qwen Selector (541xx)

- **있음**: 54100–54113 (14개)
- **없음**: 54114–54138

### Qwen RL (551xx)

- **있음**: 55100–55113 (14개)
- **없음**: 55114–55138

---

## 5. 권장 조치

1. **Gemma Phase 1 Selector 재실행**  
   - 54000–54006이 50 round까지 완료되지 않았으므로, 재제출 권장.  
   - `bash scripts/hpsearch/submit_selector.sh 1`  
   - 완료 후 필요 시 RL은 이미 완료된 55000–55006 사용 가능 (동일 TID 재실행 여부는 정책에 따름).

2. **Gemma Phase 4–6 실행**  
   - Phase 4 (Learning Rate): `submit_selector.sh 4` → `submit_rl.sh 4`  
   - Phase 5 (Combined Best): `submit_selector.sh 5` → `submit_rl.sh 5`  
   - Phase 6 (Fine-grained): `submit_selector.sh 6` → `submit_rl.sh 6`

3. **Qwen Phase 3–5 실행**  
   - Phase 3 (LR): `submit_selector_qwen.sh 3`, `submit_rl_qwen.sh 3`  
   - Phase 4 (Combined Best): `submit_selector_qwen.sh 4`, `submit_rl_qwen.sh 4`  
   - Phase 5 (Fine-grained): `submit_selector_qwen.sh 5`, `submit_rl_qwen.sh 5`

---

## 6. 참고

- Phase·TID 매핑: `docs/HYPERPARAMETER_SEARCH_PHASES_AND_TIDS.md`
- 실행 스크립트: `scripts/hpsearch/README.md`
