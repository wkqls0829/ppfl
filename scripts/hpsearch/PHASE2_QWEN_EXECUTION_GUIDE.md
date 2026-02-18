# Qwen Phase 2 실행 가이드 (54105 기반)

Phase 1에서 **54105**(prototype_scale=2.0)를 기준으로 Phase 2를 실행합니다.  
Phase 2 실험(54107-54113)은 모두 **vpl_prototype_scale=2.0**으로 고정하고, kl_weight / gp_temperature만 탐색합니다.

## 기준 설정

- **54105 기반**: `vpl_prototype_scale` = **2.0**
- `vpl_orthogonal_weight`: 1.0  
- `vpl_orthogonal_orthonorm_weight`: 0.1  
- `lr`: 0.0001  

## Phase 2 TID별 설정 (54107-54113)

| TID (Sel/RL) | vpl_prototype_scale | vpl_kl_weight | vpl_gp_temperature |
|--------------|----------------------|---------------|--------------------|
| 54107/55107 | 2.0 | **0.02** | 1.0 |
| 54108/55108 | 2.0 | **0.05** | 1.0 |
| 54109/55109 | 2.0 | **0.1** | 1.0 |
| 54110/55110 | 2.0 | **0.2** | 1.0 |
| 54111/55111 | 2.0 | 0.1 | **0.5** |
| 54112/55112 | 2.0 | 0.1 | **2.0** |
| 54113/55113 | 2.0 | 0.1 | **5.0** |

## 실행 방법

### 1. Phase 1 완료 확인 (특히 54105)

```bash
ls -lh checkpoints/*vplgp_ortho_t54105*.ckpt
```

### 2. Phase 2 Selector 제출

```bash
bash scripts/hpsearch/submit_selector_qwen.sh 2
```

- Selector TID: 54107-54113 (7개)

### 3. Selector 완료 후 RL 제출

```bash
bash scripts/hpsearch/submit_rl_qwen.sh 2
```

- RL TID: 55107-55113 (각각 Selector 54107-54113 사용)

## 참고

- 스크립트: `run_selector_hpsearch_qwen.sh`에서 Phase 2(54107-54113) 구간에 `PROTOTYPE_SCALE=2.0` 적용됨.
- Gemma Phase 2는 Phase 1 최적(54001 등)을 쓰며, Qwen만 54105(prototype_scale=2.0)를 명시적으로 Phase 2 기준으로 사용합니다.
