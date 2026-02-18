# Phase 3 실행 가이드 (Gemma)

Phase 2 결과에서 **54005**(prototype_scale=2.0), **54008**(kl_weight=0.05), **54013**(gp_temperature=5.0) 수치가 다소 높게 나온 것을 기준으로, 해당 구간을 세밀하게 탐색하는 Phase 3를 실행합니다.

## Phase 3 설정 요약

### 기준 설정 (Phase 1·2)
- `vpl_orthogonal_weight`: **1.0**
- `vpl_orthogonal_orthonorm_weight`: **0.1**
- `lr`: **0.0001**

### Phase 3 탐색 파라미터 (9개 실험)

| 구간 | TID | 탐색 파라미터 | 기준 TID |
|------|-----|----------------|----------|
| 3a | 54014-54016 | vpl_prototype_scale | 54005 (2.0) |
| 3b | 54017-54019 | vpl_kl_weight | 54008 (0.05) |
| 3c | 54020-54022 | vpl_gp_temperature | 54013 (5.0) |

## TID별 하이퍼파라미터 (Phase 3)

| TID (Sel/RL) | vpl_prototype_scale | vpl_kl_weight | vpl_gp_temperature | 비고 |
|--------------|----------------------|---------------|---------------------|------|
| 54014/55014 | **1.0** | 0.1 | 1.0 | prototype_scale 세밀 탐색 |
| 54015/55015 | **2.0** | 0.1 | 1.0 | |
| 54016/55016 | **3.0** | 0.1 | 1.0 | |
| 54017/55017 | 2.0 | **0.03** | 1.0 | kl_weight 세밀 탐색 (54008 주변) |
| 54018/55018 | 2.0 | **0.05** | 1.0 | |
| 54019/55019 | 2.0 | **0.08** | 1.0 | |
| 54020/55020 | 2.0 | 0.05 | **3.0** | gp_temperature 세밀 탐색 (54013 주변) |
| 54021/55021 | 2.0 | 0.05 | **5.0** | |
| 54022/55022 | 2.0 | 0.05 | **7.0** | |

- 3a: Phase 2 기본(kl=0.1, temp=1.0) + prototype_scale만 1.0, 2.0, 3.0
- 3b: prototype_scale=2.0(54005) 고정, kl_weight만 0.03, 0.05, 0.08
- 3c: prototype_scale=2.0, kl=0.05(54008) 고정, gp_temperature만 3.0, 5.0, 7.0

## 실행 방법

### 1. Selector Training

```bash
# Phase 3 Selector (9개 제출)
bash scripts/hpsearch/submit_selector.sh 3
```

**실행 범위**: Selector TID 54014-54022

### 2. Selector 완료 확인

```bash
ls -lh checkpoints/*vplgp_ortho_t5401[4-9]*.ckpt
ls -lh checkpoints/*vplgp_ortho_t5402[0-2]*.ckpt
```

### 3. RL Training

Selector checkpoint가 준비된 뒤:

```bash
# Phase 3 RL (9개 제출)
bash scripts/hpsearch/submit_rl.sh 3
```

**실행 범위**: RL TID 55014-55022, 각각 Selector 54014-54022 사용

## Phase 번호 변경 사항 (Gemma)

Phase 3 추가로 기존 Phase가 한 칸씩 밀렸습니다.

| Phase | Selector TID | RL TID | 내용 |
|-------|--------------|--------|------|
| 1 | 54000-54006 | 55000-55006 | Orthogonal Loss |
| 2 | 54007-54013 | 55007-55013 | VPL Core |
| **3** | **54014-54022** | **55014-55022** | **Refinement (54005/54008/54013 기반)** |
| 4 | 54023-54025 | 55023-55025 | Learning Rate |
| 5 | 54026 | 55026 | Combined Best |
| 6 | 54027-54047 | 55027-55047 | Fine-grained |

## 모니터링

```bash
squeue -u $USER
tail -f outputs/54014.log
# WandB: fvpl-selector, fvpl-rl
```

## 다음 단계

Phase 3 결과에서 최적 조합을 정한 뒤 Phase 4 (Learning Rate) 또는 Phase 5 (Combined Best)를 진행합니다.
