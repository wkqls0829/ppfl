# Hyperparameter Search Scripts

SLURM 클러스터에서 VPL-GP Hyperparameter Search 실험을 실행하기 위한 스크립트입니다.

**Phase·TID·파라미터 전체 정리**: [docs/HYPERPARAMETER_SEARCH.md](../../docs/HYPERPARAMETER_SEARCH.md)

## 실험 구조

- **Gemma**: Selector 54000–54047, RL 55000–55047 (Phase 1–6)
- **Qwen**: Selector 54100–54138, RL 55100–55138 (Phase 1–5)

각 Phase별 TID 범위와 파라미터 표는 위 문서 참고.

## 파일 구조

```
scripts/slurm/hpsearch/
├── README.md
├── run_selector_hpsearch.sh      # Gemma Selector (sbatch)
├── run_rl_hpsearch.sh           # Gemma RL (sbatch)
├── submit_selector.sh           # Gemma Selector 제출 (phase 1–6)
├── submit_rl.sh                 # Gemma RL 제출 (phase 1–6)
├── run_selector_hpsearch_qwen.sh
├── run_rl_hpsearch_qwen.sh
├── submit_selector_qwen.sh      # Qwen (phase 1–5)
├── submit_rl_qwen.sh
└── run_rl_local_only.sh         # Local RL only (baseline, TID 56000)
```

## 사용 방법

**주의**: `submit_*.sh`는 **bash**로 실행. `sbatch`로 실행하면 안 됩니다.

### Selector → RL 순서

```bash
# Gemma (예: Phase 5 Combined Best)
bash scripts/slurm/hpsearch/submit_selector.sh 5
# 완료 확인 후
bash scripts/slurm/hpsearch/submit_rl.sh 5

# Qwen (예: Phase 4 Combined Best)
bash scripts/slurm/hpsearch/submit_selector_qwen.sh 4
bash scripts/slurm/hpsearch/submit_rl_qwen.sh 4
```

### 개별 실험 (sbatch)

```bash
sbatch scripts/slurm/hpsearch/run_selector_hpsearch.sh 54026
sbatch scripts/slurm/hpsearch/run_rl_hpsearch.sh 55026 54026
```

### Local RL Only (Baseline)

Selector 없이 RL만 실행. `rlhf_use_variational_selection: False`.

```bash
bash scripts/slurm/hpsearch/run_rl_local_only.sh 56000
```

## 모니터링

```bash
squeue -u $USER
tail -f outputs/54026.log
```

체크포인트·WandB·진행 상황: [docs/HYPERPARAMETER_SEARCH.md](../../docs/HYPERPARAMETER_SEARCH.md) §5–6 참고.

## 주의사항

1. RL 실행 전 해당 Selector checkpoint 완료 필요.
2. `submit_*.sh`의 `WORK_DIR`는 `/home2/jbkoo/ppfl`. 다른 경로는 스크립트 수정 또는 sbatch 직접 사용.
3. Phase별 파라미터는 `run_selector_hpsearch.sh` / `run_selector_hpsearch_qwen.sh` 내 TID별로 설정됨. 전체 표는 docs 참고.
