# Hyperparameter Search Scripts

이 디렉토리는 다양한 알고리즘의 hyperparameter search를 위한 스크립트와 설정 파일을 포함합니다.

## 디렉토리 구조

```
scripts/hpsearch/
├── README.md (이 파일)
└── vpl-gp/          # VPL-GP 알고리즘 hyperparameter search
    ├── run_experiment.sh        # Generic experiment runner script
    ├── phase1_orthogonal.sh     # Phase 1: Orthogonal loss weight search (52000-52006)
    ├── phase2_vpl_core.sh       # Phase 2: VPL core parameters search (52007-52013)
    ├── phase3_lr.sh             # Phase 3: Learning rate search (52014-52016)
    └── phase4_combined.sh       # Phase 4: Best parameters combination (52017)

cfg/hpsearch/
└── vpl-gp/          # VPL-GP hyperparameter search configs
    ├── phase1_orthogonal_52000.yaml
    ├── phase1_orthogonal_52001.yaml
    ├── phase1_orthogonal_52002.yaml
    ├── phase1_orthogonal_52003.yaml
    ├── phase1_orthogonal_52004.yaml
    ├── phase1_orthogonal_52005.yaml
    ├── phase1_orthogonal_52006.yaml
    ├── phase2_vpl_core_52007.yaml
    ├── phase2_vpl_core_52008.yaml
    ├── phase2_vpl_core_52009.yaml
    ├── phase2_vpl_core_52010.yaml
    ├── phase2_vpl_core_52011.yaml
    ├── phase2_vpl_core_52012.yaml
    ├── phase2_vpl_core_52013.yaml
    ├── phase3_lr_52014.yaml
    ├── phase3_lr_52015.yaml
    ├── phase3_lr_52016.yaml
    └── phase4_combined_52017.yaml
```

## 사용 방법

### 1. Phase별 일괄 실행

각 phase의 스크립트를 실행하여 해당 phase의 모든 실험을 순차적으로 실행:

```bash
# Phase 1: Orthogonal loss weight search (7 experiments: 52000-52006)
bash scripts/hpsearch/vpl-gp/phase1_orthogonal.sh

# Phase 2: VPL core parameters search (7 experiments: 52007-52013)
bash scripts/hpsearch/vpl-gp/phase2_vpl_core.sh

# Phase 3: Learning rate search (3 experiments: 52014-52016)
bash scripts/hpsearch/vpl-gp/phase3_lr.sh

# Phase 4: Best parameters combination (1 experiment: 52017)
# NOTE: Update phase4_combined_52017.yaml with best parameters first!
bash scripts/hpsearch/vpl-gp/phase4_combined.sh
```

### 2. 개별 실험 실행

특정 실험만 실행하려면 `run_experiment.sh` 스크립트를 사용:

```bash
# Example: Run experiment 52000
bash scripts/hpsearch/vpl-gp/run_experiment.sh 52000 cfg/hpsearch/vpl-gp/phase1_orthogonal_52000.yaml
```

### 3. 병렬 실행

여러 GPU가 있다면 각 실험을 다른 GPU에서 병렬로 실행할 수 있습니다. 
각 config 파일의 `device` 설정을 변경하여 GPU를 할당하세요.

## 실험 ID 규칙

- **50024**: Baseline (원본 설정)
- **52000-52006**: Phase 1 (Orthogonal loss weight search)
- **52007-52013**: Phase 2 (VPL core parameters search)
- **52014-52016**: Phase 3 (Learning rate search)
- **52017**: Phase 4 (Best parameters combination)

## 결과 확인

### WandB
모든 실험은 `fvpl-selector` 프로젝트에서 추적됩니다:
- https://wandb.ai/jabinteam/fvpl-selector

### 로그 파일
각 실험의 로그는 `outputs/{experiment_id}.log`에 저장됩니다.

### Checkpoint
각 실험의 checkpoint는 다음 경로에 저장됩니다:
```
/hdd/hdd3/kjb/checkpoints/hhrl_choice_gemma_fedbiscuit_u3_vplgp_ortho_{experiment_id}.ckpt
```

## 평가 지표

각 실험에서 다음 지표를 모니터링하세요:

1. **Accuracy**: 선택 정확도 (가장 중요)
2. **Loss**: 전체 loss (train_avg_loss)
3. **VPL Losses**:
   - `vpl_kl_loss`: KL divergence loss
   - `vpl_reconstruction_loss`: Reconstruction loss
   - `vpl_orthogonal_loss`: Orthogonal loss
4. **t-SNE Visualization**: z 분포의 시각적 품질
5. **Client Separation**: Harmlessness/Helpfulness client 간 분리 정도

## 주의사항

1. **GPU 할당**: 각 실험은 독립적인 GPU에서 실행 (병렬 실행 가능)
2. **Checkpoint 저장**: 각 실험의 checkpoint는 나중에 RL training에 사용
3. **WandB Tracking**: 모든 실험을 `fvpl-selector` 프로젝트에서 추적
4. **Baseline 비교**: 50024를 baseline으로 모든 실험과 비교

## 상세 계획

자세한 hyperparameter search 계획은 다음 문서를 참고하세요:
- `docs/HPSEARCH_50024_PLAN.md`
