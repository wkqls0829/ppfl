# Ablation Study Scripts (로컬 서버 실행용)

로컬 서버에서 Ablation Study를 실행하기 위한 스크립트입니다. vpl-gp의 hhst와 hrl 스크립트 구조를 참고하여 작성되었습니다.

## 실험 구성

### 1. VPL + GP (Gumbel-Softmax Prior만, Orthogonal Loss 없음)
- **Selector**: `hhst_vplgp.sh` (TID: 62300)
- **RL**: `hrl_vplgp.sh` (TID: 63300)
- **Config**: `cfg/vpl-gp-no-ortho/hhst.yaml`, `cfg/vpl-gp-no-ortho/hrl.yaml`

### 2. VPL + Ortho (Orthogonal Loss만, GP Prior 없음)
- **Selector**: `hhst_vplortho.sh` (TID: 62310)
- **RL**: `hrl_vplortho.sh` (TID: 63310)
- **Config**: `cfg/vpl-ortho/hhst.yaml`, `cfg/vpl-ortho/hrl.yaml`

## 하이퍼파라미터 설정

### VPL + GP (ablation script 기반)
- `vpl_use_gp_prior: True`
- `vpl_kl_weight: 0.02` (FedVPA-GP와 동일)
- `vpl_gp_temperature: 1.0`
- `vpl_orthogonal_weight: 0.0` (orthogonal loss 비활성화)
- `vpl_orthogonal_orthonorm_weight: 0.0`
- `vpl_latent_dim: 32`
- `vpl_feature_method: 'choice_logits'`
- `vpl_use_feature_difference: True`
- `vpl_use_difference_only: True`
- `vpl_max_logvar: -3.0`

### VPL + Ortho (ablation script 기반)
- `vpl_use_gp_prior: False` (standard normal prior)
- `vpl_kl_weight: 0.1` (VPL baseline과 동일)
- `vpl_orthogonal_weight: 1.0` (orthogonal loss 활성화)
- `vpl_orthogonal_orthonorm_weight: 0.0` (FedVPA-GP와 동일)
- `vpl_use_manual_orthogonal_labels: True`
- `vpl_num_prototypes: 2`
- `vpl_prototype_scale: 5.0`
- `vpl_latent_dim: 32`
- `vpl_feature_method: 'choice_logits'`
- `vpl_use_feature_difference: True`
- `vpl_use_difference_only: True`
- `vpl_max_logvar: -3.0`

## 실행 방법

### 1. Selector 학습

#### VPL + GP
```bash
cd /home/kjb/ppfl
bash scripts/ablation/hhst_vplgp.sh
```

#### VPL + Ortho
```bash
cd /home/kjb/ppfl
bash scripts/ablation/hhst_vplortho.sh
```

### 2. RL 학습 (Selector 완료 후)

Selector 학습이 완료된 후:

#### VPL + GP
```bash
bash scripts/ablation/hrl_vplgp.sh
```

#### VPL + Ortho
```bash
bash scripts/ablation/hrl_vplortho.sh
```

## 설정 변경

### GPU Device 번호 변경
스크립트 상단의 `device` 변수를 수정:
```bash
device=0  # 원하는 GPU 번호로 변경
```

### TID 변경
스크립트 상단의 `tid` 변수를 수정:
```bash
tid=62300  # 원하는 TID로 변경
```

## 출력 파일

### Checkpoints
- **Selector**: `checkpoints/hhrl_choice_gemma_ablation_{method}_t{TID}.ckpt`
- **RL**: `checkpoints/hhrl_rlhf_gemma_ablation_{method}_t{TID}.ckpt`

### Log Files
- **위치**: `outputs/`
- **파일명**: `{TID}.log`

## 모니터링

### 실시간 로그 확인
```bash
tail -f outputs/62300.log  # VPL + GP Selector
tail -f outputs/63300.log  # VPL + GP RL
tail -f outputs/62310.log  # VPL + Ortho Selector
tail -f outputs/63310.log  # VPL + Ortho RL
```

### 프로세스 확인
```bash
ps aux | grep "federatedscope/main.py" | grep ablation
```

### GPU 사용량 확인
```bash
watch -n 1 nvidia-smi
```

## 주의사항

1. **Selector 완료 후 RL 실행**: RL 실험은 해당 Selector checkpoint가 필요합니다.
2. **환경 변수**: `.env` 파일에 `HF_TOKEN`과 `OPENAI_API_KEY`가 설정되어 있어야 합니다.
3. **데이터 경로**: 데이터가 `$WORK_DIR/data/`에 있어야 합니다.
4. **체크포인트 경로**: 로컬 repo의 `checkpoints/` 디렉토리에 저장됩니다.

## 비교 분석

이 ablation study로 다음을 확인할 수 있습니다:

1. **GP Prior의 효과**: 
   - FedVPL vs VPL + GP
   - VPL + Ortho vs FedVPA-GP

2. **Orthogonal Loss의 효과**:
   - FedVPL vs VPL + Ortho
   - VPL + GP vs FedVPA-GP

3. **조합 효과**:
   - FedVPL vs FedVPA-GP
