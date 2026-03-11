# Main Table Experiment Scripts

SLURM 클러스터에서 Main Table 실험을 실행하기 위한 스크립트입니다.

## 실험 구조

- **2 models**: Qwen 2, Gemma-2B
- **Selector methods**: FedBiscuit, FedVPL, FedVPA-GP (3개)
- **RL methods**: FedDPO, FedBiscuit, FedVPL, FedVPA-GP (4개)
- **3 client counts**: 10, 50, 100
- **총 18 selector experiments + 24 RL experiments = 42개**
- **Note**: FedDPO는 RL에서만 사용 (selector training 불필요)

## TID 번호 체계

### Gemma-2B
- **Selector**: 62110-62132 (FedDPO 제외)
  - FedBiscuit: 62110-62112 (N=10,50,100)
  - FedVPL: 62120-62122 (N=10,50,100)
  - FedVPA-GP: 62130-62132 (N=10,50,100)
- **RL**: 63100-63132
  - FedDPO: 63100-63102 (N=10,50,100) - selector checkpoint 불필요
  - FedBiscuit: 63110-63112 (N=10,50,100)
  - FedVPL: 63120-63122 (N=10,50,100)
  - FedVPA-GP: 63130-63132 (N=10,50,100)

### Qwen 2
- **Selector**: 62210-62232 (FedDPO 제외)
  - FedBiscuit: 62210-62212 (N=10,50,100)
  - FedVPL: 62220-62222 (N=10,50,100)
  - FedVPA-GP: 62230-62232 (N=10,50,100)
- **RL**: 63200-63232
  - FedDPO: 63200-63202 (N=10,50,100) - selector checkpoint 불필요
  - FedBiscuit: 63210-63212 (N=10,50,100)
  - FedVPL: 63220-63222 (N=10,50,100)
  - FedVPA-GP: 63230-63232 (N=10,50,100)

## 파일 구조

```
scripts/slurm/main_table/
├── README.md (이 파일)
├── run_selector_gemma.sh      # Gemma-2B selector 실행 스크립트
├── run_rl_gemma.sh            # Gemma-2B RL 실행 스크립트
├── submit_all_gemma.sh         # 모든 Gemma-2B selector 작업 제출
├── submit_rl_gemma.sh          # 모든 Gemma-2B RL 작업 제출
├── run_selector_qwen.sh        # Qwen 2 selector 실행 스크립트
├── run_rl_qwen.sh              # Qwen 2 RL 실행 스크립트
├── submit_all_qwen.sh          # 모든 Qwen 2 selector 작업 제출
└── submit_rl_qwen.sh           # 모든 Qwen 2 RL 작업 제출
```

## 사용 방법

### 1. Selector 실험 실행

#### Gemma-2B
```bash
# 개별 실험 실행
sbatch scripts/slurm/main_table/run_selector_gemma.sh fedvpagp 10 62130

# 모든 selector 실험 제출
bash scripts/slurm/main_table/submit_all_gemma.sh
```

#### Qwen 2
```bash
# 개별 실험 실행
sbatch scripts/slurm/main_table/run_selector_qwen.sh fedvpagp 10 62230

# 모든 selector 실험 제출
bash scripts/slurm/main_table/submit_all_qwen.sh
```

### 2. RL 실험 실행

Selector 실험이 완료된 후:

#### Gemma-2B
```bash
# 개별 RL 실험 실행
sbatch scripts/slurm/main_table/run_rl_gemma.sh fedvpagp 10 63130 62130

# 모든 RL 실험 제출
bash scripts/slurm/main_table/submit_rl_gemma.sh
```

#### Qwen 2
```bash
# 개별 RL 실험 실행
sbatch scripts/slurm/main_table/run_rl_qwen.sh fedvpagp 10 63230 62230

# 모든 RL 실험 제출
bash scripts/slurm/main_table/submit_rl_qwen.sh
```

## 설정 변경사항

### 체크포인트 경로
- 기존: `/hdd/hdd3/kjb/checkpoints/`
- 변경: `$WORK_DIR/checkpoints/` (로컬 repo)

### 데이터 경로
- 기존: `/hdd/hdd3/kjb/`
- 변경: `$WORK_DIR/data/` (로컬 repo)

### 클라이언트 수별 설정
- **N=10**: `client_num=10`, `sample_client_num=5`
- **N=50**: `client_num=50`, `sample_client_num=10`
- **N=100**: `client_num=100`, `sample_client_num=10` (모든 경우 라운드당 10 클라이언트)

## 하이퍼파라미터

### FedVPA-GP (하이퍼파라미터 서치 결과 기반)
- `vpl_orthogonal_weight: 1.0`
- `vpl_prototype_scale: 5.0`
- `vpl_kl_weight: 0.02` (하이퍼파라미터 서치 결과)
- `vpl_orthogonal_orthonorm_weight: 0.0` (하이퍼파라미터 서치 결과)
- `vpl_gp_temperature: 1.0`

### 모델별 Learning Rate
- **Gemma-2B**: `lr: 0.0001`
- **Qwen 2**: `lr: 0.00001`

## 모니터링

### SLURM 작업 상태 확인
```bash
squeue -u $USER
```

### 로그 확인
```bash
# 실시간 로그 확인
tail -f outputs/62130.log

# SLURM 출력 확인
tail -f /home2/jbkoo/slurm/logs/slurm-*.out
```

### 체크포인트 확인
```bash
ls -lh checkpoints/*62130*.ckpt
```

## 주의사항

1. **Selector 완료 후 RL 실행**: RL 실험은 해당 Selector checkpoint가 필요합니다.
2. **체크포인트 경로**: 로컬 repo의 `checkpoints/` 디렉토리에 저장됩니다.
3. **데이터 경로**: 데이터가 `$WORK_DIR/data/`에 있어야 합니다.
4. **GPU 메모리**: RL 실험은 메모리 사용량이 크므로 GPU당 하나씩만 실행됩니다.
