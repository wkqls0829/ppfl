# Ablation Study Plan

VPL 대비 Gumbel-Softmax Prior와 Orthogonal Loss의 효과를 확인하는 ablation study입니다.

## 실험 구성

### Baseline (이미 메인 테이블에 포함)
1. **FedVPL**: VPL with standard normal prior, no orthogonal loss
   - `vpl_use_gp_prior: False`
   - `vpl_orthogonal_weight: 0.0`

2. **FedVPA-GP**: VPL with Gumbel-Softmax prior + orthogonal loss
   - `vpl_use_gp_prior: True`
   - `vpl_orthogonal_weight: 1.0`

### Ablation Study (새로 추가)
3. **VPL + GP**: VPL with Gumbel-Softmax prior, no orthogonal loss
   - `vpl_use_gp_prior: True`
   - `vpl_orthogonal_weight: 0.0`
   - GP prior의 효과만 확인

4. **VPL + Ortho**: VPL with standard normal prior, with orthogonal loss
   - `vpl_use_gp_prior: False`
   - `vpl_orthogonal_weight: 1.0`
   - Orthogonal loss의 효과만 확인

## 실험 설계

### 모델
- **Gemma-2B**: 2개 방법 × 3 client counts = 6개 실험
- **Qwen 2**: 2개 방법 × 3 client counts = 6개 실험
- **총 12개 selector 실험**

### Client Counts
- N=10, N=50, N=100 (각 방법당 3개)

## TID 번호 체계

### Gemma-2B
- **Selector**: 62300-62332
  - VPL + GP: 62300-62302 (N=10,50,100)
  - VPL + Ortho: 62310-62312 (N=10,50,100)
- **RL**: 63300-63332
  - VPL + GP: 63300-63302 (N=10,50,100)
  - VPL + Ortho: 63310-63312 (N=10,50,100)

### Qwen 2
- **Selector**: 62400-62432
  - VPL + GP: 62400-62402 (N=10,50,100)
  - VPL + Ortho: 62410-62412 (N=10,50,100)
- **RL**: 63400-63432
  - VPL + GP: 63400-63402 (N=10,50,100)
  - VPL + Ortho: 63410-63412 (N=10,50,100)

## 하이퍼파라미터 설정

### VPL + GP (Gumbel-Softmax prior만)
- `vpl_use_gp_prior: True`
- `vpl_kl_weight: 0.02` (메인 테이블과 동일)
- `vpl_gp_temperature: 1.0`
- `vpl_orthogonal_weight: 0.0` (orthogonal loss 비활성화)
- `vpl_orthogonal_orthonorm_weight: 0.0`
- `vpl_use_manual_orthogonal_labels: False`
- `vpl_num_prototypes: 0`
- `vpl_prototype_scale: 0.0`

### VPL + Ortho (Orthogonal loss만)
- `vpl_use_gp_prior: False` (standard normal prior)
- `vpl_kl_weight: 0.1` (VPL baseline과 동일)
- `vpl_orthogonal_weight: 1.0` (orthogonal loss 활성화)
- `vpl_orthogonal_orthonorm_weight: 0.0` (메인 테이블과 동일)
- `vpl_use_manual_orthogonal_labels: True`
- `vpl_num_prototypes: 2`
- `vpl_prototype_scale: 5.0`

## 비교 분석

### 1. GP Prior의 효과
- **FedVPL** vs **VPL + GP**: GP prior가 성능에 미치는 영향
- **VPL + Ortho** vs **FedVPA-GP**: GP prior가 orthogonal loss와 함께 사용될 때의 효과

### 2. Orthogonal Loss의 효과
- **FedVPL** vs **VPL + Ortho**: Orthogonal loss가 성능에 미치는 영향
- **VPL + GP** vs **FedVPA-GP**: Orthogonal loss가 GP prior와 함께 사용될 때의 효과

### 3. 조합 효과
- **FedVPL** vs **FedVPA-GP**: GP prior + Orthogonal loss의 조합 효과

## 파일 구조

```
scripts/main_table/ablation/
├── README.md (이 파일)
├── run_selector_ablation_gemma.sh      # Gemma-2B selector ablation 실행
├── run_rl_ablation_gemma.sh            # Gemma-2B RL ablation 실행
├── run_selector_ablation_qwen.sh       # Qwen 2 selector ablation 실행
├── run_rl_ablation_qwen.sh            # Qwen 2 RL ablation 실행
├── submit_all_ablation_gemma.sh        # 모든 Gemma-2B ablation 작업 제출
└── submit_all_ablation_qwen.sh          # 모든 Qwen 2 ablation 작업 제출

cfg/
├── vpl-gp-no-ortho/  # VPL + GP (no orthogonal loss)
│   ├── hhst.yaml
│   └── hrl.yaml
└── vpl-ortho/        # VPL + Ortho (standard normal prior)
    ├── hhst.yaml
    └── hrl.yaml
```

## 실행 방법

### 1. Selector 실험 실행

#### Gemma-2B
```bash
# VPL + GP
sbatch scripts/main_table/ablation/run_selector_ablation_gemma.sh vplgp 10 62300

# VPL + Ortho
sbatch scripts/main_table/ablation/run_selector_ablation_gemma.sh vplortho 10 62310

# 모든 ablation selector 실험 제출
bash scripts/main_table/ablation/submit_all_ablation_gemma.sh
```

#### Qwen 2
```bash
# VPL + GP
sbatch scripts/main_table/ablation/run_selector_ablation_qwen.sh vplgp 10 62400

# VPL + Ortho
sbatch scripts/main_table/ablation/run_selector_ablation_qwen.sh vplortho 10 62410

# 모든 ablation selector 실험 제출
bash scripts/main_table/ablation/submit_all_ablation_qwen.sh
```

### 2. RL 실험 실행

Selector 실험이 완료된 후:

#### Gemma-2B
```bash
# VPL + GP RL
sbatch scripts/main_table/ablation/run_rl_ablation_gemma.sh vplgp 10 63300 62300

# VPL + Ortho RL
sbatch scripts/main_table/ablation/run_rl_ablation_gemma.sh vplortho 10 63310 62310

# 모든 ablation RL 실험 제출
bash scripts/main_table/ablation/submit_rl_ablation_gemma.sh
```

#### Qwen 2
```bash
# VPL + GP RL
sbatch scripts/main_table/ablation/run_rl_ablation_qwen.sh vplgp 10 63400 62400

# VPL + Ortho RL
sbatch scripts/main_table/ablation/run_rl_ablation_qwen.sh vplortho 10 63410 62410

# 모든 ablation RL 실험 제출
bash scripts/main_table/ablation/submit_rl_ablation_qwen.sh
```

## 예상 결과 분석

### 메트릭 비교
1. **Selector 성능**: Accuracy, Loss, KL Loss, Reconstruction Loss
2. **RL 성능**: Win Rate, Helpfulness, Harmlessness
3. **Latent Space**: t-SNE 시각화로 클라이언트 preference 분포 비교

### 통계적 유의성
- 각 방법별로 3번의 실험 (N=10, 50, 100)을 통해 일관성 확인
- 메인 테이블 결과와 비교하여 각 컴포넌트의 기여도 정량화
