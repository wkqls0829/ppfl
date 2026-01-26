# 50024 Hyperparameter Search 계획

## 현재 설정 (Baseline: 50024)

```yaml
# VPL-GP Core Parameters
vpl_latent_dim: 32
vpl_kl_weight: 0.1
vpl_gp_temperature: 1.0
vpl_max_logvar: -3.0

# Orthogonal Loss Parameters
vpl_orthogonal_weight: 1.0
vpl_orthogonal_orthonorm_weight: 0.1
vpl_prototype_scale: 5.0

# Training Parameters
lr: 0.0001
batch_size: 8
grad_accum_step: 4
local_update_steps: 30
```

## Search 가능한 Hyperparameters

### 1. VPL Core Parameters

#### 1.1 `vpl_kl_weight` (KL Divergence Weight)
- **현재 값**: 0.1
- **역할**: ELBO에서 KL(q(z|x) || p(z)) 항의 가중치
  - 낮을수록: z가 prior에서 더 멀어질 수 있음 (더 표현력 있는 z)
  - 높을수록: z가 prior에 가까워짐 (더 규제됨)
- **Search 범위**: [0.01, 0.05, 0.1, 0.2, 0.5]
- **예상 영향**: 
  - 낮은 값: 더 다양한 z 분포, 하지만 overfitting 위험
  - 높은 값: 더 규제된 z, 하지만 표현력 감소

#### 1.2 `vpl_gp_temperature` (Gumbel-Softmax Temperature)
- **현재 값**: 1.0
- **역할**: Gumbel-Softmax prior의 temperature
  - 낮을수록: 더 discrete한 샘플링 (harder)
  - 높을수록: 더 smooth한 샘플링 (softer)
- **Search 범위**: [0.5, 0.75, 1.0, 1.5, 2.0]
- **예상 영향**:
  - 낮은 값: 더 명확한 client 구분, 하지만 학습 불안정 가능
  - 높은 값: 더 부드러운 학습, 하지만 구분이 덜 명확할 수 있음

#### 1.3 `vpl_latent_dim` (Latent Dimension)
- **현재 값**: 32
- **역할**: Latent z의 차원
  - 낮을수록: 더 compact한 표현, 하지만 정보 손실 가능
  - 높을수록: 더 표현력 있는 z, 하지만 overfitting 위험
- **Search 범위**: [16, 24, 32, 48, 64]
- **예상 영향**:
  - 낮은 값: 더 간단한 모델, 빠른 학습, 하지만 표현력 제한
  - 높은 값: 더 복잡한 표현, 하지만 학습 시간 증가 및 overfitting 위험

#### 1.4 `vpl_max_logvar` (Maximum Log Variance)
- **현재 값**: -3.0
- **역할**: Posterior variance의 상한 (sigma <= exp(-3.0) ≈ 0.223)
  - 낮을수록: 더 tight한 variance (더 확신 있는 z)
  - 높을수록: 더 loose한 variance (더 불확실한 z)
- **Search 범위**: [-4.0, -3.5, -3.0, -2.5, -2.0]
- **예상 영향**:
  - 낮은 값: 더 확신 있는 추론, 하지만 flexibility 감소
  - 높은 값: 더 유연한 추론, 하지만 불확실성 증가

### 2. Orthogonal Loss Parameters

#### 2.1 `vpl_orthogonal_weight` (Pull Loss Weight)
- **현재 값**: 1.0
- **역할**: z를 prototype으로 끌어당기는 loss의 가중치
  - 낮을수록: prototype에 대한 제약이 약함
  - 높을수록: z가 prototype에 더 가까워짐
- **Search 범위**: [0.0, 0.5, 1.0, 2.0, 5.0]
- **예상 영향**:
  - 0.0: Orthogonal loss 비활성화 (baseline 비교용)
  - 낮은 값: 약한 prototype 제약
  - 높은 값: 강한 prototype 제약, 더 명확한 client 구분

#### 2.2 `vpl_orthogonal_orthonorm_weight` (Orthonormal Constraint Weight)
- **현재 값**: 0.1
- **역할**: Prototype 간 orthonormal 제약의 가중치
  - 낮을수록: Prototype 간 orthogonality 제약이 약함
  - 높을수록: Prototype이 더 orthogonal하게 유지됨
- **Search 범위**: [0.0, 0.05, 0.1, 0.2, 0.5]
- **예상 영향**:
  - 낮은 값: Prototype이 더 자유롭게 움직임
  - 높은 값: Prototype이 더 orthogonal하게 유지, 하지만 학습 제약

#### 2.3 `vpl_prototype_scale` (Prototype Distance from Origin)
- **현재 값**: 5.0
- **역할**: Prototype이 origin으로부터 떨어진 거리
  - 낮을수록: Prototype이 origin에 가까움
  - 높을수록: Prototype이 origin에서 멀리 떨어짐
- **Search 범위**: [2.0, 3.0, 5.0, 7.0, 10.0]
- **예상 영향**:
  - 낮은 값: Prototype이 더 중심에 가까움, z가 더 compact
  - 높은 값: Prototype이 더 멀리, z가 더 분산됨

### 3. Training Parameters

#### 3.1 `lr` (Learning Rate)
- **현재 값**: 0.0001
- **역할**: Optimizer의 learning rate
- **Search 범위**: [0.00005, 0.0001, 0.0002, 0.0005]
- **예상 영향**:
  - 낮은 값: 더 안정적인 학습, 하지만 느린 수렴
  - 높은 값: 빠른 학습, 하지만 불안정할 수 있음

#### 3.2 `batch_size` (Batch Size)
- **현재 값**: 8
- **역할**: 각 step의 batch size (effective = batch_size * grad_accum_step = 32)
- **Search 범위**: [4, 8, 16] (메모리 제약 고려)
- **예상 영향**:
  - 작은 값: 더 많은 gradient 업데이트, 하지만 noisy
  - 큰 값: 더 안정적인 gradient, 하지만 메모리 사용 증가

#### 3.3 `grad_accum_step` (Gradient Accumulation Steps)
- **현재 값**: 4
- **역할**: Effective batch size를 위한 gradient accumulation
- **Search 범위**: [2, 4, 8] (batch_size와 함께 조정)
- **예상 영향**: Effective batch size 유지하면서 메모리 사용 조절

## Search 전략

### Phase 1: Orthogonal Loss Weight Search (가장 중요)
**목표**: Orthogonal loss의 최적 가중치 찾기 (5배 범위로 확장)

| Experiment ID | vpl_orthogonal_weight | vpl_orthogonal_orthonorm_weight | vpl_prototype_scale |
|---------------|----------------------|--------------------------------|---------------------|
| 52000 | 0.2 | 0.1 | 5.0 |
| 52001 | 1.0 | 0.1 | 5.0 (baseline) |
| 52002 | 5.0 | 0.1 | 5.0 |
| 52003 | 1.0 | 0.02 | 5.0 |
| 52004 | 1.0 | 0.5 | 5.0 |
| 52005 | 1.0 | 0.1 | 2.0 |
| 52006 | 1.0 | 0.1 | 10.0 |

**예상 실험 수**: 7개
**예상 시간**: 각 실험당 ~6-8시간 (50 rounds)
**변경 사항**: 
- `orthogonal_weight`: 0.5→0.2, 2.0→5.0 (5배 범위)
- `orthonorm_weight`: 0.05→0.02, 0.2→0.5 (5배 범위)
- `prototype_scale`: 3.0→2.0, 7.0→10.0 (5배 범위)

### Phase 2: VPL Core Parameters Search
**목표**: KL weight와 temperature 최적화 (5배 범위로 확장)

| Experiment ID | vpl_kl_weight | vpl_gp_temperature | vpl_latent_dim |
|---------------|---------------|-------------------|----------------|
| 52007 | 0.02 | 1.0 | 32 |
| 52008 | 0.1 | 1.0 | 32 (baseline) |
| 52009 | 0.5 | 1.0 | 32 |
| 52010 | 0.1 | 0.5 | 32 |
| 52011 | 0.1 | 2.0 | 32 |
| 52012 | 0.1 | 1.0 | 16 |
| 52013 | 0.1 | 1.0 | 64 |

**예상 실험 수**: 7개
**변경 사항**:
- `kl_weight`: 0.05→0.02, 0.2→0.5 (5배 범위)
- `temperature`: 0.75→0.5, 1.5→2.0 (2배 범위, 더 넓게)
- `latent_dim`: 24→16, 48→64 (2배 범위)

### Phase 3: Learning Rate Search
**목표**: 최적 learning rate 찾기

| Experiment ID | lr |
|---------------|-----|
| 52014 | 0.00005 |
| 52015 | 0.0001 (baseline) |
| 52016 | 0.0002 |

**예상 실험 수**: 3개

### Phase 4: Combined Best Parameters
**목표**: Phase 1-3에서 찾은 최적 값들을 조합

| Experiment ID | Description |
|---------------|-------------|
| 52017 | Best orthogonal + best VPL core + best LR |

**예상 실험 수**: 1개

## 총 예상 실험 수

- Phase 1: 7개
- Phase 2: 7개
- Phase 3: 3개
- Phase 4: 1개
- **총계**: 18개 실험

## 평가 지표

각 실험에서 다음 지표를 모니터링:

1. **Accuracy**: 선택 정확도 (가장 중요)
2. **Loss**: 전체 loss (train_avg_loss)
3. **VPL Losses**:
   - `vpl_kl_loss`: KL divergence loss
   - `vpl_reconstruction_loss`: Reconstruction loss
   - `vpl_orthogonal_loss`: Orthogonal loss
4. **t-SNE Visualization**: z 분포의 시각적 품질
5. **Client Separation**: Harmlessness/Helpfulness client 간 분리 정도

## 실행 방법

각 실험은 다음 형식으로 실행:

```bash
# Config 파일 생성 (예: cfg/vpl-gp/hhst-ortho-50025.yaml)
# Script 파일 생성 (예: scripts/vpl-gp/hhst-ortho-50025.sh)
bash scripts/vpl-gp/hhst-ortho-50025.sh
```

## 주의사항

1. **GPU 할당**: 각 실험은 독립적인 GPU에서 실행 (병렬 실행 가능)
2. **Checkpoint 저장**: 각 실험의 checkpoint는 나중에 RL training에 사용
3. **WandB Tracking**: 모든 실험을 `fvpl-selector` 프로젝트에서 추적
4. **Baseline 비교**: 50024를 baseline으로 모든 실험과 비교

## 예상 결과 분석

1. **Orthogonal Loss Weight**: 
   - 너무 낮으면: client 구분이 약함
   - 너무 높으면: 학습이 불안정하거나 overfitting
   - 최적: client 간 명확한 구분 + 안정적인 학습

2. **KL Weight**:
   - 낮으면: 더 표현력 있는 z, 하지만 overfitting 위험
   - 높으면: 더 규제된 z, 하지만 표현력 감소

3. **Temperature**:
   - 낮으면: 더 discrete한 샘플링, 명확한 구분
   - 높으면: 더 smooth한 샘플링, 부드러운 학습
