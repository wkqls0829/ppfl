# KL Loss와 Orthogonal Loss Minimization 문제 분석

이 문서는 KL loss와 orthogonal loss가 minimize되지 않는 문제의 원인과 해결 방안을 분석합니다.

## 문제 현상

실험 50001, 50002, 50101, 50102에서 관찰된 현상:

### Loss 값 예시 (50002, Round 0-4)

| Round | KL Loss | Reconstruction Loss | Orthogonal Loss | Total Loss |
|-------|---------|---------------------|-----------------|------------|
| 0 | 0.069-0.084 | 0.17-0.24 | 1695-1759 | 1696-1760 |
| 1 | 0.061-0.084 | 0.16-0.26 | 1667-1706 | 1668-1707 |
| 2 | 0.061-0.085 | 0.18-0.29 | 1714-1751 | 1715-1752 |
| 3 | 0.080-0.114 | 0.16-0.27 | 1656-1736 | 1657-1737 |
| 4 | 0.070-0.118 | 0.24-0.29 | 1718-1719 | 1719-1720 |

**관찰**:
- KL loss: ~0.07-0.12 (거의 변하지 않음, 매우 작음)
- Reconstruction loss: ~0.15-0.30 (거의 변하지 않음, 매우 작음)
- Orthogonal loss: ~1656-1759 (매우 큼, total loss의 99% 이상 차지)
- Total loss ≈ Orthogonal loss (KL + Reconstruction이 무시됨)

### Loss 값 예시 (50101, Round 0-4)

| Round | KL Loss | Reconstruction Loss | Orthogonal Loss | Total Loss |
|-------|---------|---------------------|-----------------|------------|
| 0 | 0.061-0.084 | 0.18-0.30 | 274-282 | 275-282 |
| 1 | 0.071-0.085 | 0.13-0.19 | 272-276 | 273-277 |
| 2 | 0.070-0.085 | 0.19-0.29 | 274-279 | 275-280 |
| 3 | 0.083-0.175 | 0.13-0.20 | 274-276 | 275-277 |
| 4 | 0.118-0.170 | 0.19-0.24 | 274-274 | 275-275 |

**관찰**:
- KL loss: ~0.06-0.17 (약간 증가하는 추세이지만 여전히 작음)
- Reconstruction loss: ~0.13-0.30 (거의 변하지 않음)
- Orthogonal loss: ~274-282 (매우 큼, total loss의 99% 이상 차지)
- Total loss ≈ Orthogonal loss

---

## 원인 분석

### 1. Loss Scale 불균형 (가장 가능성 높음) ⭐

**문제**:
```
Total Loss = Reconstruction Loss + λ_kl * KL Loss + λ_ortho * Orthogonal Loss
           ≈ 0.2 + 10.0 * 0.08 + 1000.0 * 1700
           ≈ 0.2 + 0.8 + 1,700,000
           ≈ 1,700,000 (orthogonal loss가 99.9% 차지)
```

**현재 설정**:
- 50002: `vpl_kl_weight: 10.0`, `vpl_orthogonal_weight: 1000.0`
- 50101: `vpl_kl_weight: 1.0`, `vpl_orthogonal_weight: 100.0`

**Orthogonal loss의 실제 값**:
- Pull loss: `||z - prototype||²` (z와 prototype의 거리 제곱)
- z의 scale이 크면 (예: norm ≈ 10-50), pull loss = 100-2500
- `vpl_orthogonal_weight: 1000.0`이면 → 100,000-2,500,000

**결과**:
- Orthogonal loss의 gradient가 너무 커서 다른 loss의 gradient를 압도
- Optimizer가 orthogonal loss만 minimize하려고 함
- KL loss와 reconstruction loss는 거의 업데이트되지 않음

### 2. z Embedding Scale 문제

**가능한 원인**:
- z의 norm이 너무 큼 (예: ||z|| ≈ 10-50)
- Prototype의 scale (5.0)과 z의 scale이 맞지 않음
- Feature extractor나 encoder가 z를 너무 크게 생성

**확인 방법**:
```python
# z의 평균 norm 확인
z_norm = torch.norm(z, dim=-1).mean().item()
# 일반적으로 latent_dim=32일 때, ||z|| ≈ 5-10이 적절
```

### 3. Prototype Initialization 문제

**현재 구현**:
```python
# Prototype을 scale=5.0으로 초기화
self.orthogonal_prototypes = nn.Parameter(
    torch.randn(num_prototypes, latent_dim) * 2.0
)
# QR decomposition 후 scale=5.0으로 설정
Q, R = torch.linalg.qr(self.orthogonal_prototypes.T)
self.orthogonal_prototypes.data = Q.T * 5.0
```

**문제**:
- z가 원점 근처에 있으면 (||z|| ≈ 1-2), prototype (||prototype|| = 5.0)과 거리가 멈
- Pull loss = ||z - prototype||² ≈ (5.0 - 1.0)² = 16.0
- `vpl_orthogonal_weight: 1000.0`이면 → 16,000

### 4. Gradient Flow 문제

**현재 상황**:
```
∂L_total/∂θ = ∂L_recon/∂θ + λ_kl * ∂L_kl/∂θ + λ_ortho * ∂L_ortho/∂θ
             ≈ 0.01 + 10.0 * 0.001 + 1000.0 * 1000.0
             ≈ 0.01 + 0.01 + 1,000,000
             ≈ 1,000,000 (orthogonal loss gradient가 압도)
```

**결과**:
- Optimizer step이 orthogonal loss만 줄이는 방향으로 진행
- KL loss와 reconstruction loss의 gradient가 무시됨

### 5. Learning Rate 문제

**현재 설정**:
- `lr: 0.00001` (매우 작음)

**문제**:
- Orthogonal loss가 너무 커서, 작은 learning rate로는 KL/reconstruction loss가 업데이트되지 않음
- 또는 orthogonal loss의 gradient가 너무 커서, learning rate를 줄여도 다른 loss가 무시됨

---

## 해결 방안

### 방안 1: Loss Coefficient 조정 (권장) ⭐

**목표**: Loss scale을 균형있게 맞추기

**전략**:
1. **Orthogonal weight 감소**: `vpl_orthogonal_weight`를 10-100배 감소
   - 50002: `1000.0` → `10.0` 또는 `1.0`
   - 50101: `100.0` → `1.0` 또는 `0.1`

2. **KL weight 조정**: `vpl_kl_weight`를 적절히 조정
   - 현재: 1.0-10.0
   - 권장: 0.1-1.0 (reconstruction loss와 비슷한 scale)

3. **Loss normalization**: 각 loss를 normalize하여 scale 맞추기
   ```python
   # 예시: 각 loss를 평균으로 나누어 normalize
   normalized_kl = kl_loss / (kl_loss.mean() + 1e-8)
   normalized_ortho = orthogonal_loss / (orthogonal_loss.mean() + 1e-8)
   ```

### 방안 2: z Scale 정규화

**목표**: z의 norm을 적절한 범위로 유지

**구현**:
```python
# Encoder 출력 후 z를 정규화
z, mu, logvar = self.variational_encoder(extracted_features)
# z의 norm을 target_norm (예: 5.0)으로 정규화
z_norm = torch.norm(z, dim=-1, keepdim=True)
target_norm = 5.0  # Prototype scale과 맞춤
z = z / (z_norm + 1e-8) * target_norm
```

**장점**:
- z와 prototype의 scale이 맞춰짐
- Pull loss가 적절한 범위로 유지됨

### 방안 3: Prototype Scale 조정

**목표**: Prototype을 z의 초기 scale에 맞춤

**구현**:
```python
# z의 초기 norm을 측정
initial_z_norm = torch.norm(z, dim=-1).mean().item()
# Prototype scale을 z의 norm에 맞춤
prototype_scale = initial_z_norm  # 또는 약간 더 크게 (예: 1.5 * initial_z_norm)
self.orthogonal_prototypes.data = Q.T * prototype_scale
```

### 방안 4: Loss Weight Scheduling

**목표**: Training 초기에는 reconstruction에 집중, 후반에 orthogonal loss 강화

**구현**:
```python
# Round에 따라 weight 조정
current_round = ctx.cur_round
total_rounds = self._cfg.federate.total_round_num

# Orthogonal weight를 점진적으로 증가
orthogonal_weight_schedule = self.vpl_orthogonal_weight * (current_round / total_rounds)
# 또는 exponential schedule
orthogonal_weight_schedule = self.vpl_orthogonal_weight * (1 - np.exp(-current_round / 10))
```

### 방안 5: Gradient Clipping

**목표**: Orthogonal loss의 gradient가 너무 커지지 않도록 제한

**구현**:
```python
# Orthogonal loss 계산 후 gradient clipping
orthogonal_loss_val, pull_loss, orthonorm_loss = self._compute_clop_orthogonal_loss(z)
# Gradient clipping (loss 자체가 아니라 gradient)
orthogonal_loss_val = torch.clamp(orthogonal_loss_val, max=100.0)  # 예시
```

### 방안 6: Learning Rate 조정

**목표**: Orthogonal loss에 대한 learning rate를 별도로 설정

**구현**:
```python
# Optimizer에 parameter group 추가
optimizer = torch.optim.AdamW([
    {'params': [p for n, p in model.named_parameters() if 'orthogonal_prototypes' not in n],
     'lr': 0.00001},
    {'params': [p for n, p in model.named_parameters() if 'orthogonal_prototypes' in n],
     'lr': 0.000001}  # Prototypes에 대해 더 작은 LR
])
```

---

## 권장 해결 순서

### 1단계: Loss Coefficient 조정 (즉시 적용 가능)

```yaml
# 50002 수정안
vpl_kl_weight: 1.0  # 10.0 → 1.0
vpl_orthogonal_weight: 10.0  # 1000.0 → 10.0 (100배 감소)

# 50101 수정안
vpl_kl_weight: 0.1  # 1.0 → 0.1
vpl_orthogonal_weight: 1.0  # 100.0 → 1.0 (100배 감소)
```

**예상 결과**:
- Total loss ≈ 0.2 + 1.0 * 0.08 + 10.0 * 1.7 = 0.2 + 0.08 + 17.0 ≈ 17.3
- Orthogonal loss가 여전히 크지만, KL/reconstruction loss도 의미있는 기여

### 2단계: z Scale 확인 및 정규화

```python
# z의 norm을 로깅하여 확인
logger.info(f"z norm: {torch.norm(z, dim=-1).mean().item():.4f}")
# 만약 ||z|| > 10이면 정규화 적용
```

### 3단계: Prototype Scale 동적 조정

```python
# z의 초기 scale에 맞춰 prototype scale 조정
initial_z_norm = torch.norm(z.detach(), dim=-1).mean().item()
if initial_z_norm > 0.1:
    prototype_scale = initial_z_norm * 1.2  # z보다 약간 크게
else:
    prototype_scale = 5.0  # Default
```

---

## 현재 Loss 구성 분석

### Total Loss 공식

```python
vpl_loss = reconstruction_loss + self.vpl_kl_weight * kl_loss + orthogonal_loss
```

**실제 값 (50002, Round 0)**:
```
vpl_loss = 0.20 + 10.0 * 0.08 + 1700.0
         = 0.20 + 0.80 + 1700.0
         = 1701.0
```

**비율**:
- Reconstruction: 0.20 / 1701.0 = 0.01% (거의 무시됨)
- KL: 0.80 / 1701.0 = 0.05% (거의 무시됨)
- Orthogonal: 1700.0 / 1701.0 = 99.94% (압도적)

### Orthogonal Loss 구성

```python
orthogonal_loss = vpl_orthogonal_weight * pull_loss + vpl_orthogonal_orthonorm_weight * orthonorm_loss
                 = 1000.0 * pull_loss + 0.1 * orthonorm_loss
```

**Pull loss 추정**:
```
pull_loss = ||z - prototype||²
          ≈ (||z|| - ||prototype||)²  (대략적)
          ≈ (10 - 5)² = 25  (z의 norm이 10일 때)
          또는 (1 - 5)² = 16  (z의 norm이 1일 때)

orthogonal_loss = 1000.0 * 25 = 25,000  (너무 큼!)
또는
orthogonal_loss = 1000.0 * 16 = 16,000  (여전히 너무 큼!)
```

**실제 로그 값**:
- `vpl_orthogonal_loss: 1695-1759` (50002)
- 이는 `pull_loss ≈ 1.7` 정도를 의미
- 즉, `||z - prototype|| ≈ √1.7 ≈ 1.3` (적절한 거리)
- 하지만 `vpl_orthogonal_weight: 1000.0`이므로 → 1700.0

---

## 결론 및 권장사항

### 핵심 문제

1. **Loss scale 불균형**: Orthogonal loss coefficient가 너무 큼
2. **Gradient 압도**: Orthogonal loss의 gradient가 다른 loss를 압도
3. **Optimization 방향**: Optimizer가 orthogonal loss만 minimize

### 즉시 적용 가능한 해결책

1. **Orthogonal weight 대폭 감소**:
   - 50002: `1000.0` → `10.0` (100배 감소)
   - 50101: `100.0` → `1.0` (100배 감소)

2. **KL weight 조정**:
   - Reconstruction loss와 비슷한 scale로 (0.1-1.0)

3. **Loss monitoring**:
   - 각 loss component의 실제 값을 로깅하여 확인
   - Gradient norm도 모니터링

### 장기적 개선

1. **Adaptive loss weighting**: Training 중 loss scale에 따라 weight 조정
2. **Loss normalization**: 각 loss를 normalize하여 scale 맞추기
3. **Gradient balancing**: 각 loss의 gradient norm을 비슷하게 유지

---

## 참고: 적절한 Loss Scale

일반적으로 VAE/VPL에서 권장되는 loss scale:

```
Reconstruction Loss: 1.0 (baseline)
KL Loss: 0.1-1.0 (regularization)
Orthogonal Loss: 0.1-10.0 (constraint)
```

**목표 비율**:
- Reconstruction: 50-80%
- KL: 10-30%
- Orthogonal: 10-20%

**현재 비율 (50002)**:
- Reconstruction: 0.01%
- KL: 0.05%
- Orthogonal: 99.94%

**목표 비율로 조정**:
```
Total Loss ≈ 1.0 (recon) + 0.5 (KL) + 0.5 (ortho) = 2.0
```

이를 위해:
- `vpl_kl_weight: 0.5` (reconstruction과 비슷)
- `vpl_orthogonal_weight: 0.5` (reconstruction과 비슷)
- 또는 pull loss를 normalize하여 scale 맞추기
