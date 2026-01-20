# Orthogonal Loss Prototype Learning 문제 해결

## 문제 발견

WandB 로그에서 orthogonal loss가 minimize되지 않는 문제를 발견했습니다.

### 관찰된 현상

- **Reconstruction loss**: 정상적으로 감소 (0.25 → 0)
- **Orthogonal loss**: 거의 변하지 않음
  - t50102, t50002: 1600-1700에서 고정
  - t50101: 300-350에서 고정
  - t50000: 50-100에서 고정

### 근본 원인

**QR decomposition이 forward pass에서 수행되어 gradient가 차단됨**

```python
# 기존 코드 (문제)
def _compute_clop_orthogonal_loss(self, z, labels=None):
    # Forward pass 중에 QR decomposition 수행
    with torch.no_grad():  # ❌ Gradient 차단!
        Q, R = torch.linalg.qr(self.orthogonal_prototypes.T)
        self.orthogonal_prototypes.data = Q.T * prototype_scale  # ❌ Gradient 차단!
    
    # Pull loss 계산
    pull_loss = torch.mean((z - selected_prototypes) ** 2)
    # 하지만 prototype은 이미 고정되어 있음 → gradient가 흐르지 않음
```

**결과**:
1. Prototype이 학습되지 않음 (gradient가 차단됨)
2. z만 움직여야 하는데, z는 다른 loss (reconstruction, KL)에도 영향을 받음
3. z가 prototype으로 끌려가지 않음
4. Orthogonal loss가 minimize되지 않음

---

## 해결 방법

### 핵심 아이디어

**QR decomposition을 backward pass 후에만 수행하여 forward pass에서는 gradient를 유지**

### 수정 내용

1. **Forward pass에서 QR decomposition 제거**:
   - `_compute_clop_orthogonal_loss`에서 QR decomposition 제거
   - Prototype을 직접 사용하여 pull loss 계산 (gradient 유지)

2. **Backward pass 후 QR decomposition 수행**:
   - `_hook_on_batch_end`에서 QR decomposition 수행
   - Gradient 계산 후이므로 prototype 학습에 영향 없음

### 수정된 코드

```python
def _compute_clop_orthogonal_loss(self, z, labels=None):
    # QR decomposition 제거 (forward pass에서 gradient 유지)
    # Prototype을 직접 사용 (learnable)
    
    # Pull loss: z와 prototype 모두 학습됨
    selected_prototypes = self.orthogonal_prototypes[orthogonal_labels]
    pull_loss = torch.mean((z - selected_prototypes) ** 2)  # ✅ Gradient 유지
    
    # Orthonormal constraint: loss로만 적용
    PTP = torch.matmul(self.orthogonal_prototypes, self.orthogonal_prototypes.T)
    orthonorm_loss = torch.norm(PTP - identity, p='fro') ** 2  # ✅ Gradient 유지
    
    return orthogonal_loss, pull_loss, orthonorm_loss

def _hook_on_batch_end(self, ctx):
    # ... statistics update ...
    
    # Backward pass 후 QR decomposition 수행 (gradient 계산 후)
    if self.vpl_orthogonal_weight > 0.0 and self.orthogonal_prototypes is not None:
        self._apply_orthonormal_constraint_to_prototypes()  # ✅ Gradient 차단 없음
```

---

## 학습 메커니즘

### 수정 전 (문제)

```
Forward:
  z = encoder(x)  # z는 학습됨
  prototype = QR(prototype)  # ❌ no_grad()로 고정
  pull_loss = ||z - prototype||²  # z만 움직임

Backward:
  ∂pull_loss/∂z ≠ 0  # z는 업데이트됨
  ∂pull_loss/∂prototype = 0  # ❌ prototype은 고정 (gradient 차단)
```

**결과**: z만 움직이고 prototype은 고정 → z가 prototype으로 끌려가지 않음

### 수정 후 (해결)

```
Forward:
  z = encoder(x)  # z는 학습됨
  prototype = self.orthogonal_prototypes  # ✅ learnable parameter
  pull_loss = ||z - prototype||²  # z와 prototype 모두 움직임

Backward:
  ∂pull_loss/∂z ≠ 0  # z는 업데이트됨
  ∂pull_loss/∂prototype ≠ 0  # ✅ prototype도 업데이트됨

After Backward:
  prototype = QR(prototype)  # 정규화 (다음 forward를 위해)
```

**결과**: z와 prototype 모두 움직임 → 서로 가까워짐 → pull loss minimize

---

## 기대 효과

### 1. Prototype 학습

- Prototype이 z의 분포를 따라 학습됨
- Pull loss를 통해 prototype이 z에 가까워짐

### 2. Orthogonal Loss Minimize

- Pull loss: `||z - prototype||²`가 감소
- z와 prototype이 서로 가까워짐
- Orthogonal loss가 실제로 minimize됨

### 3. Loss Balance

- Prototype이 학습되므로 pull loss가 더 빠르게 감소
- Orthogonal loss의 scale이 적절해짐
- KL loss와 reconstruction loss도 의미있는 기여

---

## 추가 고려사항

### 1. Orthonormal Constraint

**현재 방법**:
- Forward: Orthonormal constraint를 loss로만 적용
- Backward 후: QR decomposition으로 정규화

**장점**:
- Prototype이 학습됨
- Orthonormal constraint도 유지됨

**단점**:
- QR decomposition이 매 batch마다 수행됨 (약간의 오버헤드)

### 2. Alternative: Differentiable QR

더 정교한 방법은 differentiable QR decomposition을 사용하는 것이지만, 구현이 복잡합니다.

현재 방법이 더 실용적이고 효과적입니다.

---

## 검증 방법

### 1. WandB 로그 확인

수정 후 다음을 확인:
- Orthogonal loss가 감소하는지
- Pull loss가 감소하는지
- Prototype의 norm이 변하는지

### 2. t-SNE 시각화

- Prototype이 z 분포로 이동하는지 확인
- 클라이언트별 z가 prototype에 가까워지는지 확인

### 3. Loss 비율

- Orthogonal loss가 total loss의 적절한 비율을 차지하는지 확인
- KL loss와 reconstruction loss도 의미있는 기여를 하는지 확인

---

## 요약

### 문제
- QR decomposition이 forward pass에서 수행되어 prototype의 gradient가 차단됨
- Prototype이 학습되지 않아 orthogonal loss가 minimize되지 않음

### 해결
- QR decomposition을 backward pass 후로 이동
- Forward pass에서 prototype을 직접 사용하여 gradient 유지
- Prototype과 z 모두 학습되어 서로 가까워짐

### 기대 효과
- Orthogonal loss가 실제로 minimize됨
- Prototype이 z 분포를 따라 학습됨
- Loss balance가 개선됨
