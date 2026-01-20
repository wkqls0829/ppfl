# Prototype: Fixed vs Learnable

## 질문: Prototype은 고정되어야 하는 게 맞는가?

**답변**: CLOP 논문에 따르면, **prototype은 일반적으로 고정된 orthonormal basis**로 사용됩니다. 하지만 두 가지 접근 방식 모두 유효합니다.

---

## 1. Fixed Prototypes (CLOP 표준 방식)

### 개념
- Prototype을 **고정된 orthonormal basis**로 사용
- Identity matrix를 기반으로 초기화: `P = I * scale`
- z embedding만 prototype에 가까워지도록 학습

### 장점
1. **명확한 기준점**: 고정된 기준으로 z를 정렬
2. **해석 가능성**: 각 prototype이 독립적인 축을 나타냄
3. **안정성**: Prototype이 변하지 않아 학습이 안정적
4. **표준 방식**: CLOP 논문의 표준 구현

### 단점
1. **유연성 부족**: 데이터 분포가 고정된 prototype과 맞지 않을 수 있음
2. **강제 정렬**: z가 억지로 고정된 prototype에 맞춰야 함

### 구현
```python
# Fixed prototypes: Identity matrix 기반
prototypes = torch.eye(num_prototypes, latent_dim) * prototype_scale
self.register_buffer('orthogonal_prototypes', prototypes)  # Buffer (not learnable)
```

---

## 2. Learnable Prototypes (현재 수정된 방식)

### 개념
- Prototype을 **학습 가능한 parameter**로 사용
- 초기화 후 orthonormal constraint를 loss로만 적용
- z와 prototype이 서로 가까워지도록 학습

### 장점
1. **유연성**: 데이터 분포에 맞춰 최적의 prototype 위치 찾기
2. **효율성**: z와 prototype이 서로 움직여 빠르게 수렴
3. **적응성**: 복잡한 데이터 분포에 대응 가능

### 단점
1. **불안정성**: Prototype이 예상치 못한 방향으로 학습될 수 있음
2. **해석 어려움**: 학습된 prototype의 의미가 불명확할 수 있음
3. **비표준**: CLOP 논문의 표준 구현과 다름

### 구현
```python
# Learnable prototypes: Parameter로 초기화
self.orthogonal_prototypes = nn.Parameter(
    torch.randn(num_prototypes, latent_dim) * 2.0
)
# Orthonormalize initially
Q, R = torch.linalg.qr(self.orthogonal_prototypes.T)
self.orthogonal_prototypes.data = Q.T * prototype_scale

# Forward pass에서 gradient 유지
pull_loss = ||z - prototype||²  # z와 prototype 모두 학습됨

# Backward 후 QR로 정규화 (orthonormal constraint 유지)
```

---

## 3. 왜 이전에 Loss가 Minimize되지 않았는가?

### Fixed Prototypes를 사용하는 경우
- Prototype이 고정되어 있으므로, **z만 움직여야 함**
- 하지만 z는 다른 loss (reconstruction, KL)에도 영향을 받음
- **Loss scale 불균형**으로 z가 prototype으로 끌려가지 않음
- 해결책: Orthogonal loss weight를 조정하거나, z의 학습률을 조정

### Learnable Prototypes를 사용하는 경우 (이전 코드)
- Prototype이 학습되어야 하는데, **QR decomposition이 forward pass에서 gradient를 차단**
- Prototype이 업데이트되지 않아 z만 움직여야 함
- 하지만 z는 다른 loss에도 영향을 받아 prototype으로 끌려가지 않음
- 해결책: QR decomposition을 backward 후로 이동 (현재 수정)

---

## 4. 어떤 방식을 선택해야 하는가?

### Fixed Prototypes를 선택하는 경우
- **CLOP 논문의 표준 구현**을 따르고 싶을 때
- **해석 가능한 prototype**이 필요할 때
- **안정적인 학습**이 중요할 때
- **명확한 기준점**이 필요할 때

### Learnable Prototypes를 선택하는 경우
- **데이터 분포에 맞춘 최적화**가 필요할 때
- **빠른 수렴**이 중요할 때
- **복잡한 데이터 분포**를 다룰 때
- **유연한 학습**이 필요할 때

---

## 5. 현재 구현

현재 코드는 **두 가지 옵션을 모두 지원**합니다:

```python
# Config에서 선택 가능
llm:
  vpl_prototypes_learnable: False  # Default: False (Fixed)
  # 또는
  vpl_prototypes_learnable: True   # Learnable
```

### Fixed Prototypes (기본값)
```python
# Identity matrix 기반 고정 prototype
prototypes = torch.eye(num_prototypes, latent_dim) * prototype_scale
self.register_buffer('orthogonal_prototypes', prototypes)
```

### Learnable Prototypes
```python
# 학습 가능한 prototype
self.orthogonal_prototypes = nn.Parameter(...)
# Forward: gradient 유지
# Backward 후: QR로 정규화
```

---

## 6. 권장 사항

### 실험 초기 단계
- **Fixed Prototypes**로 시작 (CLOP 표준)
- Loss weight를 조정하여 z가 prototype에 가까워지도록 함
- 안정적인 학습 확인

### 실험 최적화 단계
- **Learnable Prototypes**로 전환
- 더 빠른 수렴과 유연한 학습 확인
- 두 방식의 성능 비교

---

## 7. 요약

| Aspect | Fixed Prototypes | Learnable Prototypes |
|--------|------------------|---------------------|
| **CLOP 표준** | ✅ Yes | ❌ No |
| **해석 가능성** | ✅ High | ⚠️ Medium |
| **안정성** | ✅ High | ⚠️ Medium |
| **유연성** | ❌ Low | ✅ High |
| **수렴 속도** | ⚠️ Medium | ✅ Fast |
| **구현 복잡도** | ✅ Simple | ⚠️ Complex |

**결론**: CLOP 논문의 표준은 **Fixed Prototypes**이지만, **Learnable Prototypes**도 유효한 선택입니다. 실험 목적에 따라 선택하세요.
