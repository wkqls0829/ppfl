# CLOP Orthogonal Loss 작동 방식

## 개요

CLOP (Contrastive Learning with Orthonormal Prototypes) 논문을 기반으로 한 orthogonal loss는 neural collapse를 방지하고, embedding들이 서로 구분 가능한 orthogonal subspace를 형성하도록 유도합니다.

**논문**: [Preventing Collapse in Contrastive Learning with Orthonormal Prototypes (CLOP)](https://arxiv.org/pdf/2403.18699)

## 1. Neural Collapse 문제

### 문제점

Contrastive learning에서 neural collapse는 다음과 같은 현상을 의미합니다:

1. **차원 축소**: Embedding들이 낮은 차원 공간으로 수렴
2. **구분 불가**: 클래스들이 구분되지 않음 (co-linear)
3. **공간 낭비**: Full-rank space를 활용하지 못함

### CLOP의 해결책

CLOP는 **orthonormal prototypes**를 사용하여:
- Embedding들이 **orthogonal linear subspaces**를 형성하도록 유도
- Full-rank space를 활용
- 더 구분 가능한 embedding 생성

## 2. CLOP Loss 구성

### 기본 Contrastive Loss

```python
# InfoNCE loss (기존)
L_contrastive = -log(exp(sim(z_i, z_i^+)) / Σ_j exp(sim(z_i, z_j)))
```

여기서 `sim`은 cosine similarity입니다.

### CLOP Orthogonal Loss

```python
# Orthonormal prototypes: P = [p_1, p_2, ..., p_k] ∈ R^(d×k)
# 각 prototype은 unit norm이고 서로 orthogonal

# 1. Prototype 정규화 및 Orthogonal 제약
P_normalized = P / ||P||_F  # Frobenius norm으로 정규화
P_orthogonal = gram_schmidt(P_normalized)  # Gram-Schmidt로 orthogonal화

# 2. Embedding을 prototype에 pull
L_pull = Σ_i ||z_i - P[y_i]||²  # y_i는 클래스 label

# 3. Orthonormal 제약 (prototype들이 orthonormal이 되도록)
L_orthonormal = ||P^T P - I||²_F  # P^T P = I (identity matrix)

# 4. 전체 CLOP loss
L_clop = L_contrastive + λ_pull * L_pull + λ_orthonormal * L_orthonormal
```

## 3. 수학적 정의

### Orthonormal Prototypes

```
P = [p_1, p_2, ..., p_k] ∈ R^(d×k)

제약 조건:
1. ||p_i|| = 1  (unit norm)
2. p_i^T p_j = 0  (i ≠ j, orthogonal)
3. P^T P = I_k  (orthonormal matrix)
```

### Loss 함수

```
L_clop = L_contrastive + λ_pull * L_pull + λ_orthonormal * L_orthonormal

여기서:
- L_contrastive: 기존 contrastive loss (InfoNCE 등)
- L_pull = (1/|B|) Σ_{(x,y)∈B} ||z(x) - p_y||²
- L_orthonormal = ||P^T P - I_k||²_F
```

## 4. VPL-GP에서의 적용

### VPL-GP Context

VPL-GP에서는 latent z에 대해 orthogonal loss를 적용합니다:

```python
# 1. Orthonormal prototypes 초기화
# 각 클라이언트/클래스에 대한 prototype
num_prototypes = num_clients  # 또는 num_classes
prototypes = nn.Parameter(torch.randn(num_prototypes, latent_dim))
# 초기화: 큰 scale로 시작 (예: 2.0 * randn)

# 2. Orthonormal 제약 적용
def apply_orthonormal_constraint(prototypes):
    """
    Gram-Schmidt 과정으로 orthonormal화
    """
    # QR decomposition 사용 (더 안정적)
    Q, R = torch.qr(prototypes.T)  # (latent_dim, num_prototypes)
    prototypes_orthonormal = Q.T  # (num_prototypes, latent_dim)
    return prototypes_orthonormal

# 또는 직접 계산
def gram_schmidt(prototypes):
    """
    Gram-Schmidt orthogonalization
    """
    num_prototypes, latent_dim = prototypes.shape
    orthonormal = torch.zeros_like(prototypes)
    
    for i in range(num_prototypes):
        v = prototypes[i]
        # 이전 벡터들에 대한 projection 제거
        for j in range(i):
            v = v - torch.dot(v, orthonormal[j]) * orthonormal[j]
        # 정규화
        norm = torch.norm(v)
        if norm > 1e-8:
            orthonormal[i] = v / norm
        else:
            # 0 벡터인 경우 랜덤 벡터로 대체
            orthonormal[i] = torch.randn(latent_dim)
            orthonormal[i] = orthonormal[i] / torch.norm(orthonormal[i])
    
    return orthonormal
```

### Orthogonal Loss 계산

```python
def compute_orthogonal_loss(z, prototypes, labels, orthogonal_weight, orthonorm_weight):
    """
    CLOP 기반 orthogonal loss 계산
    
    Args:
        z: Latent embeddings (batch_size, latent_dim)
        prototypes: Orthonormal prototypes (num_prototypes, latent_dim)
        labels: Class labels (batch_size,) - 각 샘플이 어떤 prototype에 속하는지
        orthogonal_weight: Pull loss weight
        orthonorm_weight: Orthonormal constraint weight
    
    Returns:
        orthogonal_loss: Total orthogonal loss
    """
    batch_size, latent_dim = z.shape
    num_prototypes, _ = prototypes.shape
    
    # 1. Pull loss: z를 해당 prototype에 가깝게
    # labels[i]는 z[i]가 속하는 prototype의 인덱스
    selected_prototypes = prototypes[labels]  # (batch_size, latent_dim)
    pull_loss = torch.mean((z - selected_prototypes) ** 2)
    
    # 2. Orthonormal constraint: P^T P = I
    PTP = torch.matmul(prototypes, prototypes.T)  # (num_prototypes, num_prototypes)
    identity = torch.eye(num_prototypes, device=prototypes.device)
    orthonorm_loss = torch.norm(PTP - identity, p='fro') ** 2
    
    # 3. Total orthogonal loss
    orthogonal_loss = orthogonal_weight * pull_loss + orthonorm_weight * orthonorm_loss
    
    return orthogonal_loss, pull_loss, orthonorm_loss
```

## 5. 전체 파이프라인

### Training Step

```python
# Forward pass
z, mu, logvar = variational_encoder(features)  # (batch_size, latent_dim)

# Orthonormal prototypes 업데이트 (매 step마다)
prototypes = apply_orthonormal_constraint(self.prototypes)

# Orthogonal labels 결정
# Option 1: Manual labels (서버에서 받음)
if use_manual_labels:
    orthogonal_labels = self.orthogonal_label  # (batch_size,)
else:
    # Option 2: 자동으로 가장 가까운 prototype 선택
    similarities = torch.matmul(z, prototypes.T)  # (batch_size, num_prototypes)
    orthogonal_labels = torch.argmax(similarities, dim=1)  # (batch_size,)

# Orthogonal loss 계산
orthogonal_loss, pull_loss, orthonorm_loss = compute_orthogonal_loss(
    z, prototypes, orthogonal_labels, 
    orthogonal_weight=self.vpl_orthogonal_weight,
    orthonorm_weight=self.vpl_orthogonal_orthonorm_weight
)

# Total loss
total_loss = vpl_loss + orthogonal_loss
```

### Prototype 업데이트

```python
# 매 training step마다
def update_prototypes(self):
    """
    Prototypes를 orthonormal하게 유지
    """
    with torch.no_grad():
        # QR decomposition으로 orthonormal화
        Q, R = torch.qr(self.prototypes.T)
        self.prototypes.data = Q.T
        
        # 또는 Gram-Schmidt
        self.prototypes.data = gram_schmidt(self.prototypes)
```

## 6. CLOP vs ETF (Equiangular Tight Frame)

### ETF 방식 (기존)

```
- Simplex ETF 구조 강제
- 모든 prototype이 같은 각도로 배치
- 제약이 너무 강함
```

### CLOP 방식 (제안)

```
- Orthogonal linear subspaces 형성
- 더 유연한 구조
- Full-rank space 활용
- 더 구분 가능한 embedding
```

## 7. 구현 예시

### 초기화

```python
class VPLRewardChoiceTrainer:
    def __init__(self, ...):
        # Orthogonal loss hyperparameters
        self.vpl_orthogonal_weight = getattr(config.llm, 'vpl_orthogonal_weight', 0.0)
        self.vpl_orthogonal_orthonorm_weight = getattr(config.llm, 'vpl_orthogonal_orthonorm_weight', 0.1)
        self.vpl_use_manual_orthogonal_labels = getattr(config.llm, 'vpl_use_manual_orthogonal_labels', False)
        
        # Orthonormal prototypes 초기화
        num_prototypes = getattr(config.llm, 'vpl_num_prototypes', self.num_clients)
        self.orthogonal_prototypes = nn.Parameter(
            torch.randn(num_prototypes, self.vpl_latent_dim) * 2.0  # 큰 scale로 시작
        )
        
        # Orthogonal label (서버에서 받음)
        self.orthogonal_label = None
```

### Loss 계산

```python
def _compute_clop_orthogonal_loss(self, z, labels=None):
    """
    CLOP orthogonal loss 계산
    """
    if self.vpl_orthogonal_weight == 0.0:
        return torch.tensor(0.0, device=z.device), torch.tensor(0.0, device=z.device), torch.tensor(0.0, device=z.device)
    
    batch_size, latent_dim = z.shape
    num_prototypes, _ = self.orthogonal_prototypes.shape
    
    # Orthonormal constraint 적용
    with torch.no_grad():
        # QR decomposition으로 orthonormal화
        Q, R = torch.qr(self.orthogonal_prototypes.T)
        self.orthogonal_prototypes.data = Q.T
    
    # Orthogonal labels 결정
    if self.vpl_use_manual_orthogonal_labels and self.orthogonal_label is not None:
        # Manual labels 사용 (서버에서 받음)
        orthogonal_labels = self.orthogonal_label  # (batch_size,)
    else:
        # 자동으로 가장 가까운 prototype 선택
        similarities = torch.matmul(z, self.orthogonal_prototypes.T)  # (batch_size, num_prototypes)
        orthogonal_labels = torch.argmax(similarities, dim=1)  # (batch_size,)
    
    # Pull loss: z를 해당 prototype에 가깝게
    selected_prototypes = self.orthogonal_prototypes[orthogonal_labels]  # (batch_size, latent_dim)
    pull_loss = torch.mean((z - selected_prototypes) ** 2)
    
    # Orthonormal constraint: P^T P = I
    PTP = torch.matmul(self.orthogonal_prototypes, self.orthogonal_prototypes.T)  # (num_prototypes, num_prototypes)
    identity = torch.eye(num_prototypes, device=z.device)
    orthonorm_loss = torch.norm(PTP - identity, p='fro') ** 2
    
    # Total orthogonal loss
    orthogonal_loss = self.vpl_orthogonal_weight * pull_loss + \
                     self.vpl_orthogonal_orthonorm_weight * orthonorm_loss
    
    return orthogonal_loss, pull_loss, orthonorm_loss
```

### Training Loop

```python
def _hook_on_batch_forward(self, ctx):
    # ... VPL forward pass ...
    z, mu, logvar = self.variational_encoder(extracted_features)
    
    # Orthogonal loss 계산
    orthogonal_loss, pull_loss, orthonorm_loss = self._compute_clop_orthogonal_loss(z)
    
    # Total loss
    vpl_loss = reconstruction_loss + self.vpl_kl_weight * kl_loss
    total_loss = vpl_loss + orthogonal_loss
    
    # Backward
    total_loss.backward()
```

## 8. 핵심 포인트

1. **Orthonormal Prototypes**: 각 prototype은 unit norm이고 서로 orthogonal
2. **Pull Loss**: Embedding을 해당 prototype에 가깝게 pull
3. **Orthonormal Constraint**: `P^T P = I` 제약으로 orthonormal 유지
4. **QR Decomposition**: 매 step마다 prototypes를 orthonormal화
5. **Manual Labels**: 서버에서 받은 label 사용 가능

## 9. 효과

- ✅ Neural collapse 방지
- ✅ Full-rank space 활용
- ✅ 더 구분 가능한 embedding
- ✅ 안정적인 학습 (다양한 learning rate에서)

## 참고

- CLOP 논문: [arXiv:2403.18699](https://arxiv.org/pdf/2403.18699)
- Neural Collapse: Jing et al., "Understanding Contrastive Learning Requires Incorporating Inductive Biases", 2021
