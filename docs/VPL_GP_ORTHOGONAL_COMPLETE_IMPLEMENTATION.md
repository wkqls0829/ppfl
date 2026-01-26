# VPL-GP with Orthogonal Loss: 완전한 구현 가이드

## 목차

1. [전체 아키텍처 개요](#전체-아키텍처-개요)
2. [VPL (Variational Preference Learning)](#vpl-variational-preference-learning)
3. [Gumbel-Softmax Prior (GP Prior)](#gumbel-softmax-prior-gp-prior)
4. [Orthogonal Loss (CLOP)](#orthogonal-loss-clop)
5. [전체 학습 흐름](#전체-학습-흐름)
6. [컴포넌트 상호작용](#컴포넌트-상호작용)
7. [코드 구조](#코드-구조)
8. [설정 및 하이퍼파라미터](#설정-및-하이퍼파라미터)

---

## 전체 아키텍처 개요

### 시스템 구성

```
┌─────────────────────────────────────────────────────────────┐
│                    Federated Learning Server                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Collect z distributions (μ, log σ²) from clients    │  │
│  │  Compute mixture prior: p_mixture(z) = Σ w_i N(μ_i)  │  │
│  │  Compute orthogonal labels via k-means                │  │
│  │  Broadcast: prior + orthogonal labels                 │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ (broadcast)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Client 1, 2, ..., N                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  1. Feature Extraction                               │  │
│  │     - Extract [chosen_emb, rejected_emb, difference]  │  │
│  │     - Pass through feature_extractor MLP             │  │
│  │                                                       │  │
│  │  2. Variational Encoding                             │  │
│  │     - Encode to z, μ, log σ²                         │  │
│  │     - Sample z ~ q(z|x)                             │  │
│  │                                                       │  │
│  │  3. Latent Conditioning                              │  │
│  │     - Project z to logit adjustment                  │  │
│  │     - Add to model logits                            │  │
│  │                                                       │  │
│  │  4. Loss Computation                                 │  │
│  │     - Reconstruction loss                            │  │
│  │     - KL divergence (vs mixture prior)               │  │
│  │     - Orthogonal loss (pull + orthonorm)             │  │
│  │                                                       │  │
│  │  5. Send z distribution to server                    │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 핵심 컴포넌트

1. **Feature Extractor**: Preference features 추출
2. **Variational Encoder**: Features → Latent z distribution
3. **Latent Projection**: z → Logit adjustment
4. **Mixture Prior**: 다른 클라이언트들의 z 분포 조합
5. **Orthogonal Prototypes**: 직교 프로토타입들
6. **Server Aggregator**: z 분포 수집 및 prior 계산

---

## VPL (Variational Preference Learning)

### 수학적 배경

VPL은 사용자별 preference를 latent variable `z`로 모델링합니다:

**ELBO (Evidence Lower Bound)**:
```
ELBO = E_q(z|x)[log p(y|z,x)] - KL(q(z|x) || p(z))
     = Reconstruction Loss - KL Divergence
```

### 구현 세부사항

#### 1. Feature Extraction

**파일**: `federatedscope/llm/trainer/vpl_reward_choice_trainer.py`

```python
def _extract_preference_features(self, logits, labels, choices, hidden_states):
    """
    Extract preference features using embedding difference.
    
    Strategy depends on config:
    1. vpl_use_difference_only=False: [chosen_emb, rejected_emb, difference]
    2. vpl_use_difference_only=True: [difference] only (removes general info)
    """
    # Extract embeddings at choice token positions
    chosen_emb = extract_embedding_at_choice(hidden_states, labels, choices[0])
    rejected_emb = extract_embedding_at_choice(hidden_states, labels, choices[1])
    
    # Compute difference
    feature_diff = chosen_emb - rejected_emb
    
    # Choose feature combination based on config
    if vpl_use_difference_only:
        # Use only difference (removes general information)
        features = feature_diff  # (embedding_dim,)
    else:
        # Concatenate for richer representation
        features = torch.cat([chosen_emb, rejected_emb, feature_diff], dim=-1)  # (3 * embedding_dim,)
    
    return features
```

**특징**:
- Single forward pass: LLM의 hidden states 재사용
- **Difference-only mode**: `vpl_use_difference_only=True`로 general information 제거
- **Full mode**: `vpl_use_difference_only=False`로 richer representation 사용

#### 2. Feature Extractor Network

**Input dimension depends on `vpl_use_difference_only`**:

```python
if vpl_use_difference_only:
    # Input: difference only
    self.feature_extractor = nn.Sequential(
        nn.Linear(embedding_dim, 512),  # Input: difference only
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 128)  # Output: 128-dim features
    )
else:
    # Input: [chosen, rejected, difference]
    self.feature_extractor = nn.Sequential(
        nn.Linear(embedding_dim * 3, 512),  # Input: [chosen, rejected, diff]
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 128)  # Output: 128-dim features
    )
```

**역할**: Raw embeddings → Encoder-ready features
- **Difference-only**: Removes general information, keeps only preference signal
- **Full mode**: Includes general information from chosen/rejected embeddings

#### 3. Variational Encoder

**파일**: `federatedscope/llm/model/variational_encoder.py`

```python
class VariationalEncoder(nn.Module):
    def forward(self, x):
        # Encode to distribution parameters
        mu, logvar = self.encoder(x)  # (batch, latent_dim)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, logvar
```

**출력**:
- `z`: 샘플링된 latent vector (batch, latent_dim)
- `mu`: Posterior mean (batch, latent_dim)
- `logvar`: Posterior log variance (batch, latent_dim)

#### 4. Latent Conditioning

```python
# Project latent to choice logit adjustments
latent_adjustment = self.latent_projection(z)  # (batch, num_choices)

# Expand to match logits shape
latent_adjustment_expanded = latent_adjustment.unsqueeze(1).expand(
    -1, seq_len, -1
)  # (batch, seq_len, num_choices)

# Condition model on latent
conditioned_logits = original_logits + latent_adjustment_expanded
```

**효과**: `z`에 따라 모델의 선택이 달라짐

#### 5. Loss Computation

```python
# Reconstruction loss
reconstruction_loss = CrossEntropyLoss(conditioned_logits, labels)

# KL divergence (standard VPL: vs N(0, I))
kl_loss = -0.5 * torch.sum(
    1 + logvar - mu.pow(2) - logvar.exp(), dim=-1
).mean()

# Total VPL loss
vpl_loss = reconstruction_loss + vpl_kl_weight * kl_loss
```

---

## Gumbel-Softmax Prior (GP Prior)

### 수학적 배경

**Mixture Prior**:
```
p_mixture(z) = Σ_i w_i * N(z; μ_i, σ_i²)
```

여기서:
- `μ_i, σ_i²`: 클라이언트 `i`의 posterior 분포
- `w_i`: 클라이언트 `i`의 가중치 (현재는 uniform: 1/N)
- 합은 현재 클라이언트를 제외한 모든 클라이언트에 대해 계산

### 구현 세부사항

#### 1. VariationalEncoderGP

**파일**: `federatedscope/llm/model/variational_encoder_gp.py`

```python
class VariationalEncoderGP(VariationalEncoder):
    def __init__(self, input_dim, latent_dim, hidden_dims, 
                 temperature, num_clients):
        super().__init__(input_dim, latent_dim, hidden_dims)
        self.temperature = temperature
        self.num_clients = num_clients
        
        # Prior components (updated from server)
        self.prior_mus = None      # (num_clients, latent_dim)
        self.prior_logvars = None  # (num_clients, latent_dim)
        self.prior_weights = None  # (num_clients,)
```

#### 2. KL Divergence with Mixture Prior

**핵심**: Log-sum-exp trick으로 수치 안정성 확보

```python
def kl_divergence(self, mu, logvar, use_gumbel_prior=True):
    # Sample z from posterior
    z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
    
    # Compute log q(z|x)
    log_q = -0.5 * torch.sum(
        np.log(2 * np.pi) + logvar + (z - mu).pow(2) / torch.exp(logvar),
        dim=-1
    )
    
    # Compute log p_mixture(z) for each component
    log_p_components = []
    for i in range(num_components):
        mu_i = self.prior_mus[i]
        logvar_i = self.prior_logvars[i]
        weight_i = self.prior_weights[i]
        
        log_p_i = -0.5 * torch.sum(
            np.log(2 * np.pi) + logvar_i + 
            (z - mu_i.unsqueeze(0)).pow(2) / torch.exp(logvar_i.unsqueeze(0)),
            dim=-1
        )
        log_p_i = log_p_i + torch.log(weight_i + 1e-8)
        log_p_components.append(log_p_i)
    
    # Log-sum-exp trick for numerical stability
    log_p_stack = torch.stack(log_p_components, dim=0)
    log_p_max = torch.max(log_p_stack, dim=0, keepdim=True)[0]
    log_p_mixture = log_p_max.squeeze(0) + torch.log(
        torch.sum(torch.exp(log_p_stack - log_p_max), dim=0) + 1e-8
    )
    
    # KL divergence
    kl = (log_q - log_p_mixture).mean()
    return kl
```

**수치 안정성**:
- Log-sum-exp trick: `log(Σ exp(x_i)) = max(x_i) + log(Σ exp(x_i - max(x_i)))`
- Overflow 방지: 각 항에서 최대값을 빼서 계산

#### 3. Gumbel-Softmax Sampling

**용도**: Prior에서 샘플링 (현재는 KL 계산에 직접 사용하지 않음)

```python
def sample_prior(self, batch_size, use_gumbel=True):
    # Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand(num_components) + 1e-8))
    
    # Gumbel-Softmax
    logits = torch.log(self.prior_weights + 1e-8) + gumbel_noise / self.temperature
    probs = F.softmax(logits / self.temperature, dim=0)
    
    # Sample from selected component
    selected_idx = torch.multinomial(probs, batch_size, replacement=True)
    z_samples = []
    for idx in selected_idx:
        mu_i = self.prior_mus[idx]
        logvar_i = self.prior_logvars[idx]
        z_i = self.reparameterize(mu_i, logvar_i)
        z_samples.append(z_i)
    
    return torch.stack(z_samples)
```

#### 4. Server-side Prior Aggregation

**파일**: `federatedscope/llm/llm_local/server.py`

```python
def _collect_vpl_gp_prior_distributions(self):
    """
    Collect z distributions from clients and form mixture prior.
    """
    train_msg_buffer = self.msg_buffer['train'][self.state]
    
    client_mus = []
    client_logvars = []
    client_weights = []
    client_ids = []
    
    for client_id in train_msg_buffer.keys():
        model_para = train_msg_buffer[client_id][1]
        
        # Extract z distribution
        mu = model_para['client_z_mu']  # (latent_dim,)
        logvar = model_para['client_z_logvar']  # (latent_dim,)
        sample_size = model_para.get('sample_size', 1)
        
        client_mus.append(mu)
        client_logvars.append(logvar)
        client_weights.append(sample_size)
        client_ids.append(client_id)
    
    # Normalize weights
    total_weight = sum(client_weights)
    client_weights = [w / total_weight for w in client_weights]
    
    # Stack tensors
    client_mus = torch.stack(client_mus)  # (num_clients, latent_dim)
    client_logvars = torch.stack(client_logvars)  # (num_clients, latent_dim)
    client_weights = torch.tensor(client_weights)
    
    # Update prior
    self.vpl_gp_prior_mus = client_mus
    self.vpl_gp_prior_logvars = client_logvars
    self.vpl_gp_prior_weights = client_weights
```

#### 5. Prior Broadcasting

```python
def broadcast_model_para(self, ...):
    # Broadcast VPL-GP prior to clients
    if self.vpl_gp_prior_mus is not None and self.state > 0:
        prior_content = {
            'vpl_gp_prior_mus': self.vpl_gp_prior_mus.cpu().tolist(),
            'vpl_gp_prior_logvars': self.vpl_gp_prior_logvars.cpu().tolist(),
            'vpl_gp_prior_weights': self.vpl_gp_prior_weights.cpu().tolist(),
        }
        
        for receiver in selected_clients:
            self.comm_manager.send(
                Message(msg_type='vpl_gp_prior', ...,
                       content=prior_content)
            )
```

---

## Orthogonal Loss (CLOP)

### 수학적 배경

**목적**: Neural collapse 방지, 직교 서브스페이스 형성

**Loss 구성**:
```
L_orthogonal = λ_pull * L_pull + λ_orthonorm * L_orthonorm
```

**Pull Loss**:
```
L_pull = (1/|B|) Σ ||z(x) - p_y||²
```

**Orthonormal Constraint**:
```
L_orthonorm = ||P^T P - I||²_F
```

### 구현 세부사항

#### 1. Prototype Initialization

```python
# Initialize prototypes
num_prototypes = config.llm.vpl_num_prototypes  # Default: num_clients
prototype_scale = config.llm.vpl_prototype_scale  # Default: 5.0

self.orthogonal_prototypes = nn.Parameter(
    torch.randn(num_prototypes, latent_dim) * 2.0
)

# Orthonormalize using QR decomposition
Q, R = torch.linalg.qr(self.orthogonal_prototypes.T)
# Scale orthonormalized prototypes to be further from origin
self.orthogonal_prototypes.data = Q.T * prototype_scale  # Distance from origin = prototype_scale
```

**Prototype Scale**:
- QR decomposition makes prototypes orthonormal (norm=1, distance from origin=1)
- Scaling by `prototype_scale` places prototypes at distance `prototype_scale` from origin
- **Default**: 5.0 (prototypes are 5 units away from origin)
- **Effect**: Better separation from z embeddings, which are typically near origin

#### 2. Label Assignment

**서버 측 (K-means)**:

```python
def _compute_balanced_orthogonal_labels(self):
    # Collect z means from clients
    z_means = [client_z_mu for client in participating_clients]
    z_means = np.array(z_means)  # (num_clients, latent_dim)
    
    # K-means clustering
    # For hh-rlhf: k=2 (fixed)
    # For others: k=config.llm.vpl_num_prototypes
    dataset_type = config.data.type.lower()
    if 'hh-rlhf' in dataset_type or 'hrl' in dataset_type:
        n_clusters = 2
    else:
        n_clusters = config.llm.vpl_num_prototypes
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(z_means)
    
    # Create label dict: {client_id: orthogonal_label}
    self.vpl_orthogonal_client_labels = {
        client_id: int(label) 
        for client_id, label in zip(client_ids, labels)
    }
```

**클라이언트 측 (사용)**:

```python
def _compute_clop_orthogonal_loss(self, z):
    # Orthonormalize prototypes (each forward pass)
    Q, R = torch.linalg.qr(self.orthogonal_prototypes.T)
    self.orthogonal_prototypes.data = Q.T
    
    # Determine labels
    if self.vpl_use_manual_orthogonal_labels and self.orthogonal_label is not None:
        # Use server-assigned label
        orthogonal_labels = torch.full(
            (batch_size,), self.orthogonal_label, device=z.device
        )
    else:
        # Auto-assign to closest prototype
        similarities = torch.matmul(z, self.orthogonal_prototypes.T)
        orthogonal_labels = torch.argmax(similarities, dim=1)
    
    # Pull loss
    selected_prototypes = self.orthogonal_prototypes[orthogonal_labels]
    pull_loss = torch.mean((z - selected_prototypes) ** 2)
    
    # Orthonormal constraint
    PTP = torch.matmul(self.orthogonal_prototypes, self.orthogonal_prototypes.T)
    identity = torch.eye(num_prototypes, device=z.device)
    orthonorm_loss = torch.norm(PTP - identity, p='fro') ** 2
    
    # Total loss
    orthogonal_loss = (self.vpl_orthogonal_weight * pull_loss + 
                      self.vpl_orthogonal_orthonorm_weight * orthonorm_loss)
    
    return orthogonal_loss, pull_loss, orthonorm_loss
```

#### 3. Prototype Maintenance

**QR Decomposition**: 매 forward pass마다 실행하여 orthonormality 유지

```python
# In _compute_clop_orthogonal_loss
with torch.no_grad():
    Q, R = torch.linalg.qr(self.orthogonal_prototypes.T)
    self.orthogonal_prototypes.data = Q.T
```

**효과**: `P^T P = I` 항상 만족

---

## 전체 학습 흐름

### Round-based Federated Learning

#### Round 0 (초기화)

1. **서버**:
   - 클라이언트들이 join할 때까지 대기
   - `trigger_for_start()` 호출

2. **클라이언트**:
   - 모델 초기화
   - VPL 컴포넌트 초기화:
     - Feature extractor
     - Variational encoder (GP prior 사용 시 `VariationalEncoderGP`)
     - Latent projection
     - Orthogonal prototypes (orthogonal loss 활성화 시)

3. **서버 → 클라이언트**:
   - 모델 파라미터 브로드캐스트
   - Prior는 아직 없음 (Round 0에서는 브로드캐스트 안 함)

#### Round N (N > 0)

**1. 서버 → 클라이언트 (Broadcast)**

```python
# In server.broadcast_model_para()
# 1. 모델 파라미터 브로드캐스트
super().broadcast_model_para(...)

# 2. VPL-GP Prior 브로드캐스트 (if enabled)
if vpl_use_gp_prior and prior exists:
    broadcast_prior(prior_mus, prior_logvars, prior_weights)

# 3. Orthogonal Labels 브로드캐스트 (if enabled)
if orthogonal_labels exist:
    broadcast_orthogonal_labels(client_labels_dict)
```

**2. 클라이언트 로컬 학습**

```python
# In client.train()
for batch in train_loader:
    # Forward pass
    outputs = model(input_ids, attention_mask)
    logits = outputs.logits
    hidden_states = outputs.hidden_states[-1]  # Last layer
    
    # 1. Feature Extraction
    preference_features = extract_preference_features(
        logits, labels, choices, hidden_states
    )
    extracted_features = feature_extractor(preference_features)
    
    # 2. Variational Encoding
    z, mu, logvar = variational_encoder(extracted_features)
    
    # 3. Latent Conditioning
    latent_adjustment = latent_projection(z)
    conditioned_logits = logits + latent_adjustment
    
    # 4. Loss Computation
    reconstruction_loss = CrossEntropyLoss(conditioned_logits, labels)
    
    # KL divergence
    if vpl_use_gp_prior:
        kl_loss = variational_encoder.kl_divergence(mu, logvar)  # vs mixture
    else:
        kl_loss = standard_kl_divergence(mu, logvar)  # vs N(0,I)
    
    # Orthogonal loss
    if vpl_orthogonal_weight > 0:
        orthogonal_loss, _, _ = _compute_clop_orthogonal_loss(z)
    
    # Total loss
    total_loss = (reconstruction_loss + 
                  vpl_kl_weight * kl_loss + 
                  orthogonal_loss)
    
    # Backward pass
    total_loss.backward()
    optimizer.step()
    
    # Collect z for aggregation
    z_history.append(z.detach())
    z_mu_history.append(mu.detach())
    z_logvar_history.append(logvar.detach())
```

**3. 클라이언트 → 서버 (Upload)**

```python
# In client.train() (end of round)
# Compute average z distribution
client_z_mu = torch.mean(torch.stack(z_mu_history), dim=0)
client_z_logvar = torch.mean(torch.stack(z_logvar_history), dim=0)

# Send to server
model_para_all = {
    'model_params': trainer.get_model_para(),
    'client_z_mu': client_z_mu.cpu(),
    'client_z_logvar': client_z_logvar.cpu(),
    'sample_size': num_samples,
    'client_z_values': z_values_for_visualization,  # For t-SNE
    'client_orthogonal_prototypes': orthogonal_prototypes.cpu()  # For visualization
}
```

**4. 서버 Aggregation**

```python
# In server._perform_federated_aggregation()
# 1. 모델 파라미터 집계 (FedAvg)
aggregated_params = federated_averaging(client_params)

# 2. z 분포 수집 (for GP prior)
_collect_vpl_gp_prior_distributions()

# 3. Orthogonal labels 계산
_compute_balanced_orthogonal_labels()

# 4. t-SNE 시각화 (if enabled)
_visualize_cross_client_z()
```

**5. 다음 Round 준비**

```python
# In server._start_new_training_round()
# 다음 round를 위한 브로드캐스트 준비
# (다음 round에서 1번으로 돌아감)
```

### 전체 Loss 계산 흐름

```
Input: (input_ids, labels, choices)
  │
  ├─→ LLM Forward Pass
  │     ├─→ logits: (batch, seq_len, vocab_size)
  │     └─→ hidden_states: (batch, seq_len, hidden_dim)
  │
  ├─→ Feature Extraction
  │     ├─→ Extract embeddings at choice positions
  │     ├─→ Compute [chosen_emb, rejected_emb, difference]
  │     └─→ feature_extractor → extracted_features: (batch, 128)
  │
  ├─→ Variational Encoding
  │     ├─→ variational_encoder → z, μ, log σ²
  │     └─→ z: (batch, latent_dim)
  │
  ├─→ Latent Conditioning
  │     ├─→ latent_projection(z) → latent_adjustment: (batch, num_choices)
  │     └─→ conditioned_logits = logits + latent_adjustment
  │
  └─→ Loss Computation
        ├─→ Reconstruction Loss: CrossEntropy(conditioned_logits, labels)
        ├─→ KL Loss: KL(q(z|x) || p(z))
        │     ├─→ Standard VPL: vs N(0, I)
        │     └─→ VPL-GP: vs p_mixture(z) = Σ w_i N(μ_i, σ_i²)
        └─→ Orthogonal Loss (if enabled)
              ├─→ Pull Loss: ||z - prototype[label]||²
              └─→ Orthonorm Loss: ||P^T P - I||²_F
```

---

## 컴포넌트 상호작용

### 1. Feature Extractor ↔ Variational Encoder

```
Raw Embeddings (3 * embedding_dim)
    ↓
Feature Extractor MLP
    ↓
Extracted Features (128-dim)
    ↓
Variational Encoder
    ↓
Latent z (latent_dim)
```

**특징**:
- Single forward pass: LLM의 hidden states 재사용
- Preference-focused: 차이 계산으로 일반 정보 제거

### 2. Variational Encoder ↔ GP Prior

```
Client i's Posterior: q_i(z|x)
    ↓
Server Aggregation
    ↓
Mixture Prior: p_mixture(z) = Σ_j≠i w_j * q_j(z)
    ↓
Broadcast to Client i
    ↓
KL Divergence: KL(q_i(z|x) || p_mixture(z))
```

**효과**:
- Knowledge sharing: 다른 클라이언트들의 preference 학습
- Regularization: Mixture prior가 더 풍부한 구조 제공

### 3. Orthogonal Loss ↔ Latent z

```
Latent z (batch, latent_dim)
    ↓
Assign to Prototype (via label or similarity)
    ↓
Pull Loss: ||z - prototype||²
    ↓
Orthonormal Constraint: ||P^T P - I||²
```

**효과**:
- Neural collapse 방지
- 직교 서브스페이스 형성
- 클라이언트별 preference 분리

### 4. 전체 Loss 통합

```python
# In _hook_on_batch_forward
vpl_loss = reconstruction_loss + vpl_kl_weight * kl_loss

if vpl_orthogonal_weight > 0.0:
    orthogonal_loss, _, _ = self._compute_clop_orthogonal_loss(z)
    vpl_loss = vpl_loss + orthogonal_loss

total_loss = vpl_loss
```

**Loss 가중치**:
- `vpl_kl_weight`: KL divergence 가중치 (default: 0.1)
- `vpl_orthogonal_weight`: Pull loss 가중치 (default: 10.0)
- `vpl_orthogonal_orthonorm_weight`: Orthonorm constraint 가중치 (default: 0.1)

---

## 코드 구조

### 파일 구조

```
federatedscope/llm/
├── trainer/
│   └── vpl_reward_choice_trainer.py      # Main trainer (통합)
├── model/
│   ├── variational_encoder.py            # Standard VPL encoder
│   └── variational_encoder_gp.py          # GP Prior encoder
├── llm_local/
│   ├── server.py                          # Server logic (prior aggregation, label assignment)
│   ├── client.py                          # Client logic (z collection, label reception)
│   └── z_visualization.py                 # t-SNE visualization
└── metric/
    ├── winrate_metrics.py                 # Win-lose evaluation (GPT API support)
    └── hhrl_metrics.py                    # HRL metrics
```

### 주요 클래스

#### 1. VPLRewardChoiceTrainer

**위치**: `federatedscope/llm/trainer/vpl_reward_choice_trainer.py`

**역할**: 
- VPL, GP Prior, Orthogonal Loss 통합
- Forward pass, Loss 계산, z 수집

**주요 메서드**:
- `__init__`: 컴포넌트 초기화
- `_extract_preference_features`: Feature 추출
- `_hook_on_batch_forward`: Forward pass 및 loss 계산
- `_compute_clop_orthogonal_loss`: Orthogonal loss 계산
- `get_client_z_distribution`: z 분포 반환 (서버 전송용)
- `update_prior_from_server`: Prior 업데이트 (서버로부터)
- `update_orthogonal_label_from_server`: Orthogonal label 업데이트

#### 2. VariationalEncoderGP

**위치**: `federatedscope/llm/model/variational_encoder_gp.py`

**역할**:
- Mixture prior를 사용한 KL divergence 계산
- Log-sum-exp trick으로 수치 안정성 확보

**주요 메서드**:
- `update_prior`: Prior 업데이트 (서버로부터)
- `kl_divergence`: Mixture prior와의 KL divergence
- `sample_prior`: Gumbel-Softmax로 prior에서 샘플링

#### 3. LLMMultiLoRAServer

**위치**: `federatedscope/llm/llm_local/server.py`

**역할**:
- z 분포 수집 및 mixture prior 계산
- Orthogonal label 계산 (k-means)
- Prior 및 label 브로드캐스트

**주요 메서드**:
- `_collect_vpl_gp_prior_distributions`: z 분포 수집
- `_compute_balanced_orthogonal_labels`: K-means 레이블링
- `broadcast_model_para`: Prior 및 label 브로드캐스트
- `_visualize_cross_client_z`: t-SNE 시각화

#### 4. LLMMultiLoRAClient

**위치**: `federatedscope/llm/llm_local/client.py`

**역할**:
- 로컬 학습 및 z 분포 계산
- 서버로 z 분포 전송
- 서버로부터 prior 및 label 수신

**주요 메서드**:
- `train`: 로컬 학습 및 z 수집
- `callback_funcs_for_vpl_gp_prior`: Prior 수신 처리
- `callback_funcs_for_vpl_orthogonal_labels`: Label 수신 처리

---

## 설정 및 하이퍼파라미터

### 전체 Config 예시

```yaml
# Basic FL settings
federate:
  mode: standalone
  client_num: 10
  sample_client_num: 5
  total_round_num: 50

# Data settings
data:
  type: 'hh-rlhf@llm'
  splits: [0.9, 0.09, 0.01]

# Model settings
model:
  type: 'google/gemma-2b@huggingface_llm'

# VPL settings
llm:
  # Basic VPL
  vpl_latent_dim: 32                    # Latent dimension
  vpl_kl_weight: 0.1                    # KL divergence weight
  vpl_feature_method: 'choice_logits'   # Feature extraction method
  vpl_use_feature_difference: True      # Use embedding difference
  vpl_use_difference_only: True        # Use only difference (removes general info) ⭐ Recommended
  vpl_use_llm_feature_extractor: True  # Use MLP feature extractor
  
  # GP Prior
  vpl_use_gp_prior: True                # Enable GP prior
  vpl_gp_temperature: 1.0               # Gumbel-Softmax temperature
  
  # Orthogonal Loss
  vpl_orthogonal_weight: 10.0           # Pull loss weight
  vpl_orthogonal_orthonorm_weight: 0.1  # Orthonorm constraint weight
  vpl_use_manual_orthogonal_labels: False  # Use k-means (not manual)
  vpl_num_prototypes: 2                 # Number of prototypes (k for k-means)
                                        # For hh-rlhf: automatically fixed to 2
  vpl_prototype_scale: 5.0              # Distance of prototypes from origin

# Trainer
trainer:
  type: vplgprewardchoicetrainer
  choices: ['A', 'B']

# Evaluation
eval:
  metrics: ['loss', 'acc', 'vpl_kl_loss', 'vpl_reconstruction_loss', 
            'vpl_orthogonal_loss']
  max_samples_for_reward: 100
```

### 하이퍼파라미터 가이드

#### VPL 기본 하이퍼파라미터

| 파라미터 | 기본값 | 설명 | 권장 범위 |
|---------|--------|------|----------|
| `vpl_latent_dim` | 32 | Latent space 차원 | 16-64 |
| `vpl_kl_weight` | 0.1 | KL divergence 가중치 | 0.01-1.0 |
| `vpl_feature_method` | 'choice_logits' | Feature 추출 방법 | 'choice_logits' or 'embedding_difference' |
| `vpl_use_feature_difference` | False | Embedding 차이 사용 | True (권장) |
| `vpl_use_difference_only` | False | Difference만 사용 (general info 제거) | True (preference-only 학습 시 권장) |
| `vpl_use_llm_feature_extractor` | True | MLP feature extractor 사용 | True (권장) |

#### GP Prior 하이퍼파라미터

| 파라미터 | 기본값 | 설명 | 권장 범위 |
|---------|--------|------|----------|
| `vpl_use_gp_prior` | False | GP prior 활성화 | True/False |
| `vpl_gp_temperature` | 1.0 | Gumbel-Softmax temperature | 0.5-2.0 |

#### Orthogonal Loss 하이퍼파라미터

| 파라미터 | 기본값 | 설명 | 권장 범위 |
|---------|--------|------|----------|
| `vpl_orthogonal_weight` | 0.0 | Pull loss 가중치 | 1.0-20.0 |
| `vpl_orthogonal_orthonorm_weight` | 0.1 | Orthonorm constraint 가중치 | 0.01-1.0 |
| `vpl_use_manual_orthogonal_labels` | False | Manual labels 사용 | False (k-means 권장) |
| `vpl_num_prototypes` | num_clients | 프로토타입 개수 | 2-10 (hh-rlhf: 2) |
| `vpl_prototype_scale` | 5.0 | Prototype의 원점으로부터 거리 | 2.0-10.0 |

### 하이퍼파라미터 튜닝 가이드

#### 1. Latent Dimension (`vpl_latent_dim`)

- **작은 값 (16-32)**: 
  - 장점: 더 강한 regularization, 빠른 학습
  - 단점: 표현력 제한
- **큰 값 (64-128)**:
  - 장점: 더 풍부한 표현
  - 단점: Overfitting 위험, 느린 학습

**권장**: 32 (균형잡힌 선택)

#### 2. KL Weight (`vpl_kl_weight`)

- **작은 값 (0.01-0.1)**:
  - 장점: 더 유연한 posterior
  - 단점: Overfitting 위험
- **큰 값 (0.5-1.0)**:
  - 장점: 강한 regularization
  - 단점: Underfitting 위험

**권장**: 0.1 (기본값)

#### 3. Orthogonal Weight (`vpl_orthogonal_weight`)

- **작은 값 (1.0-5.0)**:
  - 장점: 자연스러운 학습
  - 단점: Collapse 방지 효과 약함
- **큰 값 (10.0-20.0)**:
  - 장점: 강한 separation
  - 단점: 학습 불안정 가능

**권장**: 10.0 (CLOP paper 권장)

#### 4. GP Temperature (`vpl_gp_temperature`)

- **작은 값 (0.5-1.0)**:
  - 장점: 더 discrete한 샘플링
  - 단점: Gradient가 불안정할 수 있음
- **큰 값 (1.0-2.0)**:
  - 장점: 부드러운 gradient
  - 단점: 덜 discrete

**권장**: 1.0 (기본값)

---

## 실험 설정 예시

### Baseline (VPL only)

```yaml
llm:
  vpl_use_gp_prior: False
  vpl_orthogonal_weight: 0.0
```

### VPL-GP (without orthogonal loss)

```yaml
llm:
  vpl_use_gp_prior: True
  vpl_orthogonal_weight: 0.0
```

### VPL-GP + Orthogonal Loss

```yaml
llm:
  vpl_use_gp_prior: True
  vpl_orthogonal_weight: 10.0
  vpl_orthogonal_orthonorm_weight: 0.1
  vpl_use_manual_orthogonal_labels: False  # k-means
  vpl_num_prototypes: 2  # For hh-rlhf (automatically fixed to 2)
```

---

## 참고 자료

### 논문

1. **VPL**: [Variational Preference Learning](https://github.com/WEIRDLabUW/vpl)
2. **VMTL**: [Variational Multi-Task Learning with Gumbel-Softmax Priors](https://arxiv.org/abs/2111.05323)
3. **CLOP**: [Preventing Collapse in Contrastive Learning with Orthonormal Prototypes](https://arxiv.org/pdf/2403.18699)
4. **Personalized FL**: `personalized_FL.pdf` (프로젝트 내부 문서)

### 코드 참조

- **Trainer**: `federatedscope/llm/trainer/vpl_reward_choice_trainer.py`
- **GP Encoder**: `federatedscope/llm/model/variational_encoder_gp.py`
- **Server**: `federatedscope/llm/llm_local/server.py`
- **Client**: `federatedscope/llm/llm_local/client.py`

### 관련 문서

- `docs/VPL_DOCUMENTATION.md`: VPL 상세 설명
- `docs/GP_PRIOR_DOCUMENTATION.md`: GP Prior 상세 설명
- `docs/ORTHOGONAL_LOSS_DOCUMENTATION.md`: Orthogonal Loss 상세 설명
- `docs/PAPER_IMPLEMENTATION_COMPARISON.md`: 논문과 구현 비교

---

## 요약

이 구현은 다음 세 가지 핵심 기술을 통합합니다:

1. **VPL**: 사용자별 preference를 latent variable로 모델링
2. **GP Prior**: 다른 클라이언트들의 preference를 mixture prior로 활용
3. **Orthogonal Loss**: Neural collapse 방지 및 직교 서브스페이스 형성

이들의 조합으로 **개인화된 federated preference learning**을 달성합니다.
