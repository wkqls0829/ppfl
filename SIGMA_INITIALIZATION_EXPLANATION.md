# Sigma 초기화 설명

## 1. z 샘플링 시 초기 sigma 값

### Variational Encoder의 logvar 초기화

```python
# federatedscope/llm/model/variational_encoder.py
self.fc_logvar = nn.Linear(prev_dim, latent_dim)
```

- `nn.Linear`의 기본 초기화:
  - **weight**: Kaiming uniform 초기화
    - 범위: `[-sqrt(k), sqrt(k)]` where `k = 1/in_features`
    - `prev_dim = 128` (마지막 hidden layer)이면 `k = 1/128 ≈ 0.088`
    - 따라서 weight 범위: `[-0.088, 0.088]` 정도
  - **bias**: 0으로 초기화

### 초기 logvar 값

```python
# encode 메서드
h = self.encoder(x)  # (batch, 128)
logvar = self.fc_logvar(h)  # (batch, 32)
```

- 입력 `h`가 작은 값 (예: 0.1 정도)이면:
  - `logvar ≈ weight * h + bias ≈ 0.088 * 0.1 + 0 ≈ 0.009`
  - 따라서 초기 `logvar`는 **0에 가까운 작은 값**

### 초기 sigma 값

```python
# reparameterize 메서드
std = torch.exp(0.5 * logvar)  # sigma = exp(0.5 * logvar)
z = mu + eps * std
```

- `logvar ≈ 0`이면:
  - `sigma = exp(0.5 * 0) = exp(0) = 1.0`
- 따라서 **초기 sigma ≈ 1.0** (standard normal과 유사)

### 실제 학습 중 변화

- 학습이 진행되면서 `logvar`는 학습됩니다
- KL loss가 `logvar`를 조절합니다:
  - `KL = -0.5 * (1 + logvar - mu^2 - exp(logvar))`
  - `logvar`가 작으면 KL loss가 커지므로, 적절한 값으로 조절됩니다

## 2. p_mixture 초기화 시 sigma 값

### 첫 라운드 (Round 0)

```python
# federatedscope/llm/model/variational_encoder_gp.py
if self.prior_mus is None:
    # Fallback to standard normal
    return torch.randn(batch_size, self.latent_dim, device=device)
```

- `prior_mus = None`이므로 **standard normal 사용**
- **sigma = 1.0** (고정)

### 이후 라운드 (Round ≥ 1)

```python
# 서버에서 클라이언트들의 z 분포 수집
client_mus = [...]  # (num_clients, latent_dim)
client_logvars = [...]  # (num_clients, latent_dim)

# VariationalEncoderGP.update_prior()
self.prior_mus = client_mus
self.prior_logvars = client_logvars
```

- 각 클라이언트의 `client_z_logvar`를 수집
- `client_z_logvar`는 클라이언트가 학습한 `logvar`의 평균값

### p_mixture의 sigma 계산

```python
# kl_divergence 메서드
for i in range(num_components):
    mu_i = self.prior_mus[i]  # (latent_dim,)
    logvar_i = self.prior_logvars[i]  # (latent_dim,)
    weight_i = self.prior_weights[i]
    
    # 각 component의 sigma
    sigma_i = torch.exp(0.5 * logvar_i)  # (latent_dim,)
```

- 각 mixture component의 sigma는 **클라이언트가 학습한 logvar에 따라 결정**
- 초기에는 클라이언트의 초기 logvar (≈ 0)이므로 `sigma_i ≈ 1.0`
- 학습이 진행되면 클라이언트별로 다른 sigma 값을 가짐

## 3. 요약

### z 샘플링 (q(z|x))

| 단계 | 값 | 설명 |
|------|-----|------|
| 초기 logvar | ≈ 0 | `fc_logvar`의 초기화로 인해 0에 가까움 |
| 초기 sigma | ≈ 1.0 | `exp(0.5 * 0) = 1.0` |
| 학습 후 sigma | 학습됨 | KL loss에 의해 조절됨 |

### p_mixture (p(z))

| 라운드 | sigma | 설명 |
|--------|-------|------|
| Round 0 | 1.0 | Standard normal (고정) |
| Round ≥ 1 | 학습됨 | 클라이언트들의 `client_z_logvar` 평균 |

## 4. 권장 사항

### logvar 초기화 개선 (선택사항)

현재는 PyTorch 기본 초기화를 사용하지만, 명시적으로 초기화할 수 있습니다:

```python
# 더 큰 초기 logvar (더 큰 sigma)
nn.init.constant_(self.fc_logvar.bias, -1.0)  # logvar ≈ -1.0 → sigma ≈ 0.6
# 또는
nn.init.constant_(self.fc_logvar.bias, 0.0)   # logvar ≈ 0.0 → sigma ≈ 1.0 (현재)
# 또는
nn.init.constant_(self.fc_logvar.bias, 1.0)   # logvar ≈ 1.0 → sigma ≈ 1.65
```

### p_mixture 초기화 개선 (선택사항)

첫 라운드에서도 더 큰 sigma를 사용할 수 있습니다:

```python
if self.prior_mus is None:
    # Use larger initial sigma
    return torch.randn(batch_size, self.latent_dim, device=device) * 2.0  # sigma = 2.0
```

하지만 현재 구현은 표준적인 VAE 초기화를 따르고 있으므로, 특별한 이유가 없다면 변경하지 않는 것이 좋습니다.
