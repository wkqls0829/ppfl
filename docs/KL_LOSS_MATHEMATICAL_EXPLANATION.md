# KL Loss 수학적 설명: VPL with GP Prior

## 1. 수학적 배경

### 1.1 표기법

- **Posterior (변분 인코더 출력)**: 
  $$q(z|x) = \mathcal{N}(z; \mu, \sigma^2 I)$$
  
  여기서:
  - $\mu = \mu_\phi(x)$: 평균 벡터 (batch_size, latent_dim)
  - $\sigma^2 = \exp(\log\sigma^2)$: 분산
  - $\log\sigma^2 = \log\sigma^2_\phi(x)$: 로그 분산 벡터 (batch_size, latent_dim)
  - $\phi$: 변분 인코더 파라미터

- **Mixture Prior (GP Prior)**:
  $$p_{\text{mixture}}(z) = \sum_{i=1}^{N} w_i \cdot \mathcal{N}(z; \mu_i, \sigma_i^2 I)$$
  
  여기서:
  - $N$: 다른 클라이언트 수
  - $w_i$: $i$번째 클라이언트의 가중치 (일반적으로 $w_i = 1/N$)
  - $\mu_i, \sigma_i^2$: $i$번째 클라이언트의 사전 분포 파라미터 (고정됨, 미분 불가)

- **표준 사전** (비교용):
  $$p_{\text{standard}}(z) = \mathcal{N}(z; 0, I)$$

### 1.2 KL Divergence 정의

KL divergence는 두 확률 분포 간의 "거리"를 측정합니다:

$$D_{KL}(q(z|x) || p(z)) = \mathbb{E}_{z \sim q(z|x)} \left[ \log \frac{q(z|x)}{p(z)} \right]$$

이를 전개하면:

$$D_{KL}(q(z|x) || p(z)) = \mathbb{E}_{z \sim q(z|x)} \left[ \log q(z|x) - \log p(z) \right]$$

$$= \mathbb{E}_{z \sim q(z|x)} \left[ \log q(z|x) \right] - \mathbb{E}_{z \sim q(z|x)} \left[ \log p(z) \right]$$

---

## 2. Batch Processing and Batch Size

### 2.1 Batch Size 설정

현재 실험에서 사용하는 batch size:
- **Config 파일**: `cfg/vpl-gp/hhst-ortho-50103.yaml`
- **batch_size**: `4` (dataloader 설정)

### 2.2 KL Loss의 Batch 처리

KL divergence는 **batch 전체**에 대해 계산되며, 최종적으로 **batch 평균**을 구합니다.

**입력**:
- `mu`: (batch_size, latent_dim) = (4, 32)
- `logvar`: (batch_size, latent_dim) = (4, 32)

**계산 과정**:
1. 각 샘플마다 개별적으로 `log_q(z|x)`와 `log_p_mixture(z)` 계산
2. `log_q`: (batch_size,) = (4,) 형태
3. `log_p_mixture`: (batch_size,) = (4,) 형태
4. 최종 KL loss: `kl = (log_q - log_p_mixture).mean()` → **scalar**

**수식으로 표현**:

$$D_{KL}(q || p_{\text{mixture}}) = \frac{1}{B} \sum_{b=1}^{B} \left[ \log q(z^{(b)}|x^{(b)}) - \log p_{\text{mixture}}(z^{(b)}) \right]$$

여기서:
- $B$ = batch_size = 4
- $z^{(b)}$ = $b$번째 샘플의 latent code
- 각 샘플은 독립적으로 계산됨

---

## 3. Forward Pass 수학적 유도

### 3.1 Reparameterization Trick

변분 인코더에서 $z$를 샘플링할 때, 직접 샘플링하면 미분이 불가능합니다:

$$z \sim q(z|x) = \mathcal{N}(z; \mu, \sigma^2 I)$$

**Reparameterization Trick**을 사용하면:

$$z = \mu + \epsilon \odot \sigma, \quad \epsilon \sim \mathcal{N}(0, I)$$

여기서:
- $\epsilon$: 표준 정규 분포에서 샘플링된 노이즈 (미분 불가, 랜덤)
- $\odot$: element-wise 곱셈
- $\sigma = \exp(0.5 \cdot \log\sigma^2)$

이제 $z$는 $\mu$와 $\log\sigma^2$에 대해 미분 가능합니다:

$$\frac{\partial z}{\partial \mu} = I \quad \text{(단위 행렬)}$$

$$\frac{\partial z}{\partial \log\sigma^2} = \epsilon \odot \sigma \odot 0.5 = \epsilon \odot \exp(0.5 \cdot \log\sigma^2) \odot 0.5$$

### 3.2 log q(z|x) 계산 (Batch-wise)

다변량 정규 분포의 로그 확률 밀도 함수:

$$\log \mathcal{N}(z; \mu, \sigma^2 I) = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\sum_{j=1}^{d} \log\sigma_j^2 - \frac{1}{2}\sum_{j=1}^{d} \frac{(z_j - \mu_j)^2}{\sigma_j^2}$$

여기서 $d$는 latent_dim입니다.

코드에서는:

$$\log q(z|x) = -\frac{1}{2} \sum_{j=1}^{d} \left[ \log(2\pi) + \log\sigma_j^2 + \frac{(z_j - \mu_j)^2}{\exp(\log\sigma_j^2)} \right]$$

**미분 (Gradient)**:

$$\frac{\partial \log q(z|x)}{\partial \mu_j} = \frac{z_j - \mu_j}{\sigma_j^2}$$

$$\frac{\partial \log q(z|x)}{\partial \log\sigma_j^2} = -\frac{1}{2} + \frac{1}{2} \cdot \frac{(z_j - \mu_j)^2}{\sigma_j^2}$$

벡터 형태로:

$$\frac{\partial \log q(z|x)}{\partial \mu} = \frac{z - \mu}{\sigma^2} \quad \text{(element-wise division)}$$

$$\frac{\partial \log q(z|x)}{\partial \log\sigma^2} = -\frac{1}{2} + \frac{1}{2} \cdot \frac{(z - \mu)^2}{\sigma^2}$$

### 3.3 log p_mixture(z) 계산 (Batch-wise)

Mixture prior의 로그 확률 밀도 함수:

$$\log p_{\text{mixture}}(z) = \log \left[ \sum_{i=1}^{N} w_i \cdot \mathcal{N}(z; \mu_i, \sigma_i^2 I) \right]$$

직접 계산하면 수치적 불안정성이 발생합니다. **Log-Sum-Exp Trick**을 사용:

#### Log-Sum-Exp Trick

$$\log\left(\sum_{i=1}^{N} \exp(a_i)\right) = \max_i a_i + \log\left(\sum_{i=1}^{N} \exp(a_i - \max_i a_i)\right)$$

**증명**:

$$\sum_{i=1}^{N} \exp(a_i) = \exp(\max_i a_i) \cdot \sum_{i=1}^{N} \exp(a_i - \max_i a_i)$$

양변에 로그를 취하면:

$$\log\left(\sum_{i=1}^{N} \exp(a_i)\right) = \max_i a_i + \log\left(\sum_{i=1}^{N} \exp(a_i - \max_i a_i)\right)$$

#### log p_mixture(z)에 적용

각 컴포넌트의 로그 확률:

$$a_i = \log w_i + \log \mathcal{N}(z; \mu_i, \sigma_i^2 I)$$

$$= \log w_i - \frac{d}{2}\log(2\pi) - \frac{1}{2}\sum_{j=1}^{d} \log\sigma_{i,j}^2 - \frac{1}{2}\sum_{j=1}^{d} \frac{(z_j - \mu_{i,j})^2}{\sigma_{i,j}^2}$$

그러면:

$$\log p_{\text{mixture}}(z) = \max_i a_i + \log\left(\sum_{i=1}^{N} \exp(a_i - \max_i a_i)\right)$$

**Softmax 가중치**:

$$\text{softmax}_i = \frac{\exp(a_i - \max_i a_i)}{\sum_{j=1}^{N} \exp(a_j - \max_j a_j)}$$

이 가중치는 각 컴포넌트의 기여도를 나타냅니다.

### 3.4 log p_mixture(z)의 미분

**주의**: $\mu_i, \sigma_i^2, w_i$는 **고정된 파라미터**이므로 미분 불가능합니다.
오직 $z$에 대해서만 미분 가능합니다.

**Chain Rule 적용**:

$$\frac{\partial \log p_{\text{mixture}}(z)}{\partial z_j} = \sum_{i=1}^{N} \frac{\partial a_i}{\partial z_j} \cdot \frac{\partial \log p_{\text{mixture}}}{\partial a_i}$$

$$\frac{\partial \log p_{\text{mixture}}}{\partial a_i} = \text{softmax}_i = \frac{\exp(a_i - \max_i a_i)}{\sum_{k=1}^{N} \exp(a_k - \max_k a_k)}$$

$$\frac{\partial a_i}{\partial z_j} = \frac{\partial}{\partial z_j} \left[ \log w_i + \log \mathcal{N}(z; \mu_i, \sigma_i^2 I) \right]$$

$$= \frac{\partial}{\partial z_j} \left[ -\frac{1}{2} \sum_{k=1}^{d} \frac{(z_k - \mu_{i,k})^2}{\sigma_{i,k}^2} \right]$$

$$= -\frac{(z_j - \mu_{i,j})}{\sigma_{i,j}^2}$$

따라서:

$$\frac{\partial \log p_{\text{mixture}}(z)}{\partial z_j} = -\sum_{i=1}^{N} \text{softmax}_i \cdot \frac{(z_j - \mu_{i,j})}{\sigma_{i,j}^2}$$

벡터 형태로:

$$\frac{\partial \log p_{\text{mixture}}(z)}{\partial z} = -\sum_{i=1}^{N} \text{softmax}_i \cdot \frac{z - \mu_i}{\sigma_i^2}$$

### 3.5 KL Divergence 최종 계산 (Batch Average)

**수학적 정의**:
$$D_{KL}(q(z|x) || p_{\text{mixture}}(z)) = \mathbb{E}_{z \sim q(z|x)} \left[ \log q(z|x) - \log p_{\text{mixture}}(z) \right]$$

**Monte Carlo 추정** (batch 전체에 대해):
$$D_{KL} \approx \frac{1}{B} \sum_{b=1}^{B} \left[ \log q(z^{(b)}|x^{(b)}) - \log p_{\text{mixture}}(z^{(b)}) \right]$$

여기서:
- $B$ = batch_size = 4
- $z^{(b)} = \mu^{(b)} + \epsilon^{(b)} \odot \sigma^{(b)}$: $b$번째 샘플의 latent code
- 각 샘플은 독립적으로 계산되고, 최종적으로 batch 평균을 구함

**코드 구현**:
```python
# log_q: (batch_size, 4) 형태
# log_p_mixture: (batch_size, 4) 형태
kl = (log_q - log_p_mixture).mean()  # scalar
```

**중요**: 
- 각 샘플마다 다른 $\epsilon^{(b)}$를 샘플링하므로 랜덤성 존재
- 하지만 batch 평균을 구하므로 gradient는 안정적
- batch size가 클수록 더 안정적인 gradient 추정

---

## 4. Backward Pass 수학적 유도 (Batch-wise Gradient)

### 4.1 Loss Function (Batch Average)

전체 손실 함수 (batch 평균):

$$\mathcal{L} = \frac{1}{B}\sum_{b=1}^{B} \mathcal{L}_{\text{reconstruction}}^{(b)} + \beta \cdot D_{KL}(q(z|x) || p_{\text{mixture}}(z))$$

여기서:
- $B$ = batch_size = 4
- $\beta$ = vpl_kl_weight = 100.0 (현재 실험 설정)
- $D_{KL}$는 이미 batch 평균이므로 스칼라
- $\mathcal{L}_{\text{reconstruction}}^{(b)}$: $b$번째 샘플의 reconstruction loss

### 4.2 KL Loss에 대한 Gradient (Batch-wise)

**중요**: KL loss는 이미 batch 평균이므로 스칼라입니다. Gradient는 각 샘플에 대해 계산되고 합산됩니다.

#### 4.2.1 ∂KL/∂log_q 경로 (Batch Sum)

각 샘플 $b$에 대해:
$$\frac{\partial D_{KL}}{\partial \log q(z^{(b)}|x^{(b)})} = \frac{1}{B}$$

**Batch 전체에 대한 합산**:

$$\frac{\partial D_{KL}}{\partial \log q(z|x)} = \frac{1}{B}$$

$$\frac{\partial D_{KL}}{\partial \mu_j} = \frac{\partial D_{KL}}{\partial \log q(z|x)} \cdot \frac{\partial \log q(z|x)}{\partial \mu_j}$$

$$= \frac{1}{B} \cdot \frac{z_j - \mu_j}{\sigma_j^2}$$

벡터 형태:

$$\frac{\partial D_{KL}}{\partial \mu} = \frac{1}{B} \cdot \frac{z - \mu}{\sigma^2}$$

$$\frac{\partial D_{KL}}{\partial \log\sigma_j^2} = \frac{\partial D_{KL}}{\partial \log q(z|x)} \cdot \frac{\partial \log q(z|x)}{\partial \log\sigma_j^2}$$

$$= \frac{1}{B} \cdot \left( -\frac{1}{2} + \frac{1}{2} \cdot \frac{(z_j - \mu_j)^2}{\sigma_j^2} \right)$$

벡터 형태:

$$\frac{\partial D_{KL}}{\partial \log\sigma^2} = \frac{1}{B} \cdot \left( -\frac{1}{2} + \frac{1}{2} \cdot \frac{(z - \mu)^2}{\sigma^2} \right)$$

#### 3.2.2 ∂KL/∂log_p_mixture 경로

$$\frac{\partial D_{KL}}{\partial \log p_{\text{mixture}}(z)} = -\frac{1}{B}$$

(음수인 이유: KL loss에서 `log_q - log_p_mixture`이므로)

**Chain Rule**:

$$\frac{\partial D_{KL}}{\partial z_j} = \frac{\partial D_{KL}}{\partial \log p_{\text{mixture}}(z)} \cdot \frac{\partial \log p_{\text{mixture}}(z)}{\partial z_j}$$

$$= -\frac{1}{B} \cdot \left( -\sum_{i=1}^{N} \text{softmax}_i \cdot \frac{z_j - \mu_{i,j}}{\sigma_{i,j}^2} \right)$$

$$= \frac{1}{B} \cdot \sum_{i=1}^{N} \text{softmax}_i \cdot \frac{z_j - \mu_{i,j}}{\sigma_{i,j}^2}$$

벡터 형태:

$$\frac{\partial D_{KL}}{\partial z} = \frac{1}{B} \cdot \sum_{i=1}^{N} \text{softmax}_i \cdot \frac{z - \mu_i}{\sigma_i^2}$$

**Reparameterization Trick을 통한 역전파**:

$$\frac{\partial D_{KL}}{\partial \mu_j} \leftarrow \frac{\partial D_{KL}}{\partial z_j} \cdot \frac{\partial z_j}{\partial \mu_j} = \frac{\partial D_{KL}}{\partial z_j} \cdot 1 = \frac{\partial D_{KL}}{\partial z_j}$$

$$\frac{\partial D_{KL}}{\partial \log\sigma_j^2} \leftarrow \frac{\partial D_{KL}}{\partial z_j} \cdot \frac{\partial z_j}{\partial \log\sigma_j^2} = \frac{\partial D_{KL}}{\partial z_j} \cdot \epsilon_j \cdot \sigma_j \cdot 0.5$$

#### 3.2.3 최종 Gradient 합산

두 경로에서 온 gradient를 합산:

$$\frac{\partial D_{KL}}{\partial \mu} = \frac{1}{B} \cdot \frac{z - \mu}{\sigma^2} + \frac{1}{B} \cdot \sum_{i=1}^{N} \text{softmax}_i \cdot \frac{z - \mu_i}{\sigma_i^2}$$

$$\frac{\partial D_{KL}}{\partial \log\sigma^2} = \frac{1}{B} \cdot \left( -\frac{1}{2} + \frac{1}{2} \cdot \frac{(z - \mu)^2}{\sigma^2} \right) + \frac{1}{B} \cdot \sum_{i=1}^{N} \text{softmax}_i \cdot \frac{z - \mu_i}{\sigma_i^2} \cdot \epsilon \odot \sigma \odot 0.5$$

### 3.3 Encoder 파라미터에 대한 Gradient

$$\frac{\partial D_{KL}}{\partial \phi} = \frac{\partial D_{KL}}{\partial \mu} \cdot \frac{\partial \mu}{\partial \phi} + \frac{\partial D_{KL}}{\partial \log\sigma^2} \cdot \frac{\partial \log\sigma^2}{\partial \phi}$$

여기서 $\phi$는 변분 인코더의 파라미터입니다.

**중요**: Prior 파라미터 $\mu_i, \sigma_i^2, w_i$는 미분 불가능합니다:

$$\frac{\partial D_{KL}}{\partial \mu_i} = 0 \quad \text{(고정됨)}$$

$$\frac{\partial D_{KL}}{\partial \sigma_i^2} = 0 \quad \text{(고정됨)}$$

$$\frac{\partial D_{KL}}{\partial w_i} = 0 \quad \text{(고정됨)}$$

---

## 4. 수학적 해석

### 4.1 KL Divergence의 의미

KL divergence는 두 분포 간의 "정보 손실"을 측정합니다:

$$D_{KL}(q || p) = \mathbb{E}_{z \sim q} \left[ \log \frac{q(z)}{p(z)} \right]$$

- $D_{KL}(q || p) = 0$: 두 분포가 완전히 동일
- $D_{KL}(q || p) > 0$: $q$와 $p$가 다름
- 항상 $D_{KL}(q || p) \geq 0$ (Gibbs 부등식)

### 4.2 Gradient의 의미

#### ∂KL/∂μ의 첫 번째 항:

$$\frac{1}{B} \cdot \frac{z - \mu}{\sigma^2}$$

**해석**: Posterior $q(z|x)$를 더 확산시키는 방향으로 pull (entropy 증가)

#### ∂KL/∂μ의 두 번째 항:

$$\frac{1}{B} \cdot \sum_{i=1}^{N} \text{softmax}_i \cdot \frac{z - \mu_i}{\sigma_i^2}$$

**해석**: Posterior $q(z|x)$를 mixture prior의 가중 평균 $\sum_i \text{softmax}_i \cdot \mu_i$ 방향으로 pull

**결과**: Posterior가 mixture prior에 가까워지는 방향으로 최적화됩니다.

### 4.3 Log-Sum-Exp Trick의 수치적 안정성

**문제**: 직접 계산 시

$$\log\left(\sum_{i=1}^{N} w_i \exp(a_i)\right)$$

- $a_i$가 크면: $\exp(a_i)$ → $\infty$ (overflow)
- $a_i$가 작으면: $\exp(a_i)$ → $0$ (underflow)

**해결**: Log-sum-exp trick

$$\log\left(\sum_{i=1}^{N} \exp(a_i)\right) = \max_i a_i + \log\left(\sum_{i=1}^{N} \exp(a_i - \max_i a_i)\right)$$

- $\exp(a_i - \max_i a_i) \in [0, 1]$ (범위 제한)
- 수치적으로 안정적

---

## 5. 수학적 정리

### 5.1 Gradient Flow 정리

**Theorem 1**: Reparameterization Trick을 사용하면 $z$에 대한 gradient가 $\mu, \log\sigma^2$로 흐릅니다.

**Proof**: 
$$z = \mu + \epsilon \odot \exp(0.5 \cdot \log\sigma^2), \quad \epsilon \sim \mathcal{N}(0, I)$$

$$\frac{\partial z}{\partial \mu} = I, \quad \frac{\partial z}{\partial \log\sigma^2} = \epsilon \odot \exp(0.5 \cdot \log\sigma^2) \odot 0.5$$

Chain rule에 의해:

$$\frac{\partial f(z)}{\partial \mu} = \frac{\partial f(z)}{\partial z} \cdot \frac{\partial z}{\partial \mu} = \frac{\partial f(z)}{\partial z}$$

$$\frac{\partial f(z)}{\partial \log\sigma^2} = \frac{\partial f(z)}{\partial z} \cdot \frac{\partial z}{\partial \log\sigma^2}$$

$\square$

### 5.2 Prior 파라미터 미분 불가능 정리

**Theorem 2**: Mixture prior의 파라미터 $\mu_i, \sigma_i^2, w_i$는 gradient를 받지 않습니다.

**Proof**: 
Prior 파라미터는 다른 클라이언트의 분포를 나타내며, forward pass에서 상수로 사용됩니다:

$$p_{\text{mixture}}(z) = \sum_{i=1}^{N} w_i \cdot \mathcal{N}(z; \mu_i, \sigma_i^2 I)$$

여기서 $\mu_i, \sigma_i^2, w_i$는 고정된 값입니다.

Gradient 계산 시:

$$\frac{\partial D_{KL}}{\partial \mu_i} = \frac{\partial}{\partial \mu_i} \mathbb{E}_{z \sim q} \left[ \log q(z) - \log p_{\text{mixture}}(z) \right]$$

$$= \mathbb{E}_{z \sim q} \left[ -\frac{\partial \log p_{\text{mixture}}(z)}{\partial \mu_i} \right]$$

하지만 계산 그래프에서 $\mu_i$는 leaf node가 아니므로 autograd가 gradient를 계산하지 않습니다.

$\square$

---

## 6. 수치 예제

### 예제: 단순한 경우 (latent_dim = 1, num_clients = 2)

**초기값**:
- Posterior: $\mu = 0.5, \log\sigma^2 = 0$ → $\sigma = 1$
- Prior component 1: $\mu_1 = 0, \sigma_1 = 1, w_1 = 0.5$
- Prior component 2: $\mu_2 = 2, \sigma_2 = 1, w_2 = 0.5$
- $\epsilon = 0.3$ (랜덤 샘플)

**Forward Pass**:

1. $z = \mu + \epsilon \cdot \sigma = 0.5 + 0.3 \cdot 1 = 0.8$

2. $\log q(z|x) = -\frac{1}{2}\log(2\pi) - \frac{1}{2}\log(1) - \frac{1}{2}\frac{(0.8-0.5)^2}{1} = -0.225$

3. $a_1 = \log(0.5) + \log\mathcal{N}(0.8; 0, 1) = -0.693 - 0.82 = -1.513$

   $a_2 = \log(0.5) + \log\mathcal{N}(0.8; 2, 1) = -0.693 - 1.62 = -2.313$

   $\max(a_1, a_2) = -1.513$

   $\log p_{\text{mixture}}(z) = -1.513 + \log(\exp(0) + \exp(-0.8)) = -1.513 + 0.47 = -1.043$

4. $D_{KL} = -0.225 - (-1.043) = 0.818$

**Backward Pass**:

1. $\frac{\partial D_{KL}}{\partial \log q} = 1$ (단일 샘플)

2. $\frac{\partial \log q}{\partial \mu} = \frac{0.8 - 0.5}{1} = 0.3$

3. $\frac{\partial D_{KL}}{\partial \mu} \leftarrow 1 \cdot 0.3 = 0.3$ (첫 번째 경로)

4. $\text{softmax}_1 = \frac{\exp(0)}{1 + \exp(-0.8)} = 0.69$

   $\text{softmax}_2 = 0.31$

5. $\frac{\partial \log p_{\text{mixture}}}{\partial z} = 0.69 \cdot \frac{0.8-0}{1} + 0.31 \cdot \frac{0.8-2}{1} = 0.552 - 0.372 = 0.18$

6. $\frac{\partial D_{KL}}{\partial z} = -1 \cdot 0.18 = -0.18$ (음수 부호 주의)

7. $\frac{\partial D_{KL}}{\partial \mu} \leftarrow -0.18 \cdot 1 = -0.18$ (두 번째 경로)

8. **최종**: $\frac{\partial D_{KL}}{\partial \mu} = 0.3 + (-0.18) = 0.12$

**해석**: $\mu$가 증가하는 방향으로 gradient가 흐릅니다. 이는 posterior가 두 prior component의 중간(가중 평균)으로 이동하는 것을 의미합니다.

---

## 요약

### Forward Pass 수식

1. **Reparameterization**: $z = \mu + \epsilon \odot \sigma, \epsilon \sim \mathcal{N}(0, I)$
2. **log q(z|x)**: $-\frac{1}{2}\sum_j \left[ \log(2\pi) + \log\sigma_j^2 + \frac{(z_j - \mu_j)^2}{\sigma_j^2} \right]$
3. **log p_mixture(z)**: $\max_i a_i + \log\left(\sum_i \exp(a_i - \max_i a_i)\right)$
4. **KL**: $\mathbb{E}_z[\log q(z|x) - \log p_{\text{mixture}}(z)]$

### Backward Pass 수식

1. $\frac{\partial D_{KL}}{\partial \mu} = \frac{1}{B} \cdot \left( \frac{z-\mu}{\sigma^2} + \sum_i \text{softmax}_i \cdot \frac{z-\mu_i}{\sigma_i^2} \right)$
2. $\frac{\partial D_{KL}}{\partial \log\sigma^2} = \frac{1}{B} \cdot \left( -\frac{1}{2} + \frac{(z-\mu)^2}{2\sigma^2} + \sum_i \text{softmax}_i \cdot \frac{z-\mu_i}{\sigma_i^2} \cdot \epsilon \odot \sigma \odot 0.5 \right)$
3. $\frac{\partial D_{KL}}{\partial \phi} = \frac{\partial D_{KL}}{\partial \mu} \cdot \frac{\partial \mu}{\partial \phi} + \frac{\partial D_{KL}}{\partial \log\sigma^2} \cdot \frac{\partial \log\sigma^2}{\partial \phi}$

### 핵심 수학적 통찰

1. **Reparameterization Trick**: 랜덤 샘플링을 미분 가능한 변환으로 변환
2. **Log-Sum-Exp Trick**: 수치적 안정성 보장
3. **Mixture Prior**: 여러 클라이언트의 분포를 가중 평균으로 통합
4. **KL Minimization**: Posterior를 Prior에 가깝게 만들어 클러스터링 유도
