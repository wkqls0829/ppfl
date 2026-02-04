# Appendix 내용 정리

이 문서는 논문의 Appendix에 포함되어야 할 내용을 정리합니다.

## A. Hyperparameter Details

### A.1 하이퍼파라미터 서치 결과

**Phase 1: Orthogonal Loss Parameters**
- 탐색 공간:
  - `vpl_orthogonal_weight`: [0.2, 1.0, 5.0]
  - `vpl_orthogonal_orthonorm_weight`: [0.0, 0.1, 0.5]
  - `vpl_prototype_scale`: [2.0, 5.0, 10.0]
- 최적값:
  - `vpl_orthogonal_weight`: 1.0
  - `vpl_orthogonal_orthonorm_weight`: 0.1
  - `vpl_prototype_scale`: 5.0

**Phase 2: VPL Core Parameters**
- 탐색 공간:
  - `vpl_kl_weight`: [0.02, 0.05, 0.1, 0.2]
  - `vpl_gp_temperature`: [0.5, 1.0, 2.0, 5.0]
- 최적값:
  - `vpl_kl_weight`: 0.1
  - `vpl_gp_temperature`: 1.0

**Phase 3: Learning Rate**
- 탐색 공간: `lr`: [0.00005, 0.0001, 0.0002]
- 최적값: `lr`: 0.0001

### A.2 최종 하이퍼파라미터 설정

**Selector Training (Stage 1)**
- Learning rate: $10^{-4}$ (Gemma-2B), $10^{-5}$ (Qwen-2 0.5B)
- Batch size: 8-16 (모델에 따라)
- Gradient accumulation steps: 4
- Local update steps: 30
- Total rounds: 50
- KL weight ($\beta$): 0.1
- Orthogonal loss weight ($\lambda$): 1.0
- Orthonorm weight ($\gamma$): 0.1
- Gumbel-Softmax temperature ($\tau$): 1.0
- Prototype scale ($s$): 5.0
- Latent dimension ($d$): 32
- LoRA rank ($r$): 8
- LoRA alpha ($\alpha$): 16
- LoRA dropout ($p$): 0.05

**RL Training (Stage 2)**
- Learning rate: $10^{-4}$ (Gemma-2B), $10^{-5}$ (Qwen-2 0.5B)
- Batch size: 1
- Gradient accumulation steps: 32 (Qwen-2), 4 (Gemma-2B)
- Local update steps: 30
- Total rounds: 50
- Reward coefficient: 0.1
- Max prompts for generation: 50
- Generation batch size: 3
- Max samples for reward: 30

### A.3 하이퍼파라미터 서치 실험 상세

| Phase | TID Range | 탐색 파라미터 | 고정 파라미터 |
|-------|-----------|-------------|-------------|
| Phase 1 | 52000-52006 | orthogonal_weight, orthonorm_weight, prototype_scale | kl_weight=0.1, temperature=1.0, lr=0.0001 |
| Phase 2 | 52007-52013 | kl_weight, gp_temperature | orthogonal_weight=1.0, orthonorm_weight=0.1, prototype_scale=5.0 |
| Phase 3 | 52014-52016 | lr | 모든 Phase 1-2 최적값 |

## B. Experimental Settings

### B.1 데이터셋 상세

**HH-RLHF Dataset**
- 출처: [Bai et al., 2022]
- 데이터 분할: Train (90%), Validation (9%), Test (1%)
- 클라이언트 수: $K \in \{10, 50, 100\}$
- 샘플링 비율:
  - $K=10$: 라운드당 5개 클라이언트
  - $K \in \{50, 100\}$: 라운드당 10개 클라이언트
- 데이터 타입: Pairwise preference comparisons
- Preference axes: Helpfulness, Harmlessness

### B.2 모델 상세

**Base Models**
- Qwen-2 0.5B: 0.5 billion parameters
- Gemma-2B: 2 billion parameters

**Fine-tuning Configuration**
- Method: LoRA (Low-Rank Adaptation)
- LoRA rank: $r=8$
- LoRA alpha: $\alpha=16$
- Dropout rate: $p=0.05$
- Base LLM: Frozen (selector training 시)
- Trainable: VPL components + LoRA adapters

### B.3 평가 설정

**Winrate Evaluation**
- Judge: GPT-4 (또는 gpt-4o-mini)
- 비교 대상: Fine-tuned model vs Baseline model
- 평가 샘플 수: 최대 30개 (효율성)
- 메트릭: Helpful Win-rate (%), Harmless Win-rate (%)

**Reward Model Evaluation**
- Harmlessness score: Reward model의 harmlessness 점수
- Helpfulness score: Reward model의 helpfulness 점수
- 평가 샘플 수: 최대 100개

## C. Generation Prompts

### C.1 HH-RLHF Comparison Prompt

**Selector Training용 (Binary Choice)**
```
Below is a conversation between a human and an AI assistant, followed by two responses. Pick the response that is more helpful and harmless. State your choice with a single capital letter, i.e., "A" if RESPONSE A is better, "B" if RESPONSE B is better.

### CONVERSATION:
{prompt}

### RESPONSE A: {output_A}
### RESPONSE B: {output_B}
### YOUR CHOICE:
```

**RL Generation용 (Standard Generation)**
- Prompt: 대화 이전 부분 (conversation history)
- Generation 설정:
  - `top_p`: 1.0
  - `temperature`: 0.7
  - `do_sample`: True
  - `max_new_tokens`: 512 (config에 따라)
  - `num_return_sequences`: 2 (기본값)

### C.2 GPT API Winrate Evaluation Prompt

**Winrate 비교용**
```
Below is a conversation between a human and an AI assistant, followed by two responses. Pick the response that is more helpful and harmless. State your choice with a single capital letter, i.e., "A" if RESPONSE A is better, "B" if RESPONSE B is better.

### CONVERSATION:
{prompt}

### RESPONSE A: {response_a}
### RESPONSE B: {response_b}
### YOUR CHOICE:
```

## D. Mathematical Proofs

### D.1 KL Divergence with Mixture Prior

**Theorem 1**: Mixture prior와 posterior 간의 KL divergence는 다음과 같이 계산된다:

$$\mathbb{D}_{KL}(q_i \,\|\, p_{\text{mixture}}) = \mathbb{E}_{z \sim q_i} \left[ \log q_i(z) - \log p_{\text{mixture}}(z) \right]$$

여기서:
- $q_i(z) = \mathcal{N}(z; \mu_i, \sigma_i^2 I)$: Client $i$의 posterior
- $p_{\text{mixture}}(z) = \sum_{j=1}^{|\mathcal{S}|} w_j \cdot \mathcal{N}(z; \mu_j, \sigma_j^2 I)$: Mixture prior

**Proof**: 
KL divergence의 정의에 따라:
$$\mathbb{D}_{KL}(q_i \,\|\, p_{\text{mixture}}) = \int q_i(z) \log \frac{q_i(z)}{p_{\text{mixture}}(z)} dz$$

Monte Carlo 추정:
$$\mathbb{D}_{KL} \approx \frac{1}{B} \sum_{b=1}^{B} \left[ \log q_i(z^{(b)}) - \log p_{\text{mixture}}(z^{(b)}) \right]$$

여기서 $z^{(b)} \sim q_i$는 reparameterization trick으로 샘플링:
$$z^{(b)} = \mu_i + \sigma_i \odot \epsilon^{(b)}, \quad \epsilon^{(b)} \sim \mathcal{N}(0, I)$$

$\square$

### D.2 Log-Sum-Exp Trick for Numerical Stability

**Theorem 2**: Log-sum-exp trick은 수치적 안정성을 보장한다:

$$\log\left(\sum_{i=1}^{N} \exp(a_i)\right) = \max_i a_i + \log\left(\sum_{i=1}^{N} \exp(a_i - \max_i a_i)\right)$$

**Proof**:
$$\sum_{i=1}^{N} \exp(a_i) = \exp(\max_i a_i) \cdot \sum_{i=1}^{N} \exp(a_i - \max_i a_i)$$

양변에 로그를 취하면:
$$\log\left(\sum_{i=1}^{N} \exp(a_i)\right) = \max_i a_i + \log\left(\sum_{i=1}^{N} \exp(a_i - \max_i a_i)\right)$$

이때 $\exp(a_i - \max_i a_i) \in [0, 1]$이므로 수치적으로 안정적이다.

$\square$

### D.3 Reparameterization Trick Gradient Flow

**Theorem 3**: Reparameterization trick을 사용하면 $z$에 대한 gradient가 $\mu, \log\sigma^2$로 흐른다.

**Proof**:
$$z = \mu + \epsilon \odot \exp(0.5 \cdot \log\sigma^2), \quad \epsilon \sim \mathcal{N}(0, I)$$

Chain rule에 의해:
$$\frac{\partial f(z)}{\partial \mu} = \frac{\partial f(z)}{\partial z} \cdot \frac{\partial z}{\partial \mu} = \frac{\partial f(z)}{\partial z}$$

$$\frac{\partial f(z)}{\partial \log\sigma^2} = \frac{\partial f(z)}{\partial z} \cdot \frac{\partial z}{\partial \log\sigma^2} = \frac{\partial f(z)}{\partial z} \cdot \epsilon \odot \exp(0.5 \cdot \log\sigma^2) \odot 0.5$$

$\square$

### D.4 Orthogonal Loss의 수학적 정의

**Pull Loss**:
$$\mathcal{L}_{\text{pull}}(z) = \|z - \mathbf{p}_{y_i^*}\|_2^2$$

여기서 $\mathbf{p}_{y_i^*}$는 서버가 할당한 orthogonal label $y_i^*$에 해당하는 prototype이다.

**Orthonormality Constraint**:
$$\mathcal{L}_{\text{orthonorm}} = \|\mathbf{P}^T \mathbf{P} - \mathbf{I}\|_F^2$$

여기서 $\mathbf{P} = [\mathbf{p}_0, \mathbf{p}_1]^T$는 prototype 행렬이고, $\|\cdot\|_F$는 Frobenius norm이다.

**Total Orthogonal Loss**:
$$\mathcal{L}_{\text{orthogonal}}(z) = \lambda \cdot \mathcal{L}_{\text{pull}}(z) + \gamma \cdot \mathcal{L}_{\text{orthonorm}}$$

여기서 $\lambda = 1.0$ (orthogonal_weight), $\gamma = 0.1$ (orthonorm_weight)이다.

### D.5 ELBO의 유도

**Evidence Lower Bound (ELBO)**:
$$\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\phi(z \mid \mathcal{D}_i)} \left[ \log p_\theta(\mathcal{D}_i \mid z) \right] - \beta \,\mathbb{D}_{KL}(q_\phi(z \mid \mathcal{D}_i) \,\|\, p(z))$$

**유도**:
$$\log p_\theta(\mathcal{D}_i) = \log \int p_\theta(\mathcal{D}_i \mid z) p(z) dz$$

Jensen's inequality를 적용:
$$\log p_\theta(\mathcal{D}_i) \geq \mathbb{E}_{q_\phi(z \mid \mathcal{D}_i)} \left[ \log \frac{p_\theta(\mathcal{D}_i \mid z) p(z)}{q_\phi(z \mid \mathcal{D}_i)} \right]$$

전개하면:
$$\log p_\theta(\mathcal{D}_i) \geq \mathbb{E}_{q_\phi(z \mid \mathcal{D}_i)} \left[ \log p_\theta(\mathcal{D}_i \mid z) \right] - \mathbb{D}_{KL}(q_\phi(z \mid \mathcal{D}_i) \,\|\, p(z))$$

$\beta$를 추가하여 regularization strength를 조절:
$$\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\phi(z \mid \mathcal{D}_i)} \left[ \log p_\theta(\mathcal{D}_i \mid z) \right] - \beta \,\mathbb{D}_{KL}(q_\phi(z \mid \mathcal{D}_i) \,\|\, p(z))$$

$\square$

## E. Additional Experimental Details

### E.1 Client Average Z Dictionary

**저장 시점**: Selector training 완료 후 checkpoint 저장 시
**저장 내용**: 각 클라이언트의 평균 latent vector $z_i$
**사용 목적**: 
- RL training 시 client-specific conditional generation
- Client-specific conditional selection
- t-SNE 시각화

**로드 시점**:
- `use_variational_generation=True` 또는 `use_variational_selection=True`일 때
- `load_pairwise_data()` 호출 시
- `load_selector_preference_data()` 호출 시

### E.2 Dual Selection Process

**프로세스**:
1. 전체 response pair에 대해 harmlessness selection 수행 (client 1의 z 사용)
2. 전체 response pair에 대해 helpfulness selection 수행 (client 2의 z 사용)
3. Conflicting selection 통계 계산 (같은 pair에 대해 다른 선택을 하는 경우)

**결과**:
- 각 response pair에 대해 2개의 선택 결과 생성
- 총 데이터 양: 원본 response pair 수 × 2

### E.3 t-SNE Visualization

**시각화 시점**:
- Generation 단계 (RL 시작 전)
- Training 중 (매 5라운드마다 또는 마지막 라운드)

**저장 위치**:
- `exp/{expname}/sub_exp_{timestamp}/cross_client_z_tsne_generation.png`
- `exp/{expname}/sub_exp_{timestamp}/cross_client_z_tsne_round_{round_num}.png`

**내용**:
- Client별 z 분포 시각화
- Orthogonal label에 따른 색상 구분
- Cross-client knowledge sharing 효과 확인

## F. Implementation Details

### F.1 Feature Difference Extraction

**방법**: $(prompt + chosen) - (prompt + rejected)$
- Prompt와 chosen response의 embedding
- Prompt와 rejected response의 embedding
- 두 embedding의 차이를 계산하여 preference signal 추출

**목적**: Preference-specific 정보만 추출하고 일반적인 response 특성은 억제

### F.2 Server-Side Label Assignment

**방법**: Balanced k-means clustering
- 클라이언트들의 z 평균값 $\{\bar{\mu}_i\}_{i \in \mathcal{S}^t}$ 수집
- $k=2$로 balanced k-means 수행
- 각 클라이언트에 orthogonal label $y_i^* \in \{0, 1\}$ 할당
- 클라이언트들에게 브로드캐스트

**목적**: Balanced grouping 및 mode collapse 방지

### F.3 Prototype Initialization and Maintenance

**초기화**:
1. 랜덤 초기화: $\mathbf{P} \sim \mathcal{N}(0, 2.0^2)$
2. QR decomposition으로 orthonormalization
3. Prototype scale $s=5.0$으로 스케일링

**유지**:
- 매 forward pass마다 QR decomposition으로 재정규화
- Orthonormality constraint loss로 유지

## G. Computational Resources

### G.1 GPU 사용량

**Selector Training**:
- GPU: Single GPU (A6000, RTX6000ADA)
- Memory: ~20GB (Gemma-2B), ~10GB (Qwen-2 0.5B)
- Training time: ~2 days (50 rounds)

**RL Training**:
- GPU: Single GPU (A6000, RTX6000ADA)
- Memory: ~24GB (Gemma-2B), ~12GB (Qwen-2 0.5B)
- Training time: ~3 days (50 rounds)

### G.2 데이터 저장

**Checkpoint 위치**:
- Selector: `/hdd/hdd3/kjb/checkpoints/hhrl_choice_{model}_{method}_t{tid}.ckpt`
- RL: `/hdd/hdd3/kjb/checkpoints/hhrl_rlhf_{model}_choice_{method}_t{tid}.ckpt`

**로그 위치**:
- `outputs/{tid}.log`

**실험 결과 위치**:
- `exp/{expname}/sub_exp_{timestamp}/`
