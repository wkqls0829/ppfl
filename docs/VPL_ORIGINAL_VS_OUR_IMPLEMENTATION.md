# 원본 VPL vs 우리 구현 비교

이 문서는 [원본 VPL 저장소](https://github.com/WEIRDLabUW/vpl)와 우리 구현의 차이점을 상세히 설명합니다.

## 개요

### 원본 VPL (WEIRDLabUW/vpl)
- **목적**: Robotics/RL 환경에서 human feedback을 통한 personalized preference learning
- **프레임워크**: JAX (JAXRL 사용)
- **환경**: MuJoCo, D4RL (robotics tasks)
- **구조**: Reward learning과 Policy learning 분리

### 우리 구현
- **목적**: Federated Learning 환경에서 LLM 기반 preference learning
- **프레임워크**: PyTorch (FederatedScope)
- **환경**: LLM (Language Models, e.g., Gemma-2B)
- **구조**: Binary choice selector + RLHF 통합

---

## 주요 차이점

### 1. 프레임워크 및 환경

| 항목 | 원본 VPL | 우리 구현 |
|------|----------|-----------|
| **프레임워크** | JAX | PyTorch |
| **환경** | Robotics (MuJoCo, D4RL) | LLM (Language Models) |
| **데이터 타입** | Continuous states/actions | Discrete tokens (text) |
| **모델** | Reward models, IQL policies | Transformer-based LLMs |
| **학습 설정** | Single-agent RL | Federated Learning |

### 2. 아키텍처

#### 원본 VPL
```
Preference Dataset → Reward Model (VPL) → Policy (IQL)
```
- **Reward Learning**: Preference pairs로부터 reward model 학습
- **Policy Learning**: 학습된 reward model을 사용하여 policy 학습
- **분리된 학습**: Reward와 Policy가 독립적으로 학습됨

#### 우리 구현
```
Binary Choice Data → VPL Selector → Conditional RLHF
```
- **Selector Training**: Binary choice (A/B) 데이터로 selector 학습
- **Conditional Generation**: 추론된 `z`를 조건으로 RLHF 데이터 생성
- **통합 학습**: Selector와 RLHF가 연계되어 학습됨

### 3. Feature Extraction 방법

#### 원본 VPL
원본 VPL은 **logits-based feature extraction**을 사용합니다:
- Choice token 위치의 logits를 직접 사용
- `[chosen_logits, rejected_logits]` 형태로 concatenation
- 추가적인 embedding difference 계산 없음

**원본 코드 예시** (추정):
```python
# Choice positions에서 logits 추출
chosen_logits = logits[chosen_positions]  # (batch, vocab_size)
rejected_logits = logits[rejected_positions]  # (batch, vocab_size)
features = torch.cat([chosen_logits, rejected_logits], dim=-1)
```

#### 우리 구현
우리는 **3가지 feature extraction 방법**을 지원합니다:

1. **Choice Logits** (원본 VPL과 유사)
   ```python
   # Choice token 위치의 logits 사용
   chosen_logits = logits[..., choices].mean(dim=-2)
   rejected_logits = logits[..., choices].mean(dim=-2)
   features = torch.cat([chosen_logits, rejected_logits], dim=-1)
   ```

2. **Embedding Difference (Full)**
   ```python
   # Hidden states에서 embedding 추출
   chosen_emb = hidden_states[chosen_positions].mean(dim=0)
   rejected_emb = hidden_states[rejected_positions].mean(dim=0)
   feature_diff = chosen_emb - rejected_emb
   features = torch.cat([chosen_emb, rejected_emb, feature_diff], dim=0)
   ```

3. **Embedding Difference (Difference Only)** ⭐ NEW
   ```python
   # Difference만 사용 (general information 제거)
   feature_diff = chosen_emb - rejected_emb
   features = feature_diff  # Only difference
   ```

**차이점**:
- 원본: Logits만 사용
- 우리: Embedding difference 옵션 추가 (preference 정보만 추출)

### 4. Prior Distribution

#### 원본 VPL
- **표준 정규분포**: `p(z) = N(0, I)`
- 모든 사용자가 동일한 prior 사용
- Federated learning 고려 없음

#### 우리 구현
- **표준 VPL**: `p(z) = N(0, I)` (원본과 동일)
- **VPL-GP**: `p_mixture(z) = Σ_i w_i * N(z; μ_i, σ_i²)` ⭐ NEW
  - 다른 클라이언트들의 z-distribution을 mixture prior로 사용
  - Gumbel-Softmax sampling으로 differentiable
  - Federated learning에서 클라이언트 간 지식 공유

### 5. Loss Function

#### 원본 VPL
```python
ELBO = E_q(z|x)[log p(y|z,x)] - KL(q(z|x) || N(0, I))
```

#### 우리 구현
```python
# 표준 VPL
ELBO = E_q(z|x)[log p(y|z,x)] - KL(q(z|x) || N(0, I))

# VPL-GP
ELBO = E_q(z|x)[log p(y|z,x)] - KL(q(z|x) || p_mixture(z))

# Orthogonal Loss 추가 (CLOP-based) ⭐ NEW
Total Loss = ELBO + λ_ortho * L_orthogonal
```

**Orthogonal Loss (CLOP)**:
- Pull loss: 클라이언트의 z를 해당 prototype에 가깝게
- Orthonormal constraint: Prototypes 간 orthogonality 유지
- Neural collapse 현상 활용

### 6. Federated Learning 통합

#### 원본 VPL
- **Single-agent**: 단일 사용자/환경에서 학습
- **Centralized**: 모든 데이터가 중앙에 집중
- **No communication**: 클라이언트 간 통신 없음

#### 우리 구현
- **Multi-client**: 여러 클라이언트가 분산 학습
- **Federated**: 데이터가 클라이언트별로 분산
- **Server-Client Communication**: ⭐ NEW
  - 서버가 z-distribution 수집 및 집계
  - Mixture prior 계산 및 브로드캐스트
  - Orthogonal label 계산 및 할당

### 7. Latent Conditioning 방법

#### 원본 VPL
- **Reward Model Conditioning**: `r(s, a | z) = f(s, a, z)`
- Latent `z`를 reward model의 입력으로 사용
- Policy는 reward model을 통해 간접적으로 `z`의 영향을 받음

#### 우리 구현
- **Logit Adjustment**: `logits_conditioned = logits + projection(z)` ⭐
- Latent `z`를 logits에 직접 추가 (bias 형태)
- Selector가 `z`에 직접 조건화됨

**차이점**:
- 원본: Reward model에 `z` 주입
- 우리: Logits에 `z` projection 추가

### 8. 데이터 생성 및 사용

#### 원본 VPL
```python
# Preference dataset 생성
python -m pref_learn.create_dataset \
    --num_query=<num> \
    --env=<env_name> \
    --query_len=<length>

# Reward model 학습
python pref_learn/train.py \
    --env=<env_name> \
    --dataset_path=<path>

# Policy 학습
python experiments/run_iql.py \
    --env_name=<env_name> \
    --ckpt=<reward_model_checkpoint>
```

#### 우리 구현
```python
# Binary choice selector 학습 (HHST)
python federatedscope/main.py --cfg cfg/vpl-gp/hhst-*.yaml

# Conditional RLHF (HRL)
python federatedscope/llm/rlhf/main.py \
    --selector-cfg-file cfg/vpl-gp/hhst-*.yaml \
    --cfg cfg/vpl-gp/hrl.yaml
```

### 9. Active Learning

#### 원본 VPL
- **Information Gain Sampling**: 불확실성이 높은 샘플을 선호
- `experiments/eval.py`에서 active learning 지원
- Query strategy: `sampling_method="information_gain"`

#### 우리 구현
- **Active Learning 미구현**: 현재는 passive learning만 지원
- 향후 구현 가능 (variational encoder의 uncertainty 활용)

### 10. 코드 구조

#### 원본 VPL
```
vpl/
├── pref_learn/          # Preference learning 모듈
│   ├── create_dataset.py
│   └── train.py
├── experiments/         # Policy 학습 및 평가
│   ├── run_iql.py
│   └── eval.py
├── jaxrl_m/            # IQL 구현 (JAX)
└── utils/
```

#### 우리 구현
```
federatedscope/llm/
├── trainer/
│   └── vpl_reward_choice_trainer.py  # VPL selector trainer
├── model/
│   ├── variational_encoder.py        # 표준 VPL encoder
│   └── variational_encoder_gp.py    # VPL-GP encoder ⭐
├── llm_local/
│   ├── server.py                     # Federated server (z-distribution 수집)
│   └── client.py                     # Federated client
└── rlhf/
    ├── standalone_training.py        # RLHF with VPL
    └── variational_selector.py      # Variational selection
```

---

## 핵심 확장 사항

### 1. VPL-GP (Gumbel-Softmax Prior) ⭐
- **목적**: Federated learning에서 클라이언트 간 지식 공유
- **구현**: `VariationalEncoderGP` 클래스
- **Prior**: Mixture of client z-distributions
- **Sampling**: Gumbel-Softmax relaxation

### 2. Orthogonal Loss (CLOP-based) ⭐
- **목적**: Preference space에서 클라이언트 그룹 분리
- **구현**: `_compute_clop_orthogonal_loss` 메서드
- **Components**: Pull loss + Orthonormal constraint
- **Labeling**: Server-side k-means 또는 manual

### 3. Embedding Difference Feature Extraction ⭐
- **목적**: General information 제거, preference 정보만 추출
- **구현**: `_extract_embedding_difference` 메서드
- **Options**: Full (chosen+rejected+difference) 또는 Difference-only

### 4. Federated Learning 통합 ⭐
- **Server**: z-distribution 수집, mixture prior 계산, orthogonal label 할당
- **Client**: z-distribution 전송, prior 업데이트, orthogonal loss 적용
- **Communication**: Round-based aggregation

---

## 수학적 차이점

### ELBO Formulation

#### 원본 VPL
```
ELBO = E_{q_φ(z|x)}[log p_θ(y|z,x)] - KL(q_φ(z|x) || p(z))
     = E_{q_φ(z|x)}[log p_θ(y|z,x)] - KL(q_φ(z|x) || N(0, I))
```

#### 우리 VPL-GP
```
ELBO = E_{q_φ(z|x)}[log p_θ(y|z,x)] - KL(q_φ(z|x) || p_mixture(z))
     = E_{q_φ(z|x)}[log p_θ(y|z,x)] - KL(q_φ(z|x) || Σ_i w_i * N(μ_i, σ_i²))
```

**차이점**:
- 원본: 고정된 표준 정규분포 prior
- 우리: 학습 가능한 mixture prior (다른 클라이언트들의 분포)

### Total Loss

#### 원본 VPL
```
L_total = -ELBO
```

#### 우리 VPL-GP with Orthogonal Loss
```
L_total = -ELBO + λ_ortho * L_orthogonal
         = -ELBO + λ_ortho * (L_pull + λ_orthonorm * L_orthonormal)
```

---

## 호환성 및 참고

### 원본 VPL과의 호환성
- **핵심 아이디어**: 동일 (variational inference over user latents)
- **ELBO 구조**: 동일 (reconstruction + KL)
- **Feature extraction**: 선택적으로 원본 방식 지원 (`vpl_feature_method='choice_logits'`)
- **Prior**: 기본적으로 동일 (`vpl_use_gp_prior=False`)

### 우리만의 확장
1. **Federated Learning**: 분산 환경 지원
2. **VPL-GP**: Mixture prior로 클라이언트 간 지식 공유
3. **Orthogonal Loss**: Preference space에서 클라이언트 그룹 분리
4. **Embedding Difference**: General information 제거
5. **LLM 환경**: Text-based preference learning

---

## 참고 자료

- **원본 VPL 저장소**: https://github.com/WEIRDLabUW/vpl
- **원본 VPL 논문**: "Personalizing Reinforcement Learning from Human Feedback with Variational Preference Learning"
- **우리 구현 문서**:
  - [VPL Documentation](VPL_DOCUMENTATION.md)
  - [GP Prior Documentation](GP_PRIOR_DOCUMENTATION.md)
  - [Orthogonal Loss Documentation](ORTHOGONAL_LOSS_DOCUMENTATION.md)

---

## 요약

| 특징 | 원본 VPL | 우리 구현 |
|------|----------|-----------|
| **프레임워크** | JAX | PyTorch |
| **환경** | Robotics | LLM |
| **Prior** | N(0, I) | N(0, I) 또는 Mixture |
| **Feature Extraction** | Logits only | Logits + Embedding difference |
| **Loss** | ELBO | ELBO + Orthogonal Loss |
| **Learning Setting** | Single-agent | Federated |
| **Conditioning** | Reward model | Logit adjustment |

우리 구현은 원본 VPL의 핵심 아이디어를 유지하면서, **Federated Learning 환경**과 **LLM 기반 preference learning**에 맞게 확장하고, **VPL-GP**와 **Orthogonal Loss**를 추가하여 더 강력한 개인화 학습을 지원합니다.
