# FederatedScope RLHF with VPL 및 VPL-GP 프로젝트 컨텍스트

## 프로젝트 개요

이 프로젝트는 **Variational Preference Learning (VPL)** 및 **Variational Preference Learning with Gumbel Softmax Prior (VPL-GP)**을 활용한 **Federated Reinforcement Learning from Human Feedback (RLHF)** 시스템을 구현하고 있습니다.

### 핵심 목표
1. **클라이언트별 personalized preference learning**: 각 클라이언트마다 latent vector `z`를 추론하여 개인화된 preference 학습
2. **Conditional DPO training**: 클라이언트별 `z`를 조건부로 사용하는 Direct Preference Optimization (DPO) 학습
3. **Conditional data generation**: 각 클라이언트의 `z`를 embedding에 주입하여 conditional data generation
4. **Personalized test evaluation**: 클라이언트별 test set으로 개인화된 성능 평가
5. **VPL-GP: Federated Prior Learning**: 다른 클라이언트들의 z-distribution을 mixture prior로 사용하여 federated learning에서 클라이언트 간 지식 공유 ⭐ NEW

## 현재 작업 내용

### 1. Variational Preference Learning (VPL) 통합
- **Selector model**: VPLRewardChoiceTrainer를 사용하여 클라이언트별 latent `z` 추론
- **VPL 컴포넌트**:
  - `variational_encoder`: preference features에서 latent `z` 추론
  - `latent_projection`: `z`를 choice logit adjustment로 변환
  - `z_to_embedding`: `z`를 model embedding dimension으로 projection (conditional generation용)

### 2. Variational Preference Learning with Gumbel Softmax Prior (VPL-GP) ⭐ NEW

#### 핵심 개념
- **Gumbel Softmax Prior**: 표준 정규분포 대신 다른 클라이언트들의 z-distribution을 mixture prior로 사용
- **Federated Prior Update**: 서버에서 클라이언트들의 z-distribution (mu, logvar)을 수집하고 집계하여 브로드캐스트
- **KL Divergence with Mixture**: `KL(q(z|x) || p_mixture(z))` where `p_mixture(z) = Σ_i w_i * N(z; μ_i, σ_i²)`

#### 구현 세부사항
- **VariationalEncoderGP**: 
  - `update_prior()`: 다른 클라이언트들의 z-distribution으로 prior 업데이트
  - `kl_divergence()`: Mixture prior와의 KL divergence 계산 (log-sum-exp trick 사용)
  - `sample_prior()`: Gumbel Softmax를 사용한 differentiable prior 샘플링
  
- **VPLGPRewardChoiceTrainer**:
  - VPLRewardChoiceTrainer를 상속하여 모든 VPL 기능 사용
  - `get_client_z_distribution()`: 클라이언트의 z-distribution (mu, logvar) 반환
  - `update_prior_from_server()`: 서버로부터 받은 prior 업데이트

#### Server/Client 통신
- **Server**: 
  - `_perform_federated_aggregation()`에서 클라이언트들로부터 `client_z_mu`, `client_z_logvar` 수집
  - 수집된 z-distribution들을 `self.vpl_gp_prior_mus`, `self.vpl_gp_prior_logvars`, `self.vpl_gp_prior_weights`에 저장
  - `broadcast_model_para()`에서 클라이언트들에게 prior 분포 브로드캐스트
  
- **Client**:
  - `callback_funcs_for_model_para()`에서 `get_client_z_distribution()`으로 z-distribution을 서버로 전송
  - 서버로부터 받은 prior를 `update_prior_from_server()`로 전달하여 trainer의 prior 업데이트

### 3. Conditional Data Generation
- 각 클라이언트별로 생성된 training data에 해당 클라이언트의 `z` 사용
- `_generate_pairwise_data`에서 `client_z`를 받아 embedding에 주입
- `_generate_with_z_embedding`: `z_embedding`을 input embeddings에 더하여 conditional generation 수행

### 4. Conditional DPO Training
- `VPLRewardTrainer` 사용: 각 배치에서 preference features를 추출하고 `z`를 추론하여 conditional DPO loss 계산
- `VPLGPRewardChoiceTrainer` 사용: VPL-GP의 경우 mixture prior와의 KL divergence 계산
- Config에서 `trainer.type: vplrewardtrainer` 또는 `vplgprewardchoicetrainer`로 설정 시 자동 사용

### 5. Personalized Test Performance
- `enable_personalized_test: true` 설정 시:
  - 초기 데이터를 클라이언트별로 분할 (harmlessness/helpfulness 그룹별)
  - 각 클라이언트의 test set으로 평가
  - Harmlessness 클라이언트 그룹과 Helpfulness 클라이언트 그룹별로 평균 계산

## 주요 파일 구조

### 핵심 파일
```
federatedscope/llm/trainer/
├── reward_trainer.py                # DPORewardTrainer (기본 DPO trainer)
├── vpl_reward_trainer.py            # VPLRewardTrainer (VPL DPO trainer)
├── vpl_reward_choice_trainer.py     # VPLRewardChoiceTrainer (selector용 VPL trainer)
├── vpl_gp_reward_choice_trainer.py  # VPLGPRewardChoiceTrainer (VPL-GP trainer) ⭐ NEW
└── reward_choice_trainer.py         # RewardChoiceTrainer (기본 selector trainer)

federatedscope/llm/model/
├── variational_encoder.py           # VariationalEncoder 클래스
└── variational_encoder_gp.py        # VariationalEncoderGP 클래스 (Gumbel Softmax prior) ⭐ NEW

federatedscope/llm/llm_local/
├── server.py                        # Server 구현 (z-distribution 수집 및 브로드캐스트) ⭐ MODIFIED
└── client.py                        # Client 구현 (z-distribution 전송 및 prior 업데이트) ⭐ MODIFIED

federatedscope/core/auxiliaries/
└── trainer_builder.py               # Trainer 빌더 (vplgprewardchoicetrainer 등록) ⭐ MODIFIED

cfg/
├── vpl/
│   ├── hhst.yaml                    # VPL binary selector 설정
│   └── hrl.yaml                     # VPL RL 설정
├── vpl-gp/                          # ⭐ NEW
│   ├── hhst.yaml                    # VPL-GP binary selector 설정
│   ├── hrl.yaml                     # VPL-GP RL 설정
│   └── test_hrl_selector.yaml       # VPL-GP RL용 selector checkpoint 설정
└── fedbiscuit/
    └── hhst.yaml                    # FedBiscuit HHST 설정

scripts/
├── vpl/
│   ├── hhst.sh                      # VPL HHST 실행 스크립트
│   └── hrl.sh                       # VPL HRL 실행 스크립트
├── vpl-gp/                          # ⭐ NEW
│   ├── hhst.sh                      # VPL-GP HHST 실행 스크립트
│   └── hrl.sh                       # VPL-GP HRL 실행 스크립트
└── fedbiscuit/
    ├── hhst_hpsearch.sh             # 하이퍼파라미터 서치 (GPU 0,1,4 사용)
    └── hhst_hpsearch_retry_failed.sh # 실패한 작업 재실행
```

## 최근 수정 사항 (중요!)

### 1. VPL-GP 알고리즘 구현 ⭐ NEW

#### `federatedscope/llm/model/variational_encoder_gp.py` (신규 파일)
- **VariationalEncoderGP 클래스**: Gumbel Softmax prior를 사용하는 variational encoder
- **주요 메서드**:
  - `update_prior()`: 다른 클라이언트들의 z-distribution으로 prior 업데이트
  - `kl_divergence()`: Mixture prior와의 KL divergence 계산
  - `sample_prior()`: Gumbel Softmax를 사용한 prior 샘플링
  - `gumbel_softmax_sample()`: Gumbel Softmax 샘플링 헬퍼 함수

#### `federatedscope/llm/trainer/vpl_gp_reward_choice_trainer.py` (신규 파일)
- **VPLGPRewardChoiceTrainer 클래스**: VPLRewardChoiceTrainer를 상속하여 Gumbel Softmax prior 추가
- **주요 메서드**:
  - `get_client_z_distribution()`: 클라이언트의 z-distribution (mu, logvar) 반환
  - `update_prior_from_server()`: 서버로부터 받은 prior 업데이트
  - `_hook_on_batch_forward()`: Forward pass에서 mixture prior와의 KL divergence 계산
  - `_hook_on_fit_end()`: 라운드 종료 시 z-history 수집 및 t-SNE 시각화

### 2. Server/Client 통신 메커니즘 ⭐ NEW

#### `federatedscope/llm/llm_local/server.py` (수정)
- **Z-distribution 수집** (`_perform_federated_aggregation()`):
  - 클라이언트들로부터 `client_z_mu`, `client_z_logvar` 수집
  - 수집된 z-distribution들을 `self.vpl_gp_prior_mus`, `self.vpl_gp_prior_logvars`, `self.vpl_gp_prior_weights`에 저장
  - Weighted average 계산 (클라이언트별 sample size 기반)
  
- **Prior 브로드캐스트** (`broadcast_model_para()`):
  - `vpl_gp_prior_mus`, `vpl_gp_prior_logvars`, `vpl_gp_prior_weights`를 메시지에 포함하여 클라이언트들에게 브로드캐스트
  
- **버그 수정**:
  - `callback_funcs_for_grouping()`에서 `self.msg_buffer['adapter_eval']` 초기화 추가 (KeyError 방지)
  - `import copy` 추가 (NameError 방지)

#### `federatedscope/llm/llm_local/client.py` (수정)
- **Z-distribution 전송** (`callback_funcs_for_model_para()`):
  - `get_client_z_distribution()`으로 클라이언트의 z-distribution 가져오기
  - `model_para_all`에 `client_z_mu`, `client_z_logvar` 추가 (adapter filtering 이후에 추가하여 필터링되지 않도록 보장)
  
- **Prior 수신 및 업데이트**:
  - 서버로부터 받은 `vpl_gp_prior_mus`, `vpl_gp_prior_logvars`, `vpl_gp_prior_weights` 추출
  - Trainer의 `update_prior_from_server()`로 전달하여 prior 업데이트

### 3. Trainer 등록 및 빌더 수정 ⭐ NEW

#### `federatedscope/llm/trainer/__init__.py` (수정)
- `from federatedscope.llm.trainer import vpl_gp_reward_choice_trainer` 추가

#### `federatedscope/core/auxiliaries/trainer_builder.py` (수정)
- `TRAINER_CLASS_DICT`에 `"vplgprewardchoicetrainer": "VPLGPRewardChoiceTrainer"` 추가
- `get_trainer()`에 `vplgprewardchoicetrainer` 분기 추가

### 4. t-SNE 시각화 로깅 개선 ⭐ NEW

#### `federatedscope/llm/trainer/vpl_reward_choice_trainer.py` (수정)
- Debug 로그를 INFO 레벨로 변경하여 로그 파일에 표시
- `round_num`을 `ctx.monitor.monitored_object.state`에서 가져오도록 수정

#### `federatedscope/llm/trainer/vpl_gp_reward_choice_trainer.py` (수정)
- 동일한 t-SNE 시각화 로깅 개선 적용

### 4-1. Preference feature difference 및 서버 라벨 기반 orthogonal loss ⭐ NEW

#### `federatedscope/llm/trainer/vpl_reward_choice_trainer.py` (수정)
- **Preference feature difference 기본 적용**: `(prompt + chosen) - (prompt + rejected)` feature로 z posterior 입력 생성
- **Choice 프롬프트 파싱**: `### RESPONSE A/B` 구간을 찾아 A/B 세그먼트를 분리하고 difference 계산
- **서버 라벨 기반 orthogonal loss**: 서버에서 제공한 균형 라벨로 prototype 정렬 (fallback은 기존 choice token)
- **설정 옵션**: `llm.vpl_use_server_orthogonal_labels` (default True)

#### `federatedscope/llm/llm_local/server.py` (수정)
- **서버 z 클러스터링 + 균형 할당 라벨 생성**: 클라이언트 z-mean을 balanced k-means로 클러스터링하여 라벨 생성
- **브로드캐스트**: `vpl_orthogonal_client_labels`를 각 클라이언트로 전달

#### `federatedscope/llm/llm_local/client.py` (수정)
- **라벨 수신 및 trainer 전달**: 서버 라벨을 `update_orthogonal_label_from_server()`로 전달

### 5. GPU 할당 및 설정 ⭐ NEW

#### 하이퍼파라미터 서치 스크립트 수정
- `scripts/fedbiscuit/hhst_hpsearch.sh`: GPU_IDS를 `(0 1 4)`로 변경, `CUDA_VISIBLE_DEVICES` 제거
- `scripts/fedbiscuit/hhst_hpsearch_retry_failed.sh`: GPU_IDS를 `(0 1 4)`로 변경, GPU_JOB_COUNTS 배열 크기 3으로 변경, `CUDA_VISIBLE_DEVICES` 제거
- `scripts/fedbiscuit/hrl_hpsearch_eval.sh`: GPU_IDS를 `(0 1 4)`로 변경, `CUDA_VISIBLE_DEVICES` 제거

#### VPL-GP 스크립트 생성 ⭐ NEW
- `scripts/server/vpl-gp/hhst.sh`: VPL-GP HHST 실행 스크립트 (TID: 50100, device: 5)
- `scripts/server/vpl-gp/hrl.sh`: VPL-GP HRL 실행 스크립트 (TID: 50200, device: 5)

#### VPL-GP 설정 파일 생성 ⭐ NEW
- `cfg/vpl-gp/hhst.yaml`: VPL-GP binary selector 설정 (device: 5, trainer: vplgprewardchoicetrainer)
- `cfg/vpl-gp/hrl.yaml`: VPL-GP RL 설정 (device: 5, trainer: vplgprewardchoicetrainer)
- `cfg/vpl-gp/test_hrl_selector.yaml`: VPL-GP RL용 selector checkpoint 설정

### 6. CUDA_VISIBLE_DEVICES 제거 ⭐ NEW
- 모든 스크립트에서 `CUDA_VISIBLE_DEVICES` 환경변수 제거
- Config 파일의 `device` 설정만 사용하도록 변경
- 이로 인해 `CUDA error: invalid device ordinal` 에러 해결

## Config 파일 설정

### VPL-GP HHST (`cfg/vpl-gp/hhst.yaml`) ⭐ NEW
```yaml
device: 5
trainer:
  type: vplgprewardchoicetrainer
llm:
  vpl_latent_dim: 32
  vpl_kl_weight: 0.1
  vpl_feature_method: choice_logits
  vpl_gp_temperature: 1.0  # Gumbel Softmax temperature
  vpl_use_feature_difference: true  # (chosen - rejected) feature difference
```

### VPL-GP HRL (`cfg/vpl-gp/hrl.yaml`) ⭐ NEW
```yaml
device: 5
trainer:
  type: vplgprewardchoicetrainer
llm:
  vpl_latent_dim: 32
  vpl_kl_weight: 0.1
  vpl_feature_method: choice_logits
  vpl_gp_temperature: 1.0
```

### VPL HHST (`cfg/vpl/hhst.yaml`)
```yaml
device: 0
trainer:
  type: vplrewardchoicetrainer
llm:
  vpl_latent_dim: 32
  vpl_kl_weight: 0.1
  vpl_feature_method: choice_logits
  vpl_use_feature_difference: true  # (chosen - rejected) feature difference
```

## 실행 방법

### VPL-GP HHST ⭐ NEW
```bash
bash scripts/server/vpl-gp/hhst.sh
```
- TID: 50100
- 로그: `outputs/50100.log`
- 체크포인트: `/hdd/hdd3/kjb/checkpoints/..._t50100.ckpt`
- GPU: 5

### VPL-GP HRL ⭐ NEW
```bash
bash scripts/server/vpl-gp/hrl.sh
```
- TID: 50200
- 로그: `outputs/50200.log`
- 체크포인트: `/hdd/hdd3/kjb/checkpoints/..._t50200.ckpt`
- GPU: 5

### VPL HHST
```bash
bash scripts/vpl/hhst.sh
```
- TID: 10100
- 로그: `outputs/10100.log`
- GPU: 0

### 하이퍼파라미터 서치
```bash
# 전체 하이퍼파라미터 서치 실행
bash scripts/fedbiscuit/hhst_hpsearch.sh

# 실패한 작업 재실행
bash scripts/fedbiscuit/hhst_hpsearch_retry_failed.sh
```
- GPU: 0, 1, 4 (GPU 5는 VPL-GP 전용)

## 현재 진행 상황

### 완료된 작업
✅ VPL 컴포넌트 통합 (variational_encoder, latent_projection, z_to_embedding)
✅ 클라이언트별 z 추론 및 저장
✅ Conditional data generation (z embedding injection)
✅ VPL-based response selection
✅ Personalized test evaluation (클라이언트별, 그룹별 평균)
✅ **VPL-GP 알고리즘 구현** ⭐ NEW
✅ **Server/Client 통신 메커니즘 (z-distribution 수집 및 브로드캐스트)** ⭐ NEW
✅ **VPL-GP용 스크립트 및 설정 파일 생성** ⭐ NEW
✅ **t-SNE 시각화 로깅 개선** ⭐ NEW
✅ **CUDA_VISIBLE_DEVICES 제거 및 config device 설정 사용** ⭐ NEW
✅ **Trainer 등록 및 빌더 수정** ⭐ NEW

### 진행 중/이슈
- **t-SNE 시각화**: preference 클러스터링(예: harmless/helpful) 개선 확인 중
- **서버 라벨 기반 orthogonal loss**: 균형 클러스터링이 실제로 z 군집을 개선하는지 검증 필요
- **하이퍼파라미터 서치**: 실패한 작업 재실행 중

## 주요 함수 및 메서드

### VariationalEncoderGP 클래스 ⭐ NEW

#### Prior 관리
- `update_prior(client_mus, client_logvars, client_weights)`: 다른 클라이언트들의 z-distribution으로 prior 업데이트
- `sample_prior(batch_size, use_gumbel)`: Gumbel Softmax를 사용한 prior 샘플링

#### KL Divergence
- `kl_divergence(mu, logvar, use_gumbel_prior)`: Mixture prior와의 KL divergence 계산
  - `log q(z|x)`: Posterior의 log probability
  - `log p_mixture(z)`: Mixture prior의 log probability (log-sum-exp trick 사용)
  - `KL = E_q[log q(z|x)] - E_q[log p_mixture(z)]`

### VPLGPRewardChoiceTrainer 클래스 ⭐ NEW

#### Z-distribution 관리
- `get_client_z_distribution()`: 클라이언트의 z-distribution (mu, logvar) 반환
- `update_prior_from_server(client_mus, client_logvars, client_weights)`: 서버로부터 받은 prior 업데이트

#### Training
- `_hook_on_batch_forward(ctx)`: Forward pass에서 mixture prior와의 KL divergence 계산
- `_hook_on_fit_end(ctx)`: 라운드 종료 시 z-history 수집 및 t-SNE 시각화

## 중요한 구현 세부사항

### 1. VPL-GP Prior Update
```python
# Server에서 클라이언트들의 z-distribution 수집
client_z_mus = [client1_mu, client2_mu, ...]
client_z_logvars = [client1_logvar, client2_logvar, ...]

# Weighted average 계산
weights = [sample_size1, sample_size2, ...]
weights = weights / sum(weights)

# Prior 저장
self.vpl_gp_prior_mus = torch.stack(client_z_mus)  # (num_clients, latent_dim)
self.vpl_gp_prior_logvars = torch.stack(client_z_logvars)  # (num_clients, latent_dim)
self.vpl_gp_prior_weights = weights

# Client에서 prior 업데이트
trainer.update_prior_from_server(client_mus, client_logvars, client_weights)
variational_encoder.update_prior(client_mus, client_logvars, client_weights)
```

### 2. KL Divergence with Mixture Prior
```python
# Mixture prior: p_mixture(z) = Σ_i w_i * N(z; μ_i, σ_i²)
# KL divergence: KL(q(z|x) || p_mixture(z)) = E_q[log q(z|x)] - E_q[log p_mixture(z)]

# Log-sum-exp trick for numerical stability
log_p_components = []
for i in range(num_clients):
    log_p_i = log N(z; μ_i, σ_i²) + log w_i
    log_p_components.append(log_p_i)

log_p_mixture = logsumexp(log_p_components)  # log Σ_i w_i * N(z; μ_i, σ_i²)
kl = log_q - log_p_mixture
```

### 3. Gumbel Softmax Sampling
```python
# Gumbel Softmax: differentiable approximation of categorical sampling
logits = log(prior_weights)  # (num_clients,)
gumbel_noise = -log(-log(U + eps) + eps)  # Gumbel(0, 1)
y = logits + gumbel_noise
y_soft = softmax(y / temperature)  # Differentiable soft sample

# Weighted combination of client distributions
z_samples = [sample_from_client_i() for i in range(num_clients)]
z_prior = sum(y_soft[i] * z_samples[i] for i in range(num_clients))
```

## 체크포인트 및 로그

### 체크포인트 위치
- VPL HHST: `/hdd/hdd3/kjb/checkpoints/..._t10100.ckpt`
- VPL-GP HHST: `/hdd/hdd3/kjb/checkpoints/..._t50100.ckpt` ⭐ NEW
- VPL-GP HRL: `/hdd/hdd3/kjb/checkpoints/..._t50200.ckpt` ⭐ NEW

### 로그 파일
- `outputs/10100.log`: VPL HHST 로그
- `outputs/50100.log`: VPL-GP HHST 로그 ⭐ NEW
- `outputs/50200.log`: VPL-GP HRL 로그 ⭐ NEW

## 다음 단계 및 개선 방향

### 가능한 다음 작업
1. **VPL vs VPL-GP 성능 비교**: 두 알고리즘의 성능 차이 분석
2. **Mixture Prior 효과 분석**: 다른 클라이언트들의 분포를 prior로 사용하는 것이 학습에 미치는 영향
3. **t-SNE 시각화 분석**: 클라이언트별 z 분포의 변화 추적
4. **Hyperparameter 튜닝**: `vpl_gp_temperature`, `vpl_kl_weight` 등
5. **성능 최적화**: 메모리, 속도 개선

### 알려진 이슈 및 해결책
- **CUDA error: invalid device ordinal**: ✅ 해결됨 (CUDA_VISIBLE_DEVICES 제거, config device 설정 사용)
- **Trainer 등록 에러**: ✅ 해결됨 (trainer_builder.py에 vplgprewardchoicetrainer 등록)
- **Adapter filtering 버그**: ✅ 해결됨 (server.py에서 adapter_eval 버퍼 초기화)
- **t-SNE 시각화 로깅**: ✅ 해결됨 (debug 로그를 INFO로 변경, round_num 수정)

## 환경 설정

- **Python**: 3.9 (conda 환경: `biscuit`)
- **PyTorch**: 1.10.1 (CUDA 11.3)
- **Device**: 
  - GPU 0: VPL HHST
  - GPU 5: VPL-GP HHST/HRL
  - GPU 0, 1, 4: 하이퍼파라미터 서치
- **Model**: google/gemma-2b

## 참고 사항

- **VPL-GP는 VPL의 확장**: VPLRewardChoiceTrainer를 상속하여 모든 VPL 기능 사용
- **Mixture Prior**: 다른 클라이언트들의 z-distribution을 prior로 사용하여 federated learning에서 클라이언트 간 지식 공유
- **Gumbel Softmax**: Differentiable categorical sampling을 위한 Gumbel Softmax 사용
- **t-SNE 시각화**: 10라운드마다 z-history를 시각화하여 클라이언트별 z 분포 추적
- **Wandb 로깅**: VPL-GP 메트릭 (vpl_kl_loss, vpl_reconstruction_loss) 로깅
- **균형 라벨**: 서버에서 z-mean 기반 balanced k-means로 클라이언트 라벨 생성 후 orthogonal loss에 사용

---

**작성일**: 2026-01-09
**작업자**: kjb
**프로젝트 경로**: `/home/kjb/ppfl`
**최근 업데이트**: VPL-GP 알고리즘 구현 및 Server/Client 통신 메커니즘 추가