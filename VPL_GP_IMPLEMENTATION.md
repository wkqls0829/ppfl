# VPL-GP (Variational Preference Learning with Gumbel-Softmax Prior) 구현 문서

## 개요

이 문서는 Federated Learning 환경에서 VPL-GP (Variational Preference Learning with Gumbel-Softmax Prior)를 구현한 내용을 정리합니다. VPL-GP는 다른 클라이언트들의 z-distribution을 mixture prior로 사용하여 federated learning에서 클라이언트 간 지식 공유를 가능하게 합니다.

## 핵심 개념

### 1. VPL (Variational Preference Learning)
- 각 클라이언트마다 latent vector `z`를 추론하여 개인화된 preference 학습
- ELBO 최적화: `E_q(z|x)[log p(y|z,x)] - KL(q(z|x) || p(z))`
- 표준 VPL은 고정된 정규분포 prior `p(z) = N(0, I)` 사용

### 2. VPL-GP (Gumbel-Softmax Prior)
- 다른 클라이언트들의 z-distribution을 mixture prior로 사용
- Mixture prior: `p_mixture(z) = Σ_i w_i * N(z; μ_i, σ_i²)`
- Gumbel-Softmax relaxation을 사용하여 differentiable sampling 가능
- Federated learning에서 클라이언트 간 지식 공유 가능

## 구현 파일 구조

```
federatedscope/llm/
├── model/
│   ├── variational_encoder.py          # 기본 VariationalEncoder
│   └── variational_encoder_gp.py       # VariationalEncoderGP (새로 생성) ⭐
├── trainer/
│   ├── vpl_reward_choice_trainer.py    # 기본 VPL trainer
│   └── vpl_gp_reward_choice_trainer.py # VPL-GP trainer (새로 생성) ⭐
└── llm_local/
    ├── server.py                        # Server 구현 (수정) ⭐
    ├── client.py                        # Client 구현 (수정) ⭐
    └── z_visualization.py               # t-SNE 시각화 (새로 생성) ⭐
```

## 주요 컴포넌트

### 1. VariationalEncoderGP (`federatedscope/llm/model/variational_encoder_gp.py`)

#### 클래스 구조
```python
class VariationalEncoderGP(VariationalEncoder):
    def __init__(self, input_dim, latent_dim=32, hidden_dims=[256, 128], 
                 temperature=1.0, num_clients=10):
        # VariationalEncoder 상속
        # Gumbel-Softmax temperature 설정
        # Prior distributions 초기화
```

#### 주요 메서드

##### `update_prior(client_mus, client_logvars, client_weights)`
- 다른 클라이언트들의 z-distribution으로 mixture prior 업데이트
- **입력**:
  - `client_mus`: (num_clients, latent_dim) - 다른 클라이언트들의 평균 벡터
  - `client_logvars`: (num_clients, latent_dim) - 다른 클라이언트들의 로그 분산 벡터
  - `client_weights`: (num_clients,) - 각 클라이언트 분포의 가중치
- **동작**:
  - Prior distributions 저장 및 정규화
  - 통계 정보 로깅 (mu_norm, logvar_mean, avg_mu_distance)

##### `sample_prior(batch_size, use_gumbel=True)`
- Gumbel-Softmax를 사용한 mixture prior에서 샘플링
- **입력**:
  - `batch_size`: 생성할 샘플 수
  - `use_gumbel`: Gumbel-Softmax 사용 여부
- **출력**: (batch_size, latent_dim) - 샘플링된 latent 벡터
- **알고리즘**:
  1. Gumbel noise 생성: `g = -log(-log(U))` where U ~ Uniform(0,1)
  2. Gumbel logits 계산: `logits = (log(w_i) + g_i) / temperature`
  3. Softmax로 component 선택 확률 계산
  4. 각 component에서 샘플링 후 가중 평균

##### `kl_divergence(mu, logvar, use_gumbel_prior=True)`
- Mixture prior와의 KL divergence 계산
- **입력**:
  - `mu`: Posterior 평균 (batch_size, latent_dim)
  - `logvar`: Posterior 로그 분산 (batch_size, latent_dim)
  - `use_gumbel_prior`: Gumbel-Softmax prior 사용 여부
- **출력**: KL divergence (scalar)
- **알고리즘**:
  1. Reparameterization trick으로 z 샘플링
  2. `log q(z|x)` 계산
  3. `log p_mixture(z) = log(Σ_i w_i * N(z; μ_i, σ_i²))` 계산 (log-sum-exp trick 사용)
  4. `KL = E_q[log q(z|x) - log p_mixture(z)]` 계산

### 2. VPLGPRewardChoiceTrainer (`federatedscope/llm/trainer/vpl_gp_reward_choice_trainer.py`)

#### 클래스 구조
```python
class VPLGPRewardChoiceTrainer(VPLRewardChoiceTrainer):
    def __init__(self, model, data, device, config, only_for_eval=False, monitor=None):
        # VPLRewardChoiceTrainer 상속
        # VariationalEncoderGP로 교체
        # Z history 및 client z distribution 초기화
```

#### 주요 메서드

##### `get_client_z_distribution()`
- 클라이언트의 z-distribution (mu, logvar) 반환
- **출력**: `(mu, logvar)` 튜플
- **용도**: 서버로 전송하여 mixture prior 구성

##### `get_client_z_values()`
- 클라이언트의 z 값들 반환 (시각화용)
- **출력**: (num_samples, latent_dim) 텐서
- **용도**: t-SNE 시각화

##### `update_prior_from_server(client_mus, client_logvars, client_weights)`
- 서버로부터 받은 mixture prior 업데이트
- **입력**:
  - `client_mus`: (num_clients, latent_dim)
  - `client_logvars`: (num_clients, latent_dim)
  - `client_weights`: (num_clients,)
- **동작**: `variational_encoder.update_prior()` 호출

##### `update_orthogonal_label_from_server(label)`
- 서버로부터 받은 orthogonal label 업데이트
- **입력**: `label` (int) - 클라이언트의 orthogonal label (0 또는 1)

##### `get_client_orthogonal_prototypes()`
- 클라이언트의 orthogonal prototypes 반환
- **출력**: (num_prototypes, latent_dim) 텐서 또는 None

### 3. Server 구현 (`federatedscope/llm/llm_local/server.py`)

#### 추가된 속성
```python
self.vpl_gp_prior_mus = None          # (num_clients, latent_dim)
self.vpl_gp_prior_logvars = None     # (num_clients, latent_dim)
self.vpl_gp_prior_weights = None     # (num_clients,)
self.vpl_orthogonal_client_labels = None  # {client_id: label}
self.client_z_values_dict = defaultdict(list)  # {client_id: [z_values]}
self.client_orthogonal_prototypes_dict = {}   # {client_id: prototypes}
```

#### 주요 메서드

##### `_collect_vpl_gp_prior_distributions()`
- 클라이언트들로부터 z-distribution 수집
- **호출 시점**: `_perform_federated_aggregation()` 후
- **동작**:
  1. 각 클라이언트의 `client_z_mu`, `client_z_logvar` 추출
  2. 샘플 크기를 가중치로 사용
  3. Prior 업데이트 (기존 클라이언트 업데이트, 신규 클라이언트 추가)
  4. 가중치 정규화
- **로그**: "Collected X client z distributions for VPL-GP prior..."

##### `_compute_balanced_orthogonal_labels()`
- 균형 잡힌 orthogonal labels 계산
- **우선순위**:
  1. Manual labels (설정된 경우): 참여 클라이언트의 첫 절반은 0, 나머지는 1
  2. K-means clustering: z means에 대해 k-means (k=2) 수행
- **로그**: "Computed balanced orthogonal labels for X clients..."

##### `_compute_manual_orthogonal_labels(train_msg_buffer)`
- 수동 orthogonal labels 할당
- **로직**: 참여 클라이언트의 첫 절반은 harmless (0), 나머지는 helpful (1)
- **예시**: Round 0에서 [2, 3, 5, 9, 10] 참여 → {2:0, 3:0, 5:0, 9:1, 10:1}

##### `_collect_z_values_for_visualization()`
- t-SNE 시각화를 위한 z 값 수집
- **동작**:
  1. 각 클라이언트의 `client_z_values` 추출
  2. `client_z_values_dict`에 누적 저장
  3. 10라운드마다 `_visualize_cross_client_z()` 호출
- **로그**: "Round X: Collected z values from Y clients..."

##### `_visualize_cross_client_z()`
- Cross-client z 값들의 t-SNE 시각화
- **동작**:
  1. 모든 클라이언트의 z 값 수집
  2. t-SNE로 2D 변환
  3. 클라이언트별로 색상 구분하여 scatter plot
  4. Orthogonal prototypes가 있으면 별표로 표시
  5. 파일 저장 및 WandB 로깅
- **출력**: `cross_client_z_tsne_round_{round_num}.png`

##### `broadcast_model_para()` (오버라이드)
- VPL-GP prior 및 orthogonal labels 브로드캐스트
- **동작**:
  1. 부모 메서드 호출 (모델 파라미터 브로드캐스트)
  2. Round > 0일 때만 prior 브로드캐스트
  3. `vpl_gp_prior` 메시지 타입으로 prior 전송
  4. `vpl_orthogonal_labels` 메시지 타입으로 labels 전송
- **로그**: "Broadcasting VPL-GP prior with X client distributions..."

### 4. Client 구현 (`federatedscope/llm/llm_local/client.py`)

#### 수정된 메서드

##### `callback_funcs_for_model_para()`
- z distribution 전송 로직 추가
- **동작**:
  1. 기존 모델 파라미터 수집
  2. `get_client_z_distribution()`으로 z distribution 가져오기
  3. `client_z_mu`, `client_z_logvar`, `sample_size` 추가
  4. `get_client_z_values()`로 z 값 추가 (시각화용)
  5. `get_client_orthogonal_prototypes()`로 prototypes 추가

##### `callback_funcs_for_vpl_gp_prior()` (새로 추가)
- 서버로부터 받은 VPL-GP prior 처리
- **동작**:
  1. `vpl_gp_prior` 메시지 수신
  2. Tensor 변환 및 device 이동
  3. `trainer.update_prior_from_server()` 호출
- **등록**: `_register_default_handlers()`에서 `vpl_gp_prior` 메시지 타입 등록

##### `callback_funcs_for_vpl_orthogonal_labels()` (새로 추가)
- 서버로부터 받은 orthogonal labels 처리
- **동작**:
  1. `vpl_orthogonal_labels` 메시지 수신
  2. 클라이언트 ID에 해당하는 label 추출
  3. `trainer.update_orthogonal_label_from_server()` 호출
- **등록**: `_register_default_handlers()`에서 `vpl_orthogonal_labels` 메시지 타입 등록

### 5. t-SNE 시각화 (`federatedscope/llm/llm_local/z_visualization.py`)

#### 함수: `visualize_cross_client_z()`

##### 파라미터
- `z_values`: (num_points, latent_dim) - z 값 배열
- `client_labels`: (num_points,) - 각 z 값의 클라이언트 ID
- `orthogonal_labels`: (num_points,) - 각 z 값의 orthogonal label (선택)
- `orthogonal_prototypes`: (num_prototypes, latent_dim) - Orthogonal prototypes (선택)
- `round_num`: 현재 라운드 번호
- `output_dir`: 출력 디렉토리
- `wandb_project`: WandB 프로젝트 이름 (선택)

##### 동작
1. t-SNE로 2D 변환 (perplexity 자동 조정)
2. 클라이언트별로 색상 구분하여 scatter plot
3. Orthogonal prototypes가 있으면 별표로 표시
4. Orthogonal labels가 있으면 검은 테두리로 강조
5. 파일 저장 및 WandB 로깅

## 통신 프로토콜

### 1. 클라이언트 → 서버

#### 메시지 타입: `model_para`
```python
content = (
    sample_size,  # int
    {
        # 기존 모델 파라미터들...
        'client_z_mu': tensor,           # (latent_dim,)
        'client_z_logvar': tensor,        # (latent_dim,)
        'sample_size': int,
        'client_z_values': tensor,         # (num_samples, latent_dim)
        'client_orthogonal_prototypes': tensor  # (num_prototypes, latent_dim)
    }
)
```

### 2. 서버 → 클라이언트

#### 메시지 타입: `vpl_gp_prior`
```python
content = {
    'vpl_gp_prior_mus': list,      # [[latent_dim], ...] - (num_clients, latent_dim)
    'vpl_gp_prior_logvars': list,  # [[latent_dim], ...] - (num_clients, latent_dim)
    'vpl_gp_prior_weights': list   # [float, ...] - (num_clients,)
}
```

#### 메시지 타입: `vpl_orthogonal_labels`
```python
content = {
    client_id: label,  # {2: 0, 3: 0, 5: 0, 9: 1, 10: 1}
    ...
}
```

## 실행 흐름

### Round 0 (초기 라운드)
1. **클라이언트**: 로컬 데이터로 학습, z-distribution 계산
2. **서버**: 클라이언트들의 z-distribution 수집
3. **서버**: Manual/K-means로 orthogonal labels 계산
4. **서버**: z 값 수집 및 t-SNE 시각화 (Round 0)
5. **서버**: Prior 및 labels 저장 (아직 브로드캐스트 안 함)

### Round 1+ (일반 라운드)
1. **서버**: 이전 라운드의 prior 및 labels 브로드캐스트
2. **클라이언트**: Prior 수신 및 encoder 업데이트
3. **클라이언트**: Orthogonal labels 수신
4. **클라이언트**: Mixture prior와의 KL divergence로 학습
5. **클라이언트**: 학습 후 z-distribution 계산 및 전송
6. **서버**: z-distribution 수집 및 prior 업데이트
7. **서버**: Orthogonal labels 재계산
8. **서버**: z 값 수집 및 t-SNE 시각화 (10라운드마다)

## 설정 파일 예시

```yaml
trainer:
  type: vplgprewardchoicetrainer

llm:
  vpl_use_gp_prior: True
  vpl_latent_dim: 32
  vpl_kl_weight: 0.1
  vpl_gp_temperature: 1.0
  vpl_feature_method: choice_logits
  vpl_use_manual_orthogonal_labels: True  # 또는 False (k-means 사용)
  vpl_orthogonal_weight: 10.0
```

## 주요 로그 메시지

### 서버
- `"Collected X client z distributions for VPL-GP prior. Total clients in prior: Y (updated: Z, from previous rounds: W)"`
- `"Assigned manual orthogonal labels: {...}"`
- `"Computed balanced orthogonal labels for X clients at round Y: {...}"`
- `"Round X: Collected z values from Y clients. Total accumulated: Z points across W clients"`
- `"Broadcasting VPL-GP prior with X client distributions at round Y"`
- `"Broadcasting orthogonal labels to clients at round Y"`

### 클라이언트
- `"Client X updated VPL-GP prior from server with Y client distributions at round Z"`
- `"Client X updated orthogonal label from server: Y"`

### Trainer
- `"VPLGPRewardChoiceTrainer initialized with latent_dim=X, kl_weight=Y, temperature=Z, num_clients=W"`
- `"Collected z for round X (shape: torch.Size([Y, Z]), from W batches)"`

### Encoder
- `"Updated VPL-GP prior: mu_norm=X, logvar_mean=Y, num_clients=Z, avg_mu_distance=W"`
- `"KL divergence comparison: mixture=X, standard=Y, diff=Z, active_clients=W, ..."`

### 시각화
- `"Round X: Visualizing z from Y clients across Z rounds (W total points, shape: (W, latent_dim))"`
- `"Saved cross-client z visualization to {path}"`
- `"Logged cross-client z t-SNE visualization to wandb at round X"`

## 복원 작업 요약

### 손실된 파일 복원
1. ✅ `federatedscope/llm/model/variational_encoder_gp.py` - 새로 생성
2. ✅ `federatedscope/llm/trainer/vpl_gp_reward_choice_trainer.py` - 새로 생성
3. ✅ `federatedscope/llm/llm_local/z_visualization.py` - 새로 생성

### 수정된 파일
1. ✅ `federatedscope/llm/llm_local/server.py` - VPL-GP 관련 함수들 추가
2. ✅ `federatedscope/llm/llm_local/client.py` - z distribution 전송 및 prior 수신 로직 추가
3. ✅ `federatedscope/llm/trainer/vpl_reward_choice_trainer.py` - ctx에 z 값 저장 로직 추가
4. ✅ `federatedscope/llm/metric/hhrl_metrics.py` - 잘못된 코드 제거

### 복원 기준
- `outputs/50271.log`의 로그 메시지 및 라인 번호 분석
- `PROBLEM_SITUATION.md` 및 `CODE_RESTORATION_GUIDE.md` 참고
- `personalized_FL.pdf` 논문 내용 반영

## 테스트 방법

### 1. Import 테스트
```bash
python3 -c "from federatedscope.llm.model.variational_encoder_gp import VariationalEncoderGP"
python3 -c "from federatedscope.llm.trainer.vpl_gp_reward_choice_trainer import VPLGPRewardChoiceTrainer"
python3 -c "from federatedscope.llm.llm_local.z_visualization import visualize_cross_client_z"
```

### 2. 설정 파일 확인
- `cfg/vpl-gp/hhst-fd.yaml` 또는 유사한 설정 파일 존재 확인
- `trainer.type: vplgprewardchoicetrainer` 설정 확인
- `llm.vpl_use_gp_prior: True` 설정 확인

### 3. 실행 테스트
```bash
# 50271과 동일한 설정으로 실행
python main.py --cfg cfg/vpl-gp/hhst-fd.yaml --task_id 50271

# 로그 확인
tail -f outputs/50271.log | grep -E "(VPL|vpl|prior|orthogonal|t-SNE|tsne)"
```

### 4. 확인 사항
- ✅ 서버에서 z-distribution 수집 로그 확인
- ✅ Manual orthogonal labels 할당 로그 확인
- ✅ Prior 브로드캐스트 로그 확인
- ✅ 클라이언트에서 prior 업데이트 로그 확인
- ✅ t-SNE 시각화 파일 생성 확인 (10라운드마다)
- ✅ WandB에 시각화 이미지 업로드 확인

## 참고 자료

- 논문: "Federated Variational Preference Alignment with Gumbel-Softmax Prior for Personalized user preferences"
- 원본 VPL 구현: https://github.com/WEIRDLabUW/vpl
- Gumbel-Softmax: Jang et al. (2017) "Categorical reparameterization with gumbel-softmax"

## 주의사항

1. **라운드 0**: Prior가 아직 없으므로 표준 정규분포 prior 사용
2. **라운드 1+**: Mixture prior 사용
3. **t-SNE 시각화**: 10라운드마다 실행 (메모리 절약)
4. **Manual labels**: 참여 클라이언트만 할당 (전체 클라이언트 아님)
5. **Prior 업데이트**: 기존 클라이언트는 업데이트, 신규 클라이언트는 추가
6. **가중치 정규화**: 항상 합이 1이 되도록 정규화

## 향후 개선 사항

1. **Prior 관리**: 오래된 클라이언트 제거 또는 decay
2. **시각화 개선**: 3D t-SNE, UMAP 등 추가 옵션
3. **Orthogonal loss**: Prototype 기반 orthogonal loss 구현
4. **하이퍼파라미터 튜닝**: Temperature, KL weight 등 자동 튜닝
5. **성능 최적화**: Prior 업데이트 및 시각화 최적화
