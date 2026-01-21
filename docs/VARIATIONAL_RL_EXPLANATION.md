# Variational RL (Reinforcement Learning with Variational Preference Learning)

## 개요

Variational RL은 **Variational Preference Learning (VPL)**로 학습된 selector model의 latent representation `z`를 활용하여 **z-conditioned RL training**을 수행하는 방법입니다. 이는 각 클라이언트의 개인화된 preference를 latent space에서 표현하고, 이를 RL training에 반영하여 더 효과적인 personalized policy를 학습할 수 있게 합니다.

## 핵심 개념

### 1. Z-Conditioned Generation

RL training 시 각 샘플의 latent `z`를 추론하여 input embedding에 주입함으로써, 모델이 클라이언트별 preference에 맞춰 conditional generation을 수행할 수 있도록 합니다.

```
Input → Extract Features → Infer z from q(z|x) → Inject z into Embeddings → Generate
```

### 2. Variational Selection

Preference data 생성 시에도 VPL posterior를 활용하여 z-conditioned selection을 수행합니다. 이는 단순히 selector model의 logits만 사용하는 것보다 더 일관된 preference를 반영합니다.

### 3. 두 가지 주요 사용 시나리오

1. **Variational Selection** (`rlhf_use_variational_selection=True`): Preference data 생성 시 z를 추론하여 선택
2. **Variational Generation** (`rlhf_use_variational_generation=True`): DPO training 시 z를 embedding에 주입하여 conditional generation

## 아키텍처

### VPL Components

RL training에 필요한 VPL 컴포넌트들은 selector model checkpoint에서 로드됩니다:

1. **Variational Encoder**: Preference features에서 latent `z`를 추론
   - Input: Preference features (extracted from selector model)
   - Output: `mu`, `logvar` (posterior parameters)
   - Sampling: `z ~ q(z|x) = N(z; mu, exp(logvar))`

2. **Feature Extractor**: Raw preference features를 variational encoder 입력 형태로 변환
   - Input: Raw features (choice logits 또는 embedding difference)
   - Output: Processed features (128-dim)

3. **Latent Projection**: `z`를 choice logits로 변환 (variational selection용)
   - Input: `z` (latent_dim)
   - Output: Choice logits (num_choices)

4. **Z-to-Embedding**: `z`를 model embedding dimension으로 projection (conditional generation용)
   - Input: `z` (latent_dim)
   - Output: Embedding vector (embedding_dim)
   - Usage: `inputs_embeds = base_embeddings + z_to_embedding(z)`

## 구현 세부사항

### 1. Variational Selection (`variational_selector.py`)

Preference data 생성 시 z-conditioned selection을 수행합니다.

#### Workflow

```python
# 1. Extract preference features from pairwise data
features = extract_preference_features_for_variational(
    selector_model, selector_tokenizer, list_pairwise_data, ...
)

# 2. Process through feature extractor
features = feature_extractor(features)

# 3. Infer z from posterior q(z|x)
mu, logvar = variational_encoder.encode(features)
z = variational_encoder.reparameterize(mu, logvar)

# 4. Make z-conditioned choice
z_projection = latent_projection(z)  # (batch, num_choices)
choice_logits = base_logits + z_projection  # Add z-conditioned bias
choice = argmax(choice_logits)
```

#### 주요 함수

- `extract_preference_features_for_variational()`: Selector model에서 preference features 추출
  - Choice token 위치의 hidden states 추출
  - Embedding difference 계산 (chosen - rejected)
  - Feature concatenation: `[chosen_emb, rejected_emb, difference]` 또는 `[difference]`

- `variational_better_response()`: Z-conditioned selection 수행
  - `use_provided_z=True`: 데이터에 이미 `z`가 있으면 재사용 (이전 round에서 생성된 z)
  - `use_provided_z=False`: 데이터에서 z를 추론
  - Multiple sampling 지원 (`num_samples`): 여러 z 샘플에 대해 majority vote

#### 데이터 저장

선택된 preference data에는 다음 정보가 저장됩니다:
- `choice`: 선택된 응답 (0=A, 1=B)
- `z`: Sampled latent vector (list)
- `z_mu`: Posterior mean (list)
- `z_logvar`: Posterior log variance (list)

### 2. Z-Dependent Generation (`reward_trainer.py`)

DPO training 시 z를 embedding에 주입하여 conditional generation을 수행합니다.

#### Workflow

```python
# 1. Get z from batch (provided or inferred)
z = _get_z_from_batch(ctx, batch_size, input_ids, attention_mask)

# 2. Infer z if not provided
if z is None:
    z = _infer_z_from_input(ctx, input_ids, attention_mask)

# 3. Inject z into embeddings
inputs_embeds = _inject_z_to_embeddings(ctx, input_ids, z)

# 4. Forward pass with z-conditioned embeddings
outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, ...)
```

#### 주요 함수

- `_get_z_from_batch()`: Batch에서 z 추출
  - `LLMComparisonDataset`에서 `z`, `z_mu`, `z_logvar` 필드 확인
  - 제공된 z가 있으면 사용, 없으면 `None` 반환

- `_infer_z_from_input()`: Input에서 z 추론
  ```python
  # 1. Get hidden states from base model
  hidden_states = model.get_input_embeddings()(input_ids)
  
  # 2. Extract features (mean pooling or choice position)
  features = extract_features(hidden_states, attention_mask)
  
  # 3. Process through feature extractor
  features = feature_extractor(features)
  
  # 4. Encode to get z
  mu, logvar = variational_encoder.encode(features)
  z = variational_encoder.reparameterize(mu, logvar)
  ```

- `_inject_z_to_embeddings()`: Z를 embedding에 주입
  ```python
  # 1. Get base embeddings
  base_embeddings = model.get_input_embeddings()(input_ids)
  
  # 2. Project z to embedding dimension
  z_embedding = z_to_embedding(z)  # (batch, seq_len, embedding_dim)
  
  # 3. Add z to embeddings (broadcast across sequence)
  inputs_embeds = base_embeddings + z_embedding.unsqueeze(1)
  ```

#### DeepSpeed 지원

DeepSpeed를 사용하는 경우 `_batch_forward_deepspeed()`에서도 동일하게 z injection을 수행합니다.

### 3. VPL Components Loading (`load_vpl_components.py`)

Selector checkpoint에서 VPL 컴포넌트를 로드합니다.

#### Workflow

```python
variational_encoder, feature_extractor, latent_projection, z_to_embedding = \
    load_vpl_components_from_checkpoint(checkpoint_path, config, device)
```

#### 로딩 과정

1. Checkpoint에서 model state dict 로드
2. VPL hyperparameters 추출 (config에서)
   - `vpl_latent_dim`: Latent dimension (default: 32)
   - `vpl_use_gp_prior`: GP prior 사용 여부
   - `vpl_use_feature_difference`: Feature difference 사용 여부
3. 컴포넌트 초기화
   - VariationalEncoder 또는 VariationalEncoderGP
   - Feature extractor (MLP)
   - Latent projection (Linear)
   - Z-to-embedding (Linear)
4. Checkpoint에서 가중치 로드
   - Key matching: `variational_encoder.*`, `feature_extractor.*`, etc.
   - `strict=False`로 부분 로딩 허용

### 4. Standalone Training Integration (`standalone_training.py`)

RLHF standalone training에서 variational selection을 통합합니다.

#### Workflow

```python
# In load_selector_preference_data()
if use_variational_selection:
    # Load VPL components
    variational_encoder, feature_extractor, latent_projection, z_to_embedding = \
        load_vpl_components_from_checkpoint(selector_ckpt_path, config, device)
    
    # Use variational selection
    list_preference_data = variational_better_response(
        list_pairwise_data,
        selector_model, selector_tokenizer,
        variational_encoder, feature_extractor,
        prompt_template, choices,
        latent_projection=latent_projection,
        z_to_embedding=z_to_embedding,
        use_provided_z=use_provided_z  # Use z from previous rounds if available
    )
else:
    # Standard selection
    list_preference_data = _choose_better_response(...)
```

#### Z 재사용

- `use_provided_z=True`: 이전 round에서 생성된 z를 재사용
  - 데이터에 `z` 필드가 있으면 추론 없이 사용
  - 일관성 있는 preference 반영
  - 계산 비용 절감

## 설정 옵션

### Config 파일 설정

```yaml
llm:
  # Variational generation for DPO training
  rlhf_use_variational_generation: True  # Enable z-dependent generation
  
  # Variational selection for preference data generation
  rlhf_use_variational_selection: True  # Enable z-conditioned selection
  
  # Selector checkpoint path
  rlhf_selector_checkpoint: "/path/to/selector/checkpoint.ckpt"
  
  # VPL hyperparameters (should match selector training config)
  vpl_latent_dim: 32
  vpl_use_feature_difference: True
  vpl_use_gp_prior: False  # Set to True if selector used GP prior
  
  # Variational selection options
  rlhf_variational_num_samples: 1  # Number of z samples for selection (majority vote)
```

### Script 실행

```bash
# RL training with variational generation
python federatedscope/llm/rlhf/main.py \
    --cfg cfg/vpl-gp/hrl-ortho-51022.yaml \
    --selector-cfg-file cfg/vpl-gp/hhst-ortho-50022.yaml
```

- `--cfg`: RL training config (DPO training 설정)
- `--selector-cfg-file`: Selector model config (VPL components 로딩용)

## 데이터 흐름

### Round 1: Preference Data 생성

```
1. Generate pairwise responses (A, B)
2. Extract preference features
3. Infer z from q(z|x)
4. Make z-conditioned choice
5. Save preference data with z, z_mu, z_logvar
```

### Round 2+: DPO Training

```
1. Load preference data (with z from previous round)
2. Use provided z or infer new z
3. Inject z into embeddings
4. Forward pass with z-conditioned embeddings
5. Compute DPO loss
6. Backward pass
```

## 장점

1. **Personalized Generation**: 클라이언트별 preference에 맞춘 conditional generation
2. **Consistency**: Preference data 생성과 RL training 모두에서 동일한 z 사용
3. **Efficiency**: 이전 round의 z 재사용으로 추론 비용 절감
4. **Flexibility**: Variational selection과 generation을 독립적으로 활성화 가능

## 제한사항

1. **Selector Dependency**: Selector model checkpoint가 필요
2. **Latent Dimension**: `vpl_latent_dim`이 selector와 RL training에서 일치해야 함
3. **Feature Extraction**: Selector와 동일한 feature extraction 방법 사용 필요

## 참고 파일

- `federatedscope/llm/rlhf/variational_selector.py`: Variational selection 구현
- `federatedscope/llm/trainer/reward_trainer.py`: Z-dependent generation 구현
- `federatedscope/llm/rlhf/load_vpl_components.py`: VPL components 로딩
- `federatedscope/llm/rlhf/standalone_training.py`: RLHF training 통합
- `federatedscope/llm/dataset/llm_dataset.py`: Z 필드 저장/로딩
