# RL 데이터 생성 및 VPL Conditional Generation 가이드

## RL 데이터 생성 프로세스

### 1. RL 전 데이터 생성 (Pre-RL Generation)

**위치**: `federatedscope/llm/rlhf/standalone_training.py` - `load_pairwise_data()`

**프로세스**:
```python
def load_pairwise_data(self):
    # 파일 경로: rlhf_pair_data_{model_name}_{dataset_name}_{num_comp}.json
    gen_fp = os.path.join(
        self.data_root,
        f"rlhf_pair_data_{model_name}_{dataset_name}_{num_comp}.json"
    )
    
    if os.path.exists(gen_fp):
        # 이미 생성된 파일이 있으면 로드
        list_pairwise_data = json.load(open(gen_fp, "r"))
    else:
        # 파일이 없으면 새로 생성
        list_pairwise_data = self._generate_pairwise_data(
            self.list_train_prompts,
            self.model,  # 원본 모델 (generator)
            self.generator_tokenizer,
            self.generation_prompt,
            max_new_tokens=self.config.llm.max_new_token,
            num_completions=self.config.llm.num_completions
        )
        # 파일로 저장
        json.dump(list_pairwise_data, open(gen_fp, "w"))
```

**생성 양**:
- 각 prompt에 대해 `num_completions` 개수만큼 response 생성
- `num_return_sequences = max(2, num_completions)` (최소 2개)
- 생성된 response들을 조합하여 pairwise 데이터 생성:
  - 예: 3개 response 생성 → 3C2 = 3개의 (A, B) 쌍

**생성 설정**:
```python
generate_kwargs = dict(
    top_p=1.0,
    temperature=0.7,
    do_sample=True,
    max_new_tokens=max_new_tokens,  # config.llm.max_new_token
    num_return_sequences=max(2, num_completions),  # config.llm.num_completions
)
```

**저장 위치**: `{data_root}/rlhf_pair_data_{model_name}_{dataset_name}_{num_comp}.json`

### 2. RL 중 데이터 생성 (During-RL Generation)

**현재 구현**: RL training 중에는 **새로운 generation이 없습니다**.

**프로세스**:
```python
def train(self, saveto=None, early_exiting=False):
    # 1. Selector preference data 로드 (한 번만 실행)
    list_train_dict = self.load_selector_preference_data(saveto, early_exiting)
    
    # 2. DPO trainer 생성
    self.trainer = DPORewardTrainer(...)
    
    # 3. Training rounds (generation 없음)
    for r in range(self.config.federate.total_round_num):
        sample_size, model_para_all, results = self.trainer.train()
        # ... checkpoint 저장 ...
```

**특징**:
- `load_selector_preference_data()`는 `load_pairwise_data()`를 호출하지만, 이미 파일이 있으면 로드만 함
- RL training 중에는 pre-generated 데이터를 재사용
- 새로운 generation은 없음

**Federated RL 예외** (`fedserver.py`):
```python
def _start_new_training_round(self, aggregated_num=0):
    if self.state % self._cfg.llm.fedrlhf.frequency == 0:
        # 주기적으로 DPO-based selection 수행
        list_train_dict = self.policy_trainer.dpo_better_response()
        # 하지만 이것도 generation이 아니라 selection
```

### 3. 데이터 생성 요약

| 단계 | Generation 여부 | 양 | 시점 |
|------|----------------|-----|------|
| **RL 전** | ✅ Yes | `num_completions` per prompt | `load_pairwise_data()` 호출 시 (한 번) |
| **RL 중** | ❌ No | 0 | Training rounds 동안 없음 |
| **Federated RL** | ❌ No (Selection만) | 0 | 주기적으로 selection만 수행 |

## VPL Conditional Generation

### VPL은 Generation이 아니라 Selection에 사용됨

**중요**: VPL은 **response generation**이 아니라 **response selection**에 사용됩니다.

### 1. VPL의 역할

**위치**: `federatedscope/llm/rlhf/variational_selector.py` - `variational_better_response()`

**프로세스**:
```python
def variational_better_response(list_data_dict, selector_model, ...):
    # 1. 이미 생성된 response pair (output_A, output_B)에서 시작
    # 2. Preference features 추출
    features = extract_preference_features_for_variational(...)
    
    # 3. Variational encoder로 z 샘플링
    mu, logvar = variational_encoder.encode(features)
    z = variational_encoder.reparameterize(mu, logvar)  # q(z|x)
    
    # 4. z를 조건으로 선택 (conditional selection)
    z_projection = latent_projection(z)  # z → choice logits
    score_A = win_A_logit + z_bias_A  # z-conditioned score
    score_B = lose_B_logit + z_bias_B
    
    # 5. 선택 (generation 아님!)
    choice = 0 if score_A > score_B else 1
```

### 2. Conditional Selection vs Conditional Generation

**Conditional Selection (현재 VPL 구현)**:
- ✅ 이미 생성된 response pair 중에서 선택
- ✅ z를 조건으로 선택 점수 조정
- ❌ 새로운 response 생성은 하지 않음

**Conditional Generation (미구현)**:
- ❌ z를 조건으로 새로운 response 생성
- ❌ Generation 시 z를 prompt에 포함하거나 logits에 반영
- ❌ 예: `model.generate(..., z=z)` 같은 형태

### 3. VPL이 Generation에 사용될 수 있는 방법 (미구현)

만약 VPL을 conditional generation에 사용하려면:

```python
# 예시 (현재 구현되지 않음)
def conditional_generate(model, prompt, z, ...):
    # z를 prompt에 포함하거나
    prompt_with_z = f"{prompt} [z={z}]"
    
    # 또는 z를 logits에 반영
    outputs = model(input_ids, ...)
    logits = outputs.logits
    z_adjusted_logits = logits + latent_projection(z)
    
    # z-conditioned generation
    generated = model.generate(
        input_ids,
        logits_processor=lambda logits: z_adjusted_logits,
        ...
    )
```

**현재 상태**: 이런 conditional generation은 구현되지 않았습니다.

## 전체 데이터 흐름

```
1. RL 전 (Pre-RL)
   └─→ load_pairwise_data()
       └─→ _generate_pairwise_data()
           └─→ model.generate()  # num_completions 개 생성
           └─→ combinations()  # pairwise 쌍 생성
           └─→ 저장: rlhf_pair_data_*.json

2. Selector Preference (Pre-RL)
   └─→ load_selector_preference_data()
       └─→ load_pairwise_data()  # 이미 생성된 데이터 로드
       └─→ _choose_better_response() 또는 variational_better_response()
           └─→ 선택 결과 저장: generated_choose_*.json

3. RL Training (During-RL)
   └─→ train()
       └─→ load_selector_preference_data()  # 한 번만 호출
       └─→ DPORewardTrainer.train()  # 여러 rounds
           └─→ Pre-generated 데이터 재사용
           └─→ 새로운 generation 없음
```

## 설정 파라미터

### Generation 관련 설정

```yaml
llm:
  num_completions: 2  # 각 prompt당 생성할 response 개수
  max_new_token: 60   # 생성할 최대 토큰 수
```

### VPL Selection 관련 설정

```yaml
llm:
  rlhf_use_variational_selection: True  # VPL selection 사용 여부
  rlhf_selector_checkpoint: "path/to/selector.ckpt"  # VPL selector 체크포인트
  rlhf_variational_num_samples: 1  # z 샘플링 횟수 (평균화용)
  vpl_use_feature_difference: True  # Feature extraction 방법
```

## 요약

1. **RL 전 데이터 생성**:
   - ✅ `load_pairwise_data()`에서 한 번만 실행
   - ✅ 각 prompt당 `num_completions` 개 response 생성
   - ✅ 파일로 저장되어 재사용

2. **RL 중 데이터 생성**:
   - ❌ 현재 구현에서는 없음
   - ✅ Pre-generated 데이터만 재사용

3. **VPL Conditional Generation**:
   - ❌ VPL은 generation이 아니라 **selection**에 사용
   - ✅ z를 조건으로 response pair 중에서 선택
   - ❌ z-conditioned response generation은 미구현

4. **VPL Conditional Selection** (현재 구현):
   - ✅ `variational_better_response()`에서 구현
   - ✅ z를 샘플링하여 선택 점수에 반영
   - ✅ 이미 생성된 response 중에서 선택

## 새로운 Generation Logic (2026-01-22 업데이트)

### 1. Client-Specific Conditional Generation

**위치**: `federatedscope/llm/rlhf/standalone_training.py` - `_generate_pairwise_data()`

**핵심 기능**:
- Selector checkpoint에서 학습된 client별 평균 z 값을 사용하여 conditional generation 수행
- 각 prompt에 대해 harmlessness와 helpfulness 두 세트의 데이터 생성
- Client ID에 따라 다른 z 값을 사용하여 client-specific response 생성

**프로세스**:
```python
def _generate_pairwise_data(self, list_data_dict, model, tokenizer, prompt, ...):
    # 1. Client average z 로드 (selector checkpoint에서)
    if use_variational_generation and self.client_average_z_dict is None:
        selector_ckpt_path = getattr(self.config.llm, 'rlhf_selector_checkpoint', None)
        self.client_average_z_dict = load_client_average_z_from_checkpoint(
            selector_ckpt_path, device=self.device
        )
    
    # 2. 각 샘플의 client_id에 따라 z 선택
    for data in input_data:
        client_id = data.get('client_id', None)
        if client_id is not None and client_id in client_average_z_dict:
            z = client_average_z_dict[client_id]  # Client-specific z 사용
        else:
            z = overall_avg_z  # Fallback: 전체 평균 z
        
        # 3. z를 embedding에 주입하여 conditional generation
        z_embedding = z_to_embedding(z)  # (latent_dim,) -> (embedding_dim,)
        inputs_embeds = input_embeddings + z_embedding.expand(seq_len, -1)
        output_ids = model.generate(inputs_embeds=inputs_embeds, ...)
```

**특징**:
- ✅ Selector checkpoint에 저장된 client별 평균 z 재사용 (새로 compute하지 않음)
- ✅ Client ID에 따라 다른 z 사용 → client-specific response 생성
- ✅ Harmlessness (client_id=1)와 helpfulness (client_id=2) 두 세트 생성

### 2. Dual Client ID Assignment for Each Prompt

**위치**: `federatedscope/llm/rlhf/standalone_training.py` - `load_pairwise_data()`

**프로세스**:
```python
def load_pairwise_data(self):
    # VPL 모델인 경우, 각 prompt에 대해 두 개의 client_id 할당
    if is_vpl_model:
        prompts_with_client_id = []
        for idx, prompt_data in enumerate(self.list_train_prompts):
            # Harmlessness client_id 할당
            harmless_client_id = 1  # RL training: 항상 client_id=1
            # Helpfulness client_id 할당
            helpful_client_id = 2   # RL training: 항상 client_id=2
            
            # 두 개의 복사본 생성
            prompt_data_harmless = copy.deepcopy(prompt_data)
            prompt_data_harmless['client_id'] = harmless_client_id
            prompt_data_harmless['preference_type'] = 'harmlessness'
            prompts_with_client_id.append(prompt_data_harmless)
            
            prompt_data_helpful = copy.deepcopy(prompt_data)
            prompt_data_helpful['client_id'] = helpful_client_id
            prompt_data_helpful['preference_type'] = 'helpfulness'
            prompts_with_client_id.append(prompt_data_helpful)
```

**결과**:
- 각 prompt에 대해 2개의 데이터 생성:
  - `{prompt, client_id=1, preference_type='harmlessness'}` → harmlessness z로 generation
  - `{prompt, client_id=2, preference_type='helpfulness'}` → helpfulness z로 generation
- 총 데이터 양: 원본 prompt 수 × 2

### 3. Selector Config의 Client Num 사용

**위치**: `federatedscope/llm/rlhf/main.py`, `standalone_training.py`

**프로세스**:
```python
# main.py
selector_cfg = global_cfg.clone()
if selector_args.selector_cfg_file:
    selector_cfg.merge_from_file(selector_args.selector_cfg_file)

RLHF_finetuning(
    ...,
    selector_cfg=selector_cfg,  # Selector config 전달
)

# standalone_training.py
def __init__(self, ..., selector_cfg=None, ...):
    # Selector config의 client_num 우선 사용
    if selector_cfg is not None:
        selector_client_num = getattr(selector_cfg.federate, 'client_num', None)
        if selector_client_num is not None:
            self.num_clients = selector_client_num  # Selector의 client_num 사용
            logger.info(f"Using selector config's client_num: {self.num_clients}")
```

**효과**:
- ✅ Selector가 훈련된 client setting (예: 10 clients)을 그대로 사용
- ✅ RL training에서도 selector와 동일한 client distribution 사용
- ✅ Client average z가 selector의 client_num과 일치

### 4. Client Average Z Checkpoint 저장 및 로드

**위치**: 
- 저장: `federatedscope/llm/llm_local/server.py` - `check_and_save()`
- 로드: `federatedscope/llm/rlhf/load_vpl_components.py` - `load_client_average_z_from_checkpoint()`

**저장 프로세스** (Selector 훈련 시):
```python
# server.py
def check_and_save(self):
    # Checkpoint 저장 전에 client 평균 z 계산
    self._compute_client_average_z_for_checkpoint()
    
    # Checkpoint에 저장
    self.aggregator.save_model(
        path, 
        self.state, 
        client_average_z_dict=self.client_average_z_dict
    )

def _compute_client_average_z_for_checkpoint(self):
    """t-SNE에 사용된 모든 client의 z 값으로부터 평균 계산"""
    for client_id in range(1, self.client_num + 1):
        if client_id in self.client_z_values_dict:
            z_list = self.client_z_values_dict[client_id]
            z_array = np.array(z_list)  # (num_samples, latent_dim)
            avg_z = np.mean(z_array, axis=0)  # (latent_dim,)
            self.client_average_z_dict[client_id] = torch.tensor(avg_z)
```

**로드 프로세스** (RL training 시):
```python
# load_vpl_components.py
def load_client_average_z_from_checkpoint(checkpoint_path, device='cuda:0'):
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Priority 1: client_average_z_dict 확인 (새 형식)
    if 'client_average_z_dict' in ckpt:
        client_average_z_dict = ckpt['client_average_z_dict']
        # {client_id: z_tensor} 반환
        return convert_to_tensors(client_average_z_dict, device)
    
    # Priority 2: client_z_mus 확인 (기존 형식)
    # ...
    
    # 없으면 None 반환 → training data에서 새로 compute
    return None
```

**효과**:
- ✅ Selector 훈련 시 모든 client의 평균 z가 checkpoint에 저장됨
- ✅ RL training 시 checkpoint에서 로드하여 재사용 (새로 compute하지 않음)
- ✅ 없으면 fallback: training data에서 새로 compute

### 5. Binary Selector의 Client-Specific Selection

**위치**: `federatedscope/llm/rlhf/variational_selector.py` - `variational_better_response()`

**프로세스**:
```python
def variational_better_response(..., client_average_z_dict=None):
    # 1. Client-specific z 사용 확인
    if client_average_z_dict is not None:
        for sample in list_data_dict:
            client_id = sample.get('client_id', None)
            if client_id in client_average_z_dict:
                z = client_average_z_dict[client_id]  # Client-specific z 사용
                provided_z_tensors.append(z)
    
    # 2. Client-specific z로 conditional selection
    if use_provided_z:
        # z를 직접 사용 (새로 추론하지 않음)
        z_mean = np.array([z.cpu().numpy() for z in provided_z_tensors])
    else:
        # 데이터에서 z 추론
        mu, logvar = variational_encoder.encode(features)
        z = variational_encoder.reparameterize(mu, logvar)
    
    # 3. z-conditioned selection
    z_projection = latent_projection(z)
    choice = select_better_response(z_projection, ...)
```

**효과**:
- ✅ Generation 시 사용한 client_id와 동일한 z를 selection에도 사용
- ✅ Client-specific preference에 맞는 선택 수행
- ✅ Harmlessness와 helpfulness 각각에 맞는 선택

### 6. 전체 데이터 흐름 (업데이트)

```
1. Selector 훈련 (Binary Selector)
   └─→ Federated training with 10 clients
       └─→ 각 client의 z 값 수집 (client_z_values_dict)
       └─→ t-SNE 시각화에 사용
       └─→ Checkpoint 저장 시 client 평균 z 계산 및 저장
           └─→ client_average_z_dict: {1: z_1, 2: z_2, ..., 10: z_10}

2. RL 전 데이터 생성 (Pre-RL Generation)
   └─→ load_pairwise_data()
       └─→ Selector config의 client_num 사용 (10 clients)
       └─→ 각 prompt에 대해 2개 client_id 할당:
           ├─→ client_id=1 (harmlessness)
           └─→ client_id=2 (helpfulness)
       └─→ _generate_pairwise_data()
           └─→ Checkpoint에서 client_average_z_dict 로드
           └─→ 각 샘플의 client_id에 따라 z 선택
           └─→ z를 embedding에 주입하여 conditional generation
           └─→ 저장: rlhf_pair_data_*.json (2x prompts)

3. Selector Preference (Pre-RL)
   └─→ load_selector_preference_data()
       └─→ load_pairwise_data()  # 이미 생성된 데이터 로드
       └─→ variational_better_response()
           └─→ client_average_z_dict 사용 (generation과 동일한 z)
           └─→ Client-specific selection 수행
           └─→ 저장: generated_choose_*.json

4. RL Training (During-RL)
   └─→ train()
       └─→ load_selector_preference_data()  # 한 번만 호출
       └─→ DPORewardTrainer.train()  # 여러 rounds
           └─→ Pre-generated 데이터 재사용
           └─→ Client-specific preference 반영된 데이터 사용
```

### 7. 설정 파라미터 (업데이트)

```yaml
# Selector config (hhst-ortho-50024.yaml)
federate:
  client_num: 10  # Selector 훈련 시 사용된 client 수

# RL config (hrl-ortho-51024.yaml)
federate:
  client_num: 1  # RL training은 standalone (무시됨)

llm:
  rlhf_use_variational_generation: True  # Z-dependent generation 활성화
  rlhf_selector_checkpoint: "path/to/selector.ckpt"  # Selector checkpoint
  rlhf_use_variational_selection: True  # VPL selection 사용
```

### 8. 주요 개선 사항 요약

1. **Client-Specific Generation**:
   - ✅ Selector checkpoint의 client별 평균 z 재사용
   - ✅ Client ID에 따라 다른 z 사용 → client-specific response 생성

2. **Dual Client ID Assignment**:
   - ✅ 각 prompt에 대해 harmlessness와 helpfulness 두 세트 생성
   - ✅ 총 데이터 양: 원본 × 2

3. **Selector Config 통합**:
   - ✅ Selector의 client_num을 RL에서도 사용
   - ✅ Selector와 RL 간 client distribution 일치

4. **Checkpoint 저장/로드**:
   - ✅ Selector 훈련 시 client 평균 z 저장
   - ✅ RL training 시 checkpoint에서 로드 (새로 compute하지 않음)

5. **Client-Specific Selection**:
   - ✅ Generation과 동일한 client z를 selection에도 사용
   - ✅ Client-specific preference에 맞는 선택

## 최신 Generation 및 Evaluation Logic (2026-01-22 이후 업데이트)

### 1. Standard Generation (Z-Conditional Generation 없음)

**위치**: `federatedscope/llm/rlhf/standalone_training.py` - `_generate_pairwise_data()`

**핵심 변경사항**:
- ❌ Z-conditional generation 비활성화 (`use_variational_generation = False`)
- ✅ Standard generation 수행 (각 prompt에 대해 한 번만)
- ✅ `client_id`와 `preference_type`은 generation 시점에 없음 (selection 시점에 할당)

**프로세스**:
```python
def _generate_pairwise_data(self, list_data_dict, model, tokenizer, prompt, ...):
    # Standard generation (no z-conditional generation)
    use_variational_generation = False  # Disabled
    
    # Generate responses once per prompt
    for prompt_data in list_data_dict:
        # Standard generation without z
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_completions,
            ...
        )
        # No client_id or preference_type in generated data
```

**특징**:
- ✅ 모든 prompt에 대해 동일한 방식으로 generation 수행
- ✅ `client_id`와 `preference_type`은 generation 후 selection 단계에서 할당
- ✅ 생성된 response pair는 selection 단계에서 harmlessness와 helpfulness로 각각 처리

### 2. 전체 Response Pair에 대한 Dual Selection

**위치**: `federatedscope/llm/rlhf/standalone_training.py` - `load_selector_preference_data()`

**핵심 변경사항**:
- ✅ 생성된 **전체 response pair**에 대해 harmlessness와 helpfulness로 각각 binary selection 수행
- ✅ 각 pair에 대해 두 번의 selection:
  1. Harmlessness selection: Client 1의 z (z_1) 사용
  2. Helpfulness selection: Client 2의 z (z_2) 사용

**프로세스**:
```python
def load_selector_preference_data(self, saveto, early_exiting=False):
    list_pairwise_data = self.load_pairwise_data()  # Standard generation 결과
    
    # 전체 response pair에 대해 harmlessness selection (client 1, z_1)
    harmless_preference = variational_better_response(
        copy.deepcopy(list_pairwise_data),  # 전체 pair 사용
        ...,
        client_average_z_dict={1: z_1}  # Client 1의 z 사용
    )
    for sample in harmless_preference:
        sample['preference_type'] = 'harmlessness'
        sample['client_id'] = 1
    
    # 전체 response pair에 대해 helpfulness selection (client 2, z_2)
    helpful_preference = variational_better_response(
        copy.deepcopy(list_pairwise_data),  # 전체 pair 사용
        ...,
        client_average_z_dict={2: z_2}  # Client 2의 z 사용
    )
    for sample in helpful_preference:
        sample['preference_type'] = 'helpfulness'
        sample['client_id'] = 2
    
    # 두 세트 결합
    list_preference_data = harmless_preference + helpful_preference
```

**결과**:
- 각 response pair에 대해 2개의 선택 결과 생성:
  - Harmlessness 선택: `{prompt, output_A, output_B, choice, preference_type='harmlessness', client_id=1}`
  - Helpfulness 선택: `{prompt, output_A, output_B, choice, preference_type='helpfulness', client_id=2}`
- 총 데이터 양: 원본 response pair 수 × 2

### 3. Conflicting Selection 통계

**위치**: `federatedscope/llm/rlhf/standalone_training.py` - `load_selector_preference_data()`

**핵심 기능**:
- 같은 response pair에 대해 harmlessness와 helpfulness가 다른 선택을 하는 경우를 "conflicting selection"으로 식별
- 통계 계산 및 로깅:
  - 총 preference samples 수
  - 총 unique response pairs 수
  - Conflicting pairs 수 및 비율
  - Non-conflicting pairs 수 및 비율
- 처음 5개의 conflicting 예시를 로그에 출력

**프로세스**:
```python
# Group by (prompt, output_A, output_B) to find same pairs
pair_to_selections = {}
for sample in list_preference_data:
    pair_key = (sample['prompt'], sample['output_A'], sample['output_B'])
    if pair_key not in pair_to_selections:
        pair_to_selections[pair_key] = []
    pair_to_selections[pair_key].append(sample)

# Find conflicting pairs
conflicting_pairs = []
for pair_key, selections in pair_to_selections.items():
    harmless_selection = [s for s in selections if s.get('preference_type') == 'harmlessness']
    helpful_selection = [s for s in selections if s.get('preference_type') == 'helpfulness']
    
    if len(harmless_selection) > 0 and len(helpful_selection) > 0:
        harmless_choice = harmless_selection[0].get('choice')
        helpful_choice = helpful_selection[0].get('choice')
        
        if harmless_choice != helpful_choice:
            # Conflicting: same pair, different choices
            conflicting_pairs.append(...)

# Log statistics
logger.info(f"Total unique response pairs: {total_pairs}")
logger.info(f"Conflicting pairs: {conflicting_count} ({conflicting_ratio:.1f}%)")
logger.info(f"Non-conflicting pairs: {non_conflicting_count} ({non_conflicting_ratio:.1f}%)")
```

### 4. Test Set Loading (Prompt만 로드)

**위치**: `federatedscope/llm/rlhf/standalone_training.py` - `train()`

**핵심 변경사항**:
- ✅ Test set은 **prompt만 로드** (chosen/rejected pair 없음)
- ✅ `load_hh_rlhf_for_rlhf()`에 `raw_no_prompt=True` 전달
- ✅ `LLMDataset`과 `LLMDataCollator` 사용 (LLMComparisonDataset 아님)

**프로세스**:
```python
def train(self, saveto=None, early_exiting=False):
    # Test set: prompt만 로드 (evaluation을 위한 새 response generation)
    from federatedscope.llm.dataloader.hh_rlhf import load_hh_rlhf_for_rlhf
    
    list_test_prompts, _, _ = load_hh_rlhf_for_rlhf(
        self.data_root,
        self.config,
        max_num_test=-1,
        raw_no_prompt=True  # Prompt만 반환
    )
    
    # LLMDataset 사용 (LLMComparisonDataset 아님)
    test_dataset = LLMDataset(list_test_prompts, self.tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=self.config.dataloader.batch_size,
        collate_fn=LLMDataCollator(self.tokenizer)
    )
```

**효과**:
- ✅ Test 시 모델이 새로운 response를 생성하여 evaluation 수행
- ✅ Reward model evaluation과 winrate evaluation 모두 생성된 response 사용
- ✅ Chosen/rejected pair와의 비교가 아닌, 모델 자체의 성능 평가

### 5. Winrate Evaluation with GPT API

**위치**: `federatedscope/llm/metric/winrate_metrics.py` - `_get_winrate_scores_with_gpt_api()`

**핵심 기능**:
- Fine-tuned model과 baseline model의 response를 GPT API로 비교
- Fine-tuned model: `ctx.model`로 response 생성
- Baseline model: Adapter를 비활성화한 `ctx.model`로 response 생성
- GPT API (기본값: `gpt-4o-mini`)로 두 response 비교하여 winrate 계산

**프로세스**:
```python
def _get_winrate_scores_with_gpt_api(ctx, prompt_template, metric_name="winrate"):
    # 1. Fine-tuned model로 response 생성
    fine_tuned_responses = generate_responses(
        ctx.model,  # Fine-tuned model with adapter
        test_prompts,
        ...
    )
    
    # 2. Baseline model로 response 생성 (adapter 비활성화)
    ctx.model.disable_adapter()  # Temporarily disable adapter
    baseline_responses = generate_responses(
        ctx.model,  # Baseline model (adapter disabled)
        test_prompts,
        ...
    )
    ctx.model.enable_adapter()  # Re-enable adapter
    
    # 3. GPT API로 두 response 비교
    for prompt, fine_tuned_resp, baseline_resp in zip(...):
        comparison_prompt = prompt_template.format(
            prompt=prompt,
            response_a=fine_tuned_resp,  # Fine-tuned model response
            response_b=baseline_resp     # Baseline model response
        )
        
        gpt_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": comparison_prompt}]
        )
        
        # Parse GPT response: "A" or "B"
        choice = parse_gpt_choice(gpt_response)
        if choice == "A":
            wins += 1  # Fine-tuned model wins
    
    # 4. Winrate 계산
    winrate = wins / total_comparisons * 100
```

**설정**:
```yaml
eval:
  use_gpt_api_for_winrate: True  # GPT API 사용 활성화
  openai_model: "gpt-4o-mini"    # 기본값 (비용 효율적)
  max_samples_for_reward: 30      # 평가할 샘플 수
```

**효과**:
- ✅ Fine-tuned model vs baseline model의 직접 비교
- ✅ GPT-4o-mini를 judge로 사용하여 객관적인 평가
- ✅ FedBiscuit 논문 및 최신 preference learning 논문과 동일한 평가 방식

### 6. 전체 데이터 흐름 (최신 버전)

```
1. Selector 훈련 (Binary Selector)
   └─→ Federated training with 10 clients
       └─→ 각 client의 z 값 수집 (client_z_values_dict)
       └─→ t-SNE 시각화에 사용
       └─→ Checkpoint 저장 시 client 평균 z 계산 및 저장
           └─→ client_average_z_dict: {1: z_1, 2: z_2, ..., 10: z_10}

2. RL 전 데이터 생성 (Pre-RL Generation)
   └─→ load_pairwise_data()
       └─→ Standard generation (z-conditional generation 없음)
       └─→ 각 prompt에 대해 한 번만 generation 수행
       └─→ 저장: rlhf_pair_data_*.json (response pairs)

3. Selector Preference (Pre-RL)
   └─→ load_selector_preference_data()
       └─→ load_pairwise_data()  # 이미 생성된 데이터 로드
       └─→ 전체 response pair에 대해:
           ├─→ Harmlessness selection (client 1, z_1)
           └─→ Helpfulness selection (client 2, z_2)
       └─→ Conflicting selection 통계 계산 및 로깅
       └─→ 저장: generated_choose_*.json (2x response pairs)

4. RL Training (During-RL)
   └─→ train()
       └─→ load_selector_preference_data()  # 한 번만 호출
       └─→ DPORewardTrainer.train()  # 여러 rounds
           └─→ Pre-generated 데이터 재사용
           └─→ Client-specific preference 반영된 데이터 사용

5. Test Evaluation
   └─→ Test set: prompt만 로드 (chosen/rejected pair 없음)
       └─→ Fine-tuned model로 새 response 생성
       └─→ Reward model evaluation (harmlessness, helpfulness)
       └─→ Winrate evaluation:
           ├─→ Fine-tuned model response 생성
           ├─→ Baseline model response 생성 (adapter 비활성화)
           └─→ GPT API로 두 response 비교하여 winrate 계산
```

### 7. 설정 파라미터 (최신 버전)

```yaml
# Selector config (hhst-ortho-50024.yaml)
federate:
  client_num: 10  # Selector 훈련 시 사용된 client 수

# RL config (hrl-ortho-51024.yaml)
federate:
  client_num: 1  # RL training은 standalone (무시됨)

llm:
  rlhf_use_variational_generation: False  # Z-conditional generation 비활성화 (standard generation)
  rlhf_use_variational_selection: True     # VPL selection 사용
  rlhf_selector_checkpoint: "path/to/selector.ckpt"  # Selector checkpoint

eval:
  use_gpt_api_for_winrate: True  # GPT API 사용 활성화
  openai_model: "gpt-4o-mini"     # GPT 모델 (기본값)
  max_samples_for_reward: 30       # 평가할 샘플 수
```

### 8. 주요 변경사항 요약 (최신 버전)

1. **Standard Generation**:
   - ❌ Z-conditional generation 비활성화
   - ✅ 각 prompt에 대해 한 번만 standard generation 수행
   - ✅ `client_id`와 `preference_type`은 selection 시점에 할당

2. **Dual Selection on All Pairs**:
   - ✅ 전체 response pair에 대해 harmlessness와 helpfulness로 각각 selection 수행
   - ✅ Client 1의 z (z_1)로 harmlessness selection
   - ✅ Client 2의 z (z_2)로 helpfulness selection

3. **Conflicting Selection 통계**:
   - ✅ 같은 pair에 대해 harmlessness와 helpfulness가 다른 선택을 하는 경우 식별
   - ✅ 통계 계산 및 로깅 (conflicting pairs 수 및 비율)
   - ✅ 처음 5개 conflicting 예시 로그 출력

4. **Test Set Loading**:
   - ✅ Prompt만 로드 (chosen/rejected pair 없음)
   - ✅ 모델이 새 response를 생성하여 evaluation 수행

5. **Winrate Evaluation with GPT API**:
   - ✅ Fine-tuned model vs baseline model 비교
   - ✅ GPT API (gpt-4o-mini)를 judge로 사용
   - ✅ FedBiscuit 논문 및 최신 논문과 동일한 평가 방식

## Client Average Z Dictionary 로드 및 t-SNE 시각화 (2026-01-22 업데이트)

### 1. Client Average Z Dictionary 로드 시점

**위치**: `federatedscope/llm/rlhf/standalone_training.py` - `load_pairwise_data()`

**핵심 변경사항**:
- ✅ `use_variational_generation` 또는 `use_variational_selection` 중 하나라도 True이면 `client_average_z_dict` 로드
- ✅ Selection 단계와 Training 단계에서도 사용 가능하도록 초기에 로드
- ✅ 여러 시점에서 fallback 로드 로직 포함

**프로세스**:
```python
def load_pairwise_data(self):
    use_variational_generation = getattr(self.config.llm, 'rlhf_use_variational_generation', False)
    use_variational_selection = getattr(self.config.llm, 'rlhf_use_variational_selection', False)
    
    # If either variational generation or selection is enabled, load client average z
    if use_variational_generation or use_variational_selection:
        selector_ckpt_path = getattr(self.config.llm, 'rlhf_selector_checkpoint', None)
        # ... checkpoint path resolution ...
        
        if selector_ckpt_path and os.path.exists(selector_ckpt_path):
            # Load VPL components
            variational_encoder, feature_extractor, _, _ = load_vpl_components_from_checkpoint(...)
            
            # Load client average z (required for both generation and selection)
            if self.client_average_z_dict is None:
                self.client_average_z_dict = load_client_average_z_from_checkpoint(
                    selector_ckpt_path, device=self.device
                )
                if self.client_average_z_dict is not None and len(self.client_average_z_dict) > 0:
                    logger.info(f"Loaded client average z for {len(self.client_average_z_dict)} clients "
                               f"(for {'generation' if use_variational_generation else ''} "
                               f"{'and ' if use_variational_generation and use_variational_selection else ''}"
                               f"{'selection' if use_variational_selection else ''})")
```

**Fallback 로드 시점**:
1. `load_pairwise_data()`: 초기 로드 (generation 전)
2. `load_selector_preference_data()`: Selection 단계에서 fallback
3. `_generate_pairwise_data()`: Generation 단계에서 fallback
4. `train()`: Training 시작 전 fallback

### 2. t-SNE 시각화 저장 위치

**Generation 단계 (RL 시작 전)**:
- 경로: `exp/vplgp_hrl_ortho_t51024/sub_exp_YYYYMMDDHHMMSS/cross_client_z_tsne_generation.png`
- 시점: `load_pairwise_data()`에서 generation 완료 후
- 조건: `is_vpl_model=True`이고 `client_average_z_dict`가 로드된 경우
- Round 번호: `-1` (generation 단계 표시)

**Training 중 (매 5라운드마다 또는 마지막 라운드)**:
- 경로: `exp/vplgp_hrl_ortho_t51024/sub_exp_YYYYMMDDHHMMSS/cross_client_z_tsne_round_{round_num}.png`
- 시점: Round 5, 10, 15, 20, ... 또는 마지막 라운드
- 조건: `z_values_list`에 z 값이 수집된 경우

**코드 위치**:
```python
# Generation 단계
if is_vpl_model and self.client_average_z_dict is not None and len(self.client_average_z_dict) > 0:
    visualize_cross_client_z(
        z_values=z_array,
        client_labels=client_labels_list,
        orthogonal_labels=orthogonal_labels_list,
        round_num=-1,  # Generation 단계
        output_dir=self.config.outdir,
        wandb_project=wandb_project
    )

# Training 중
if (len(z_values_list) > 0 and 
    ((r + 1) % 5 == 0 or r == self.config.federate.total_round_num - 1)):
    visualize_cross_client_z(
        z_values=z_array,
        client_labels=client_labels_list,
        round_num=r,
        output_dir=output_dir,
        wandb_project=wandb_project
    )
```

### 3. Client Average Z Dictionary가 비어있을 때의 문제점

**문제점**:
1. ❌ Generation 단계에서 t-SNE 시각화 불가
2. ❌ Selection 단계에서 client-specific z 사용 불가
3. ❌ Training 중 z-conditional generation 사용 불가

**해결책**:
- ✅ `use_variational_generation` 또는 `use_variational_selection` 중 하나라도 True이면 초기에 로드
- ✅ 여러 시점에서 fallback 로드 로직 포함
- ✅ 로드 실패 시 경고 메시지 출력

**확인 방법**:
```bash
# 로그에서 client average z 로드 확인
grep -i "Loaded client average z\|No client average z" outputs/51024.log

# t-SNE 파일 확인
find exp/vplgp_hrl_ortho_t51024 -name "*tsne*" -o -name "*cross_client*"
```

### 4. 주요 개선 사항 (2026-01-22)

1. **Client Average Z Dictionary 로드 개선**:
   - ✅ `use_variational_selection`이 True일 때도 초기에 로드
   - ✅ Selection과 Training에서 사용 가능하도록 보장
   - ✅ 여러 fallback 로드 시점 제공

2. **t-SNE 시각화 개선**:
   - ✅ Generation 단계에서도 시각화 수행
   - ✅ 저장 위치 명확화 (`exp/{expname}/sub_exp_{timestamp}/`)
   - ✅ 로그에 저장 경로 출력

3. **에러 처리 개선**:
   - ✅ `client_average_z_dict`가 비어있을 때 경고 메시지 출력
   - ✅ 로드 실패 시 fallback 로직 동작
