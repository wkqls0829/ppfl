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
