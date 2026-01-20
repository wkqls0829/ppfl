# RLHF 메모리 최적화 가이드

RLHF 학습 시 메모리 문제를 해결하기 위한 최적화 방법들을 정리합니다.

## 현재 적용된 최적화

코드베이스에서 이미 적용된 메모리 최적화 설정들:

### 1. Batch Size 및 Gradient Accumulation
- `dataloader.batch_size: 1` - 메모리 사용량을 최소화하기 위해 매우 낮게 설정
- `llm.grad_accum_step: 32` - Gradient accumulation으로 effective batch size 유지
  - Effective batch size = `batch_size * grad_accum_step = 1 * 32 = 32`

### 2. Mixed Precision Training
- `train.is_enable_half: True` - bfloat16 사용으로 메모리 50% 절감
  - 자동으로 `torch.bfloat16` dtype 적용

### 3. Generation 최적화
- `llm.generation_batch_size: 3` - 텍스트 생성 시 배치 크기 제한
- `llm.max_new_token: 512` - 생성할 최대 토큰 수 제한
- `llm.max_prompts_for_generation: 50` - 라운드당 생성할 프롬프트 수 제한

### 4. Sequence Length 제한
- `llm.tok_len: 1024` - 입력 시퀀스 최대 길이

## 추가로 적용 가능한 최적화 방법

### 1. Gradient Checkpointing (메모리 대폭 절감)
- **효과**: 메모리 사용량 약 50-70% 절감, 약 20-30% 속도 저하
- **적용 위치**: 모델 로딩 시점 (`model_builder.py`)
- **설정 방법**:
  ```yaml
  llm:
    gradient_checkpointing: True  # 추가 필요
  ```
- **구현 코드**:
  ```python
  # federatedscope/llm/model/model_builder.py
  if config.llm.gradient_checkpointing:
      model.gradient_checkpointing_enable()
  ```

### 2. CPU Offloading (극한 상황)
- **효과**: GPU 메모리를 더 절약, 속도는 느려짐
- **적용**: LoRA 레이어만 GPU에 유지, 베이스 모델은 CPU로 offload
- **주의**: 매우 느려지므로 최후의 수단으로만 사용

### 3. LoRA Rank 감소
- **효과**: LoRA 파라미터 수 감소 → 메모리 약간 절감
- **현재**: `r: 8, lora_alpha: 16`
- **권장**: `r: 4, lora_alpha: 8`로 감소 시도
  ```yaml
  llm:
    adapter:
      args: [ { 'adapter_package': 'peft', 'adapter_method': 'lora', 'r': 4, 'lora_alpha': 8, 'lora_dropout': 0.05 } ]
  ```
- **주의**: 모델 성능에 영향 가능

### 4. Sequence Length 추가 감소
- **효과**: 메모리 사용량 선형적으로 감소
- **현재**: `tok_len: 1024, max_new_token: 512`
- **권장**:
  ```yaml
  llm:
    tok_len: 512  # 1024 → 512 (50% 감소)
    max_new_token: 256  # 512 → 256 (50% 감소)
  ```
- **주의**: 더 긴 문맥이 필요한 경우 성능 저하 가능

### 5. Generation Batch Size 추가 감소
- **효과**: 텍스트 생성 시 메모리 절감
- **현재**: `generation_batch_size: 3`
- **권장**: `generation_batch_size: 1` (더 느려지지만 메모리 절감)
  ```yaml
  llm:
    generation_batch_size: 1
  ```

### 6. _choose_better_response Batch Size 감소
- **문제**: `standalone_training.py`에서 하드코딩된 `batch_size=10`
- **해결**: Config에서 설정 가능하게 수정
- **현재 코드**:
  ```python
  # federatedscope/llm/rlhf/standalone_training.py:384
  dataloader = DataLoader(
      dataset=token_dataset,
      batch_size=10,  # 하드코딩됨
      ...
  )
  ```
- **수정 필요**: Config 값 사용하도록 변경
  ```python
  batch_size = getattr(self.config.llm, 'selector_batch_size', 4)
  dataloader = DataLoader(
      dataset=token_dataset,
      batch_size=batch_size,
      ...
  )
  ```
- **Config 설정**:
  ```yaml
  llm:
    selector_batch_size: 4  # 기본값 10에서 감소
  ```

### 7. Generation 최적화 개선
- **문제**: 현재 generation이 한 번에 하나씩 처리됨
- **개선**: 더 효율적인 배치 처리 또는 streaming generation
- **효과**: 메모리 피크 감소

### 8. Gradient Accumulation Step 조정
- **현재**: `grad_accum_step: 32`
- **옵션**: 더 작은 batch_size + 더 많은 grad_accum_step
  ```yaml
  dataloader:
    batch_size: 1
  llm:
    grad_accum_step: 64  # 32 → 64 (더 작은 메모리 피크)
  ```
- **주의**: 너무 크면 속도 저하

### 9. Evaluation 최적화
- **현재**: `max_samples_for_reward: 20`
- **추가**: Evaluation 시에도 gradient tracking 비활성화 확인
  ```python
  @torch.no_grad()
  def evaluate(...):
      ...
  ```

### 10. 메모리 정리 (Garbage Collection)
- **추가**: Generation 후 명시적 메모리 정리
  ```python
  import gc
  import torch
  
  # Generation 후
  torch.cuda.empty_cache()
  gc.collect()
  ```

## 우선순위별 적용 순서

### 즉시 적용 가능 (코드 수정 없음)
1. ✅ `llm.generation_batch_size: 1` (3 → 1)
2. ✅ `llm.tok_len: 512` (1024 → 512)
3. ✅ `llm.max_new_token: 256` (512 → 256)
4. ✅ LoRA rank 감소 (`r: 4, lora_alpha: 8`)

### 코드 수정 필요 (간단)
1. ⚙️ Gradient checkpointing 추가
2. ⚙️ `_choose_better_response` batch_size config화
3. ⚙️ Generation 후 명시적 메모리 정리

### 코드 수정 필요 (복잡)
1. 🔧 CPU offloading 구현
2. 🔧 Generation 최적화 (streaming, 배치 처리 개선)

## 권장 Config 설정 (메모리 최적화)

```yaml
llm:
  rlhf: True
  tok_len: 512  # 1024 → 512
  max_new_token: 256  # 512 → 256
  generation_batch_size: 1  # 3 → 1
  selector_batch_size: 4  # 하드코딩된 10 → 4
  grad_accum_step: 64  # 32 → 64 (더 작은 메모리 피크)
  max_prompts_for_generation: 50  # 유지
  gradient_checkpointing: True  # 새로 추가
  adapter:
    args: [ { 
      'adapter_package': 'peft', 
      'adapter_method': 'lora', 
      'r': 4,  # 8 → 4
      'lora_alpha': 8,  # 16 → 8
      'lora_dropout': 0.05 
    } ]

dataloader:
  batch_size: 1  # 유지

train:
  is_enable_half: True  # 유지
```

## 메모리 사용량 추정

각 최적화 방법의 예상 메모리 절감량 (Gemma-2B 기준):

| 최적화 방법 | 메모리 절감 | 속도 영향 |
|------------|------------|----------|
| Mixed precision (이미 적용) | ~50% | 거의 없음 |
| Gradient checkpointing | ~50-70% | ~20-30% 느려짐 |
| Sequence length 50% 감소 | ~25% | 거의 없음 |
| Generation batch size 3→1 | ~15-20% | ~30% 느려짐 |
| LoRA rank 8→4 | ~5-10% | 거의 없음 |
| Selector batch size 10→4 | ~5% | 거의 없음 |

**총 예상 절감량**: 60-80% (모든 방법 적용 시)

## 주의사항

1. **성능 vs 메모리 트레이드오프**: 메모리를 줄이면 보통 속도가 느려짐
2. **모델 성능**: Sequence length 감소나 LoRA rank 감소는 모델 성능에 영향 가능
3. **단계적 적용**: 한 번에 모든 최적화를 적용하지 말고, 하나씩 테스트하면서 적용
4. **모니터링**: WandB나 `nvidia-smi`로 메모리 사용량 모니터링

## 참고

- PyTorch Gradient Checkpointing: https://pytorch.org/docs/stable/checkpoint.html
- LoRA: https://arxiv.org/abs/2106.09685
- Mixed Precision Training: https://pytorch.org/docs/stable/amp.html