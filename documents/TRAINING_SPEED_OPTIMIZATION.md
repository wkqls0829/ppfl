# 트레이닝 속도 최적화 가이드

이 문서는 모델 트레이닝 속도를 개선할 수 있는 방법들을 정리한 것입니다. 트레이닝 품질에 큰 영향을 주지 않으면서 속도를 향상시킬 수 있는 방법들을 우선순위별로 나열했습니다.

## 현재 설정 분석

### 현재 Config (`cfg/vpl-gp/hhst-60000.yaml`)
```yaml
dataloader:
  batch_size: 4
  # num_workers: 0 (기본값, 설정되지 않음)
  # pin_memory: False (기본값, 설정되지 않음)

train:
  local_update_steps: 30
  is_enable_half: True  # FP16 활성화

llm:
  grad_accum_step: 2

eval:
  freq: 2  # 매 2 round마다 평가
```

## 개선 방법 (우선순위 순)

### 1. DataLoader 최적화 (즉시 적용 가능, 영향: 중간)

**현재 상태**: `num_workers: 0`, `pin_memory: False` (기본값)

**개선 방법**:
```yaml
dataloader:
  batch_size: 4
  num_workers: 4  # CPU 코어 수에 맞게 조정 (2-8 권장)
  pin_memory: True  # GPU 전송 속도 향상
  persistent_workers: True  # Worker 재사용으로 오버헤드 감소
```

**예상 효과**: 데이터 로딩 시간 20-40% 감소

**주의사항**:
- `num_workers`는 CPU 코어 수를 초과하지 않도록 설정
- 메모리 사용량이 약간 증가할 수 있음

**코드 위치**: `federatedscope/core/auxiliaries/dataloader_builder.py`

---

### 2. Evaluation 빈도 줄이기 (즉시 적용 가능, 영향: 큼)

**현재 상태**: `eval.freq: 2` (매 2 round마다 평가)

**개선 방법**:
```yaml
eval:
  freq: 5  # 매 5 round마다 평가 (또는 10)
```

**예상 효과**: Evaluation 시간 50-70% 감소 (전체 트레이닝 시간의 10-20% 단축)

**주의사항**:
- 모델 성능 모니터링 빈도가 줄어듦
- 최종 성능에는 영향 없음

**코드 위치**: `federatedscope/core/workers/server.py:355`

---

### 3. Mixed Precision Training 확인 및 최적화 (검토 필요, 영향: 중간)

**현재 상태**: `train.is_enable_half: True`로 설정되어 있지만, 실제 구현 확인 필요

**확인 사항**:
- `torch.autocast` 또는 `torch.cuda.amp` 사용 여부
- `GradScaler` 사용 여부

**개선 방법** (코드 수정 필요):
```python
# federatedscope/llm/trainer/trainer.py 또는 vpl_reward_choice_trainer.py
from torch.cuda.amp import autocast, GradScaler

# __init__에서
if config.train.is_enable_half:
    self.scaler = GradScaler()

# _hook_on_batch_forward에서
with autocast():
    outputs = model(...)
    loss = ...
    
# _hook_on_batch_backward에서
self.scaler.scale(loss).backward()
self.scaler.step(optimizer)
self.scaler.update()
```

**예상 효과**: Forward/Backward pass 속도 30-50% 향상, 메모리 사용량 50% 감소

**주의사항**:
- FP16은 수치적 불안정성을 일으킬 수 있음
- BF16이 더 안정적 (최신 GPU에서 지원)

**코드 위치**: `federatedscope/llm/trainer/trainer.py`, `federatedscope/llm/trainer/vpl_reward_choice_trainer.py`

---

### 4. WandB 로깅 최적화 (즉시 적용 가능, 영향: 작음)

**현재 상태**: 동기식 로깅 사용

**개선 방법**:
```python
# federatedscope/llm/trainer/vpl_reward_choice_trainer.py
# wandb.log() 호출을 배치로 모아서 한 번에 로깅
wandb_metrics_batch = {}
# ... 여러 메트릭 수집 ...
wandb.log(wandb_metrics_batch, step=step)  # 한 번만 호출
```

또는 비동기 로깅:
```python
import wandb
wandb.init(..., settings=wandb.Settings(_disable_stats=True))
```

**예상 효과**: 로깅 오버헤드 10-20% 감소

**코드 위치**: `federatedscope/llm/trainer/vpl_reward_choice_trainer.py:573`, `federatedscope/llm/llm_local/server.py:301`

---

### 5. Gradient Accumulation 최적화 (검토 필요, 영향: 작음)

**현재 상태**: `grad_accum_step: 2`

**개선 방법**:
- Effective batch size를 유지하면서 `batch_size`를 늘리고 `grad_accum_step`을 줄이기
- 예: `batch_size: 8`, `grad_accum_step: 1` (effective batch size = 8)

**예상 효과**: Gradient accumulation 오버헤드 감소 (5-10% 속도 향상)

**주의사항**:
- GPU 메모리가 충분해야 함
- Effective batch size는 동일하게 유지

---

### 6. torch.compile 사용 (PyTorch 2.0+, 영향: 중간-큼)

**현재 상태**: 사용되지 않음

**개선 방법**:
```python
# federatedscope/llm/trainer/trainer.py 또는 vpl_reward_choice_trainer.py
# __init__에서
if hasattr(torch, 'compile'):
    self.model = torch.compile(self.model, mode='reduce-overhead')
```

**예상 효과**: Forward pass 속도 20-30% 향상 (첫 실행 후)

**주의사항**:
- PyTorch 2.0+ 필요
- 첫 실행 시 컴파일 시간 소요
- 일부 연산에서 호환성 문제 가능

**코드 위치**: `federatedscope/llm/trainer/trainer.py`, `federatedscope/llm/trainer/vpl_reward_choice_trainer.py`

---

### 7. 불필요한 계산 제거 (코드 검토 필요, 영향: 작음-중간)

**확인 사항**:
- `ctx.data_batch`를 CPU로 이동하는 부분 (`cpu()` 호출) - 이미 구현됨
- 불필요한 gradient 계산 (`torch.no_grad()` 사용)
- 중복된 forward pass

**개선 방법**:
```python
# Evaluation 시 no_grad 사용 확인
@torch.no_grad()
def evaluate(...):
    ...

# 불필요한 텐서 복사 제거
# 예: ctx.data_batch['input_ids'].cpu() 대신 del 사용
```

**예상 효과**: 메모리 사용량 감소, 약간의 속도 향상

---

### 8. Checkpoint 저장 빈도 조정 (즉시 적용 가능, 영향: 작음)

**현재 상태**: `save_freq: 10`

**개선 방법**:
```yaml
federate:
  save_freq: 20  # 또는 50 (더 긴 간격)
```

**예상 효과**: I/O 시간 감소 (전체 시간의 1-5% 단축)

**주의사항**:
- 체크포인트 손실 위험 증가
- 최종 모델은 항상 저장됨

---

## 우선순위별 적용 가이드

### 즉시 적용 가능 (Config만 수정)
1. ✅ **DataLoader 최적화** (`num_workers`, `pin_memory`)
2. ✅ **Evaluation 빈도 줄이기** (`eval.freq`)
3. ✅ **Checkpoint 저장 빈도 조정** (`save_freq`)

### 코드 수정 필요 (중간 우선순위)
4. ⚠️ **Mixed Precision 확인 및 최적화** (AMP 사용)
5. ⚠️ **WandB 로깅 최적화** (배치 로깅)
6. ⚠️ **torch.compile 적용** (PyTorch 2.0+)

### 검토 필요 (낮은 우선순위)
7. ⚠️ **Gradient Accumulation 최적화**
8. ⚠️ **불필요한 계산 제거**

## 예상 전체 효과

모든 최적화를 적용했을 때:
- **데이터 로딩**: 20-40% 빠름
- **Evaluation**: 50-70% 시간 감소
- **Forward/Backward**: 20-50% 빠름 (Mixed Precision + compile)
- **전체 트레이닝 시간**: **30-50% 단축** 예상

## 적용 예시

### Config 파일 수정 (`cfg/vpl-gp/hhst-60000.yaml`)
```yaml
dataloader:
  batch_size: 4
  num_workers: 4  # 추가
  pin_memory: True  # 추가
  persistent_workers: True  # 추가

eval:
  freq: 5  # 2에서 5로 변경

federate:
  save_freq: 20  # 10에서 20으로 변경 (선택사항)
```

## 참고사항

- 모든 최적화는 점진적으로 적용하는 것을 권장
- 각 최적화 후 성능과 속도를 모니터링
- GPU 메모리 사용량도 함께 확인
- 트레이닝 품질 (loss, accuracy)에 영향이 없는지 확인
