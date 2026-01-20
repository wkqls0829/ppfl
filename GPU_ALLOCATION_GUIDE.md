# GPU 할당 가이드

## 개요

실험 60000은 멀티 GPU를 지원하도록 설정되었습니다. Config 파일에서 `federate.process_num`을 설정하여 사용할 GPU 개수를 지정할 수 있습니다.

## 설정 방법

### 1. Config 파일에서 설정

#### hhst.yaml / hhst-ortho.yaml

```yaml
use_gpu: True
device: 0  # Base device, will be overridden by process_num
backend: 'torch'
federate:
  mode: standalone
  share_local_model: False  # Required for multi-GPU parallel training
  process_num: 8  # Number of GPUs to use (1 server + 7 clients)
```

**중요 설정**:
- `share_local_model: False`: 멀티 GPU 사용을 위해 필수
- `process_num`: 사용할 GPU 개수
  - 1개 = 서버만 (단일 GPU)
  - 2개 이상 = 서버 1개 + 클라이언트 (process_num - 1)개
  - 예: `process_num: 8` → 서버 1개 + 클라이언트 7개

### 2. 스크립트에서 자동 감지

스크립트는 자동으로 사용 가능한 GPU 개수를 감지하고 `process_num`을 설정합니다:

```bash
# 자동으로 GPU 개수 감지
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
```

**동작 방식**:
1. `nvidia-smi`로 GPU 개수 감지
2. 감지된 GPU 개수로 `process_num` 자동 설정
3. GPU를 감지할 수 없으면 config 파일의 기본값 사용

## GPU 할당 방식

### StandaloneMultiGPURunner

`process_num > 1`일 때 `StandaloneMultiGPURunner`가 사용됩니다:

```
Process 0 (Rank 0): Server → GPU 0
Process 1 (Rank 1): Client 1-2 → GPU 1
Process 2 (Rank 2): Client 3-4 → GPU 2
Process 3 (Rank 3): Client 5-6 → GPU 3
...
```

**할당 규칙**:
- 서버는 항상 rank 0 (GPU 0)
- 클라이언트는 rank 1부터 순차적으로 할당
- 각 프로세스는 독립적인 GPU를 사용

### 클라이언트 분배

클라이언트는 프로세스 간에 균등하게 분배됩니다:

```python
client_num_per_process = client_num // (process_num - 1)
```

예: `client_num=10`, `process_num=8`
- 프로세스당 클라이언트: 10 // 7 = 1.4 → 1-2개씩 분배
- Process 1: Client 1-2
- Process 2: Client 3-4
- Process 3: Client 5-6
- Process 4: Client 7-8
- Process 5: Client 9-10
- Process 6-7: 사용 안 함 (클라이언트 수 부족)

## 사용 예시

### 예시 1: 8개 GPU 사용

```yaml
federate:
  process_num: 8  # 서버 1개 + 클라이언트 7개
```

**할당**:
- GPU 0: Server
- GPU 1-7: Clients (각 GPU당 1-2개 클라이언트)

### 예시 2: 4개 GPU 사용

```yaml
federate:
  process_num: 4  # 서버 1개 + 클라이언트 3개
```

**할당**:
- GPU 0: Server
- GPU 1-3: Clients (각 GPU당 3-4개 클라이언트)

### 예시 3: 단일 GPU (기본)

```yaml
federate:
  process_num: 1  # 또는 설정 안 함
  share_local_model: True  # 단일 GPU에서는 True 가능
```

**할당**:
- GPU 0: Server + 모든 Clients (순차 실행)

## 주의사항

### 1. share_local_model 설정

- **멀티 GPU (`process_num > 1`)**: `share_local_model: False` 필수
- **단일 GPU (`process_num = 1`)**: `share_local_model: True` 가능

### 2. process_num 제한

- `process_num`은 사용 가능한 GPU 개수보다 작거나 같아야 함
- `process_num > client_num`인 경우 경고 발생 및 자동 조정

### 3. 메모리 사용

각 GPU는 독립적인 프로세스에서 실행되므로:
- 각 GPU의 메모리 사용량 = 모델 크기 + 배치 크기
- GPU 메모리가 부족하면 `batch_size` 또는 `grad_accum_step` 조정

## 실행 방법

### 자동 GPU 감지 (권장)

```bash
bash scripts/vpl-gp/hhst.sh
# 또는
bash scripts/vpl-gp/hhst-ortho.sh
```

스크립트가 자동으로 GPU 개수를 감지하고 설정합니다.

### 수동 설정

Config 파일에서 직접 `process_num`을 지정:

```yaml
federate:
  process_num: 4  # 원하는 GPU 개수
```

## 확인 방법

### 1. GPU 사용 확인

```bash
watch -n 1 nvidia-smi
```

각 GPU가 사용 중인지 확인할 수 있습니다.

### 2. 로그 확인

```bash
tail -f outputs/60000.log
```

로그에서 다음과 같은 메시지를 확인:

```
Multi-GPU are starting for parallel training ...
Detected 8 GPUs. Using process_num=8 for multi-GPU training
```

### 3. 프로세스 확인

```bash
ps aux | grep python | grep federatedscope
```

여러 프로세스가 실행 중인지 확인할 수 있습니다.

## 성능 향상

멀티 GPU 사용 시:
- **병렬 처리**: 여러 클라이언트가 동시에 학습
- **속도 향상**: 단일 GPU 대비 약 3-5배 빠름 (GPU 개수에 따라)
- **리소스 활용**: 모든 GPU를 효율적으로 사용

## 문제 해결

### 문제: "process_num is more than client number"

**원인**: `process_num`이 클라이언트 수보다 많음

**해결**: `process_num`을 클라이언트 수 이하로 설정

### 문제: "CUDA out of memory"

**원인**: GPU 메모리 부족

**해결**:
- `batch_size` 감소
- `grad_accum_step` 증가
- `process_num` 감소 (더 적은 GPU 사용)

### 문제: GPU가 감지되지 않음

**원인**: `nvidia-smi`가 작동하지 않음

**해결**: Config 파일에서 직접 `process_num` 설정

## 참고

- FederatedScope Parallel Runner: `federatedscope/core/parallel/parallel_runner.py`
- GPU Manager: `federatedscope/core/gpu_manager.py`
- Runner Builder: `federatedscope/core/auxiliaries/runner_builder.py`
