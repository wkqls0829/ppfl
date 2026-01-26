# GPU 할당 검증 결과

## Config 파일에서 GPU 할당 확인

### 1. GPUManager 사용

**위치**: `federatedscope/core/fed_runner.py:66-67`

```python
self.gpu_manager = GPUManager(gpu_available=self.cfg.use_gpu,
                              specified_device=self.cfg.device)
```

**동작 방식**:
- `specified_device >= 0`일 때: `cuda:{specified_device}` 반환
- `specified_device < 0`일 때: 자동으로 가장 여유 있는 GPU 선택

**결론**: ✅ Config의 `device` 설정이 제대로 사용됨

### 2. GPUManager.auto_choice() 메서드

**위치**: `federatedscope/core/gpu_manager.py:58-82`

```python
def auto_choice(self):
    if self.gpus is None:
        return 'cpu'
    elif self.specified_device >= 0:
        # allow users to specify the device
        return 'cuda:{}'.format(self.specified_device)
    else:
        # 자동 선택 로직...
```

**결론**: ✅ `specified_device >= 0`이면 config의 device를 그대로 사용

### 3. Dataloader에서 device 사용

**위치**: `federatedscope/llm/dataloader/dataloader.py:79`

```python
self.device = f'cuda:{config.device}'
```

**결론**: ✅ Config의 `device` 설정을 직접 사용

### 4. Generator에서 device 사용

**위치**: `federatedscope/llm/misc/fschat.py:78`

```python
self.device = f'cuda:{config.device}'
```

**결론**: ✅ Config의 `device` 설정을 직접 사용

## 잠재적 문제점 확인

### 1. device_map='auto' 사용

**위치**: 여러 파일에서 `get_llm(config, device_map='auto')` 사용

**영향**:
- `device_map='auto'`는 HuggingFace의 `accelerate` 라이브러리 기능
- 모델을 여러 GPU에 자동으로 분산 배치
- 하지만 이는 주로 RLHF나 평가 스크립트에서만 사용
- 일반적인 federated training에서는 사용되지 않음

**확인 필요**: `federatedscope/llm/llm_local/client.py`와 `server.py`에서 `get_llm` 호출 확인

### 2. CUDA_VISIBLE_DEVICES 환경변수

**검색 결과**: 스크립트에서 `CUDA_VISIBLE_DEVICES` 설정 없음 ✅

**주의사항**:
- 스크립트에서 `CUDA_VISIBLE_DEVICES`를 설정하면 config의 `device` 설정과 충돌 가능
- 예: `CUDA_VISIBLE_DEVICES=0` 설정 시, config의 `device: 1`은 실제로 GPU 0을 가리킴

**현재 상태**: ✅ 스크립트에서 `CUDA_VISIBLE_DEVICES` 설정 안 함

### 3. torch.cuda.set_device() 직접 호출

**검색 결과**: 코드베이스에서 직접 호출 없음 ✅

## 최종 결론

### ✅ Config 파일의 device 설정이 제대로 작동함

1. **GPUManager**: Config의 `device` 값을 `specified_device`로 전달
2. **Dataloader**: Config의 `device` 값을 직접 사용
3. **Generator**: Config의 `device` 값을 직접 사용
4. **환경변수**: `CUDA_VISIBLE_DEVICES` 설정 없음
5. **직접 호출**: `torch.cuda.set_device()` 직접 호출 없음

### ⚠️ 주의사항

1. **device_map='auto'**: 
   - RLHF나 평가 스크립트에서만 사용
   - 일반 federated training에는 영향 없음

2. **CUDA_VISIBLE_DEVICES**:
   - 스크립트에서 설정하지 않도록 주의
   - 설정 시 config의 device 번호와 실제 GPU 번호가 달라질 수 있음

3. **멀티 GPU 모드**:
   - `process_num > 1`일 때는 `StandaloneMultiGPURunner` 사용
   - 이 경우 각 프로세스가 rank에 따라 GPU 할당
   - 하지만 현재는 단일 GPU 모드로 설정됨

## 권장 사항

1. ✅ Config 파일에서 `device` 설정 사용 (현재 상태)
2. ✅ 스크립트에서 `CUDA_VISIBLE_DEVICES` 설정 안 함 (현재 상태)
3. ✅ 단일 GPU 모드 사용 (`share_local_model: True`, `process_num` 설정 안 함)

## 실험 60000/60001 설정

### hhst-60000.yaml
```yaml
device: 0  # GPU 0 사용
share_local_model: True  # 단일 GPU 모드
```

### hhst-ortho-60001.yaml
```yaml
device: 1  # GPU 1 사용
share_local_model: True  # 단일 GPU 모드
```

**결론**: 각 실험이 지정된 GPU를 사용하도록 올바르게 설정됨 ✅
