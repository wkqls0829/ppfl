# HHST Unseen Client 설정 확인

## 확인 결과

**HHST (selector training)에서 unseen 클라이언트는 제대로 반영되고 있습니다.**

## 설정 확인

### 1. Config 파일
모든 HHST config 파일에 `unseen_clients_id`가 설정되어 있습니다:

- `cfg/fedbiscuit-unseen/hhst.yaml`: `unseen_clients_id: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]`
- `cfg/fedvpl-unseen/hhst.yaml`: `unseen_clients_id: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]`
- `cfg/fedvpl-gp-ortho-unseen/hhst.yaml`: `unseen_clients_id: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]`

### 2. 클라이언트 구성
- `client_num: 20`: 총 20개 클라이언트 생성 (10 harmless + 10 helpful)
- `sample_client_num: 5`: 각 round마다 5개 클라이언트만 샘플링
- `unseen_clients_id: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]`: Unseen 클라이언트 ID 명시

## 동작 메커니즘

### 1. 초기화 단계
1. **`fed_runner.py`** (line 73-77):
   - Config에서 `unseen_clients_id`를 읽어서 `self.unseen_clients_id`에 저장
   - 로그 출력: `"Using directly specified unseen_clients_id: [11, 12, ...]"`

2. **`_setup_server`** (line 174):
   - Server 초기화 시 `unseen_clients_id=self.unseen_clients_id` 전달

3. **`Server.__init__`** (line 178-179):
   - `self.unseen_clients_id = [] if unseen_clients_id is None else unseen_clients_id`
   - Server 인스턴스에 unseen 클라이언트 ID 저장

### 2. Training 단계
1. **`broadcast_model_para`** (line 690-711):
   - `filter_unseen_clients=True` (기본값)
   - `self.sampler.change_state(self.unseen_clients_id, 'unseen')` 호출
   - Unseen 클라이언트를 sampler에서 제외

2. **Sampling**:
   - `sample_client_num=5`일 때, unseen 클라이언트(11-20)는 샘플링에서 제외
   - Seen 클라이언트(1-10) 중에서만 5개 샘플링

3. **복원** (line 777-779):
   - Broadcasting 후 `self.sampler.change_state(self.unseen_clients_id, 'seen')` 호출
   - Unseen 클라이언트 상태를 원래대로 복원

## 확인 방법

### 로그 확인
HHST 실행 시 다음 로그가 출력되어야 합니다:
```
Using directly specified unseen_clients_id: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
```

### 실제 동작 확인
1. **Sampling 확인**: 각 round에서 샘플링된 클라이언트 ID가 1-10 범위에만 있어야 합니다.
2. **Unseen 클라이언트 확인**: 클라이언트 11-20은 training에 참여하지 않아야 합니다.

## 요약

- ✅ **Config 설정**: 모든 HHST config에 `unseen_clients_id` 설정됨
- ✅ **초기화**: `fed_runner.py`에서 config에서 읽어서 Server에 전달
- ✅ **필터링**: `broadcast_model_para`에서 unseen 클라이언트를 sampler에서 제외
- ✅ **동작**: Seen 클라이언트(1-10)만 training에 참여, Unseen 클라이언트(11-20)는 제외

**결론**: HHST에서 unseen 클라이언트는 제대로 반영되고 있습니다.
