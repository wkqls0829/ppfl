# Unseen Experiment Logging Fix

## 문제점

70001.log (FedDPO unseen experiment)에서 unseen test winrate가 WandB에 로깅되지 않았습니다.

## 원인

1. **Config 파일에 `unseen_clients_id` 누락**: 
   - `cfg/feddpo-unseen/hrl.yaml`에 `unseen_clients_id`가 설정되어 있지 않았습니다.
   - 다른 RL config 파일들(`fedbiscuit-unseen/hrl.yaml`, `fedvpl-unseen/hrl.yaml`, `fedvpl-gp-ortho-unseen/hrl.yaml`)도 동일하게 누락되어 있었습니다.

2. **코드에서 config 확인 누락**:
   - `standalone_training.py`에서 `unseen_clients_id`를 `selector_cfg`에서만 가져오고 있었습니다.
   - FedDPO는 selector가 없으므로 `selector_cfg`가 `None`이고, `unseen_clients_id`가 빈 리스트가 됩니다.

## 수정 사항

### 1. Config 파일 수정

모든 unseen RL config 파일에 `unseen_clients_id` 추가:

- `cfg/feddpo-unseen/hrl.yaml`
- `cfg/fedbiscuit-unseen/hrl.yaml`
- `cfg/fedvpl-unseen/hrl.yaml`
- `cfg/fedvpl-gp-ortho-unseen/hrl.yaml`

```yaml
federate:
  client_num: 20
  total_round_num: 50
  unseen_clients_id: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # Unseen clients (not in selector training)
```

### 2. 코드 수정

`federatedscope/llm/rlhf/standalone_training.py` (line 1595-1598):

**변경 전**:
```python
unseen_clients_id = []
if self.selector_cfg is not None:
    unseen_clients_id = getattr(self.selector_cfg.federate, 'unseen_clients_id', [])
```

**변경 후**:
```python
unseen_clients_id = []
if self.selector_cfg is not None:
    unseen_clients_id = getattr(self.selector_cfg.federate, 'unseen_clients_id', [])
# Fallback to main config if selector_cfg is None (e.g., FedDPO)
if len(unseen_clients_id) == 0:
    unseen_clients_id = getattr(self.config.federate, 'unseen_clients_id', [])
```

## 로깅 형식

Unseen experiment에서 test evaluation 시 다음 형식으로 WandB에 로깅됩니다:

### Seen 클라이언트 (1-10)
- `Server, {metric}`: 표준 형식 (기존 호환성)
- `Server_Seen, {metric}`: Seen 전용 형식

### Unseen 클라이언트 (11-20)
- `Server_Unseen, {metric}`: Unseen 전용 형식

### 예시
- `Server, test_helpfulness_winrate`: Seen 클라이언트의 helpfulness winrate
- `Server_Seen, test_helpfulness_winrate`: Seen 클라이언트의 helpfulness winrate (명시적)
- `Server_Unseen, test_helpfulness_winrate`: Unseen 클라이언트의 helpfulness winrate

## 확인 방법

1. **로그 확인**: 
   ```bash
   grep "Unseen experiment detected" outputs/70001.log
   grep "Server_Unseen" outputs/70001.log
   ```

2. **WandB 확인**:
   - WandB 프로젝트 `fvpl-unseen-rl`에서 `Server_Unseen, test_helpfulness_winrate` 및 `Server_Unseen, test_harmlessness_winrate` 메트릭 확인

## 영향받는 실험

다음 실험들이 이 수정의 영향을 받습니다:
- **TID 70001** (FedDPO): 재실행 필요
- **TID 70004** (FedBiscuit RL): 재실행 필요
- **TID 70005** (FedVPL RL): 재실행 필요
- **TID 70006** (FedVPL-GP-Ortho RL): 재실행 필요

## 주의사항

1. **기존 실행 중인 실험**: 이미 실행 중인 실험은 이 수정의 영향을 받지 않습니다. 재실행이 필요합니다.

2. **Selector 기반 알고리즘**: FedBiscuit, FedVPL, FedVPL-GP-Ortho는 `selector_cfg`에서 `unseen_clients_id`를 가져올 수 있지만, config에도 설정하는 것이 안전합니다.

3. **FedDPO**: Selector가 없으므로 반드시 config에 `unseen_clients_id`를 설정해야 합니다.
