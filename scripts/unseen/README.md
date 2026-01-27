# Unseen Client Experiment

VPL-GP가 unknown client에 대해서도 adapt가 가능한지 테스트하는 실험입니다.

## 실험 목적

- **연구 질문**: VPL-GP가 training 중에 보지 못한 클라이언트에 대해서도 적응할 수 있는가?
- **가설**: Gumbel-Softmax prior를 통해 학습된 latent distribution이 unseen client의 preference도 잘 표현할 수 있을 것이다.

## 실험 구성

### 클라이언트 구성
- **총 20개 클라이언트**: 10개 harmless + 10개 helpful
- **Training 참여**: 10개만 (5개 harmless + 5개 helpful)
- **Unseen 클라이언트**: 10개 (5개 harmless + 5개 helpful)
- **RL Evaluation**: 20개 모두 참여

### 실험 단계

#### 1. Selector Training (hhst_unseen.sh)
- **클라이언트 수**: 10개만 사용
- **목적**: 처음 10개 클라이언트로 binary selector 학습
- **Config**: `cfg/vpl-gp-unseen/hhst.yaml`
- **TID**: 70000

#### 2. RL Training & Evaluation (hrl_unseen.sh)
- **Training**: 처음 10개 클라이언트만 사용 (standalone RLHF)
- **Evaluation**: 20개 클라이언트 모두 사용
- **목적**: Unseen 클라이언트에 대한 adaptation 성능 측정
- **Config**: `cfg/vpl-gp-unseen/hrl.yaml`
- **TID**: 70001

## 실행 방법

### 1. Selector 학습
```bash
cd /home/kjb/ppfl
bash scripts/unseen/hhst_unseen.sh
```

### 2. RL 학습/평가 (Selector 완료 후)
```bash
bash scripts/unseen/hrl_unseen.sh
```

## 하이퍼파라미터

VPL-GP 표준 설정 사용:
- `vpl_use_gp_prior: True`
- `vpl_kl_weight: 0.02`
- `vpl_gp_temperature: 1.0`
- `vpl_orthogonal_weight: 0.0`
- `vpl_latent_dim: 32`

## 구현 주의사항

### 데이터 분할
현재 구현에서는:
- Selector training: `client_num=10`으로 설정하여 10개 클라이언트만 생성
- RL evaluation: `client_num=20`으로 설정하여 20개 클라이언트 생성

**주의**: 이렇게 하면 selector training과 RL evaluation에서 클라이언트 데이터 분할이 다를 수 있습니다.

### 권장 수정사항
완전한 unseen 실험을 위해서는 다음 중 하나를 구현해야 합니다:

1. **Option 1**: Selector training에서도 20개 클라이언트를 생성하되, 처음 10개만 training에 사용
   - `federate.client_num: 20`으로 설정
   - Training 시에는 client_id 1-10만 사용하도록 코드 수정

2. **Option 2**: 데이터 분할을 고정하여 selector와 RL에서 동일한 클라이언트 ID 사용
   - Random seed 고정
   - 동일한 데이터 분할 로직 사용

3. **Option 3**: Custom data loader 생성
   - 20개 클라이언트를 생성하되, 처음 10개는 training에, 나머지 10개는 evaluation에만 사용

## 평가 지표

RL evaluation에서 다음을 비교:
- **Seen clients (1-10)**: Training에 참여한 클라이언트의 성능
- **Unseen clients (11-20)**: Training에 참여하지 않은 클라이언트의 성능

### 예상 결과
- **성공**: Unseen clients의 성능이 seen clients와 유사하면 adaptation 성공
- **실패**: Unseen clients의 성능이 크게 떨어지면 adaptation 실패

## 출력 파일

### Checkpoints
- **Selector**: `checkpoints/hhrl_choice_gemma_unseen_vplgp_t70000.ckpt`
- **RL**: `checkpoints/hhrl_rlhf_gemma_unseen_vplgp_t70001.ckpt`

### Log Files
- **Selector**: `outputs/70000.log`
- **RL**: `outputs/70001.log`

## 모니터링

```bash
# Selector 학습 모니터링
tail -f outputs/70000.log

# RL 학습/평가 모니터링
tail -f outputs/70001.log

# GPU 사용량 확인
watch -n 1 nvidia-smi
```

## 참고사항

- 이 실험은 VPL-GP의 generalization 능력을 테스트합니다
- Unseen client adaptation은 federated learning에서 중요한 문제입니다
- 결과에 따라 VPL-GP의 실용성을 평가할 수 있습니다
