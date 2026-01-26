# Main Table Experiment Plan

## 개요

이 문서는 논문의 Main Table (Table 1)을 위한 실험 계획을 설명합니다. HH-RLHF 데이터셋에서 다양한 클라이언트 수와 모델에 대해 여러 방법들을 비교 평가합니다.

## 실험 목표

**Main Table**: GPT-4 Win-rate (%) 비교
- **데이터셋**: HH-RLHF
- **평가 지표**: Helpful Win-rate, Harmless Win-rate
- **클라이언트 수**: N ∈ {10, 50, 100}
- **모델**: Qwen 2, Gemma-2B
- **방법**: FedDPO, FedBiscuit, FedVPL, **FedVPA-GP** (우리 방법)

## 실험 구조

각 실험은 **2단계**로 구성됩니다:

1. **Selector Training (HHST)**: Binary preference classification 학습
   - 입력: (prompt, response_A, response_B)
   - 출력: 더 나은 response 선택 (A 또는 B)
   - 목적: Preference selector 모델 학습

2. **RL Training (HRL)**: Selector checkpoint를 사용한 RLHF 학습
   - Selector에서 학습한 preference를 reward로 사용
   - Policy 모델을 RL로 fine-tuning
   - 최종 평가: GPT-4 Win-rate

## 실험 매트릭스

### 총 실험 수
- **2 models** × **4 methods** × **3 client counts** = **24 experiments**
- 각 실험은 Selector + RL 2단계이므로 총 **48개 실험** (Selector 24개 + RL 24개)

### 실험 조합

| Model | Method | Client Count | Selector TID | RL TID |
|-------|--------|--------------|--------------|--------|
| Qwen 2 | FedDPO | 10 | 62200 | 63200 |
| Qwen 2 | FedDPO | 50 | 62201 | 63201 |
| Qwen 2 | FedDPO | 100 | 62202 | 63202 |
| Qwen 2 | FedBiscuit | 10 | 62210 | 63210 |
| Qwen 2 | FedBiscuit | 50 | 62211 | 63211 |
| Qwen 2 | FedBiscuit | 100 | 62212 | 63212 |
| Qwen 2 | FedVPL | 10 | 62220 | 63220 |
| Qwen 2 | FedVPL | 50 | 62221 | 63221 |
| Qwen 2 | FedVPL | 100 | 62222 | 63222 |
| Qwen 2 | FedVPA-GP | 10 | 62230 | 63230 |
| Qwen 2 | FedVPA-GP | 50 | 62231 | 63231 |
| Qwen 2 | FedVPA-GP | 100 | 62232 | 63232 |
| Gemma-2B | FedDPO | 10 | 62100 | 63100 |
| Gemma-2B | FedDPO | 50 | 62101 | 63101 |
| Gemma-2B | FedDPO | 100 | 62102 | 63102 |
| Gemma-2B | FedBiscuit | 10 | 62110 | 63110 |
| Gemma-2B | FedBiscuit | 50 | 62111 | 63111 |
| Gemma-2B | FedBiscuit | 100 | 62112 | 63112 |
| Gemma-2B | FedVPL | 10 | 62120 | 63120 |
| Gemma-2B | FedVPL | 50 | 62121 | 63121 |
| Gemma-2B | FedVPL | 100 | 62122 | 63122 |
| Gemma-2B | FedVPA-GP | 10 | 62130 | 63130 |
| Gemma-2B | FedVPA-GP | 50 | 62131 | 63131 |
| Gemma-2B | FedVPA-GP | 100 | 62132 | 63132 |

## 방법별 설정

### 1. FedDPO (Baseline)
- **Trainer**: `llmdporewardchoicetrainer` (DPO-based)
- **특징**: Direct Preference Optimization, 개인화 없음
- **Config**: `cfg/feddpo/hhst.yaml`, `cfg/feddpo/hrl.yaml`

### 2. FedBiscuit (Baseline)
- **Trainer**: `llmrewardchoicetrainer` (Reward model)
- **특징**: Multiple LoRA adapters (U=3), adapter grouping
- **Config**: `cfg/fedbiscuit/hhst.yaml`, `cfg/fedbiscuit/hrl.yaml`

### 3. FedVPL (Baseline)
- **Trainer**: `vplrewardchoicetrainer` (VPL without GP prior)
- **특징**: Variational Preference Learning, standard normal prior
- **Config**: `cfg/vpl/hhst.yaml`, `cfg/vpl/hrl.yaml`

### 4. FedVPA-GP (Our Method)
- **Trainer**: `vplgprewardchoicetrainer` (VPL-GP)
- **특징**: 
  - Variational Preference Learning with Gumbel-Softmax Prior
  - Orthogonal loss for preference disentanglement
  - Feature difference embedding
- **Config**: `cfg/vpl-gp/hhst.yaml`, `cfg/vpl-gp/hrl.yaml`
- **하이퍼파라미터** (하이퍼파라미터 서치 결과 기반):
  - `vpl_orthogonal_weight: 1.0`
  - `vpl_prototype_scale: 5.0`
  - `vpl_kl_weight: 0.1`
  - `vpl_gp_temperature: 1.0`
  - `lr: 0.0001`

## 클라이언트 수별 설정

### N = 10 Clients
```yaml
federate:
  client_num: 10
  sample_client_num: 5  # 50% sampling rate
```

### N = 50 Clients
```yaml
federate:
  client_num: 50
  sample_client_num: 10  # 20% sampling rate
```

### N = 100 Clients
```yaml
federate:
  client_num: 100
  sample_client_num: 10  # 10 clients per round (same as N=10)
```

## 공통 설정

### Selector Training (HHST)
- **Dataset**: HH-RLHF
- **Total Rounds**: 50
- **Local Update Steps**: 30
- **Batch Size**: 8-16 (모델에 따라 조정)
- **Learning Rate**: 0.0001 (모델에 따라 조정)
- **Evaluation**: Every 5 rounds
- **Metrics**: loss, acc, vpl_kl_loss (VPL methods)

### RL Training (HRL)
- **Dataset**: HH-RLHF
- **Total Rounds**: 50
- **Local Update Steps**: 30
- **Batch Size**: 1
- **Gradient Accumulation**: 4-32 (effective batch size 유지)
- **Learning Rate**: 0.0001
- **Reward Coefficient**: 0.1
- **Max Prompts for Generation**: 50
- **Evaluation**: Every round
- **Metrics**: 
  - loss, acc
  - avg_helpfulness, avg_harmlessness
  - helpfulness_winrate, harmlessness_winrate
  - avg_winlose_rate

### 평가 설정
- **GPT-4 Win-rate Evaluation**:
  - `use_gpt_api_for_winrate: true`
  - `openai_model: gpt-4o-mini` (비용 절감)
  - `use_baseline_model_for_winrate: true`
  - `max_samples_for_reward: 30` (평가 속도 향상)

## 모델별 설정

### Qwen 2
- **Model**: `Qwen/Qwen2-0.5B@huggingface_llm`
- **Learning Rate**: 0.00001 (Qwen 2에 최적화)
- **Batch Size**: 16 (Selector), 1 (RL)
- **Gradient Accumulation**: 1 (Selector), 32 (RL)

### Gemma-2B
- **Model**: `google/gemma-2b@huggingface_llm`
- **Learning Rate**: 0.0001 (Gemma-2B에 최적화)
- **Batch Size**: 8-16 (Selector), 1 (RL)
- **Gradient Accumulation**: 2-4 (Selector), 4-32 (RL)

## 실험 실행 순서

### Phase 1: Selector Training (24 experiments)
1. 모든 방법과 모델, 클라이언트 수 조합에 대해 Selector 학습
2. 각 실험은 독립적으로 실행 가능 (병렬 실행 권장)
3. 완료 후 checkpoint 저장: `final_hhrl_choice_*_t{TID}.ckpt`

### Phase 2: RL Training (24 experiments)
1. Phase 1에서 완료된 Selector checkpoint 사용
2. 각 Selector에 대응하는 RL 실험 실행
3. RL 실험은 Selector 완료 후 순차 실행

### Phase 3: Evaluation & Results Collection
1. 모든 RL 실험 완료 후 GPT-4 Win-rate 수집
2. WandB에서 metrics 추출
3. Main table 작성

## 파일 구조

```
cfg/
├── main_table/
│   ├── qwen2/
│   │   ├── feddpo/
│   │   │   ├── hhst_n10_60000.yaml
│   │   │   ├── hhst_n50_60001.yaml
│   │   │   ├── hhst_n100_60002.yaml
│   │   │   ├── hrl_n10_70000.yaml
│   │   │   ├── hrl_n50_70001.yaml
│   │   │   └── hrl_n100_70002.yaml
│   │   ├── fedbiscuit/
│   │   ├── fedvpl/
│   │   └── fedvpagp/
│   └── gemma2b/
│       ├── feddpo/
│       ├── fedbiscuit/
│       ├── fedvpl/
│       └── fedvpagp/
scripts/
└── main_table/
    ├── qwen2/
    └── gemma2b/
```

## 체크포인트 위치

### Selector Checkpoints
- 경로: `/hdd/hdd3/kjb/checkpoints/`
- 파일명: `final_hhrl_choice_{model}_fedbiscuit_u3_{method}_t{TID}.ckpt`

### RL Checkpoints
- 경로: `/hdd/hdd3/kjb/checkpoints/`
- 파일명: `hhrl_rlhf_{model}_choice_{method}_t{TID}.ckpt`

## WandB 프로젝트

- **Selector 실험**: `fvpl-selector-main-table`
- **RL 실험**: `fvpl-rl-main-table`

각 실험의 이름은 `{method}_{model}_n{client_count}_t{TID}` 형식입니다.

## 예상 실행 시간

### Selector Training
- Qwen 2: ~2-3 hours per experiment
- Gemma-2B: ~3-4 hours per experiment
- 총 24개 실험: ~60-80 hours (병렬 실행 시 GPU 수에 따라 단축)

### RL Training
- Qwen 2: ~4-6 hours per experiment
- Gemma-2B: ~5-7 hours per experiment
- 총 24개 실험: ~120-150 hours (병렬 실행 시 GPU 수에 따라 단축)

### 총 예상 시간
- **Sequential**: ~180-230 hours (~7.5-9.5 days)
- **Parallel (8 GPUs)**: ~22-30 hours (~1-1.25 days)

## GPU 할당 전략

### 권장 GPU 할당
- **GPU 0-1**: Qwen 2 실험 (작은 모델, 빠른 실행)
- **GPU 2-5**: Gemma-2B 실험 (큰 모델, 더 많은 메모리 필요)
- **GPU 6-7**: RL 실험 (메모리 집약적)

### 병렬 실행 전략
1. **Selector 우선**: 모든 Selector 실험을 먼저 완료
2. **RL 순차 실행**: Selector 완료 후 RL 실험 실행 (OOM 방지)
3. **클라이언트 수별 그룹화**: 같은 클라이언트 수 실험을 그룹으로 실행

## 결과 수집 및 분석

### WandB에서 수집할 Metrics

#### Selector Metrics
- `train_avg_loss`: Training loss
- `test_avg_loss`: Test loss
- `train_avg_acc`: Training accuracy
- `test_avg_acc`: Test accuracy
- `vpl_kl_loss`: KL divergence (VPL methods)
- `vpl_orthogonal_loss`: Orthogonal loss (FedVPA-GP)

#### RL Metrics
- `avg_helpfulness`: Average helpfulness score
- `avg_harmlessness`: Average harmlessness score
- `helpfulness_winrate`: Helpful response win rate (%)
- `harmlessness_winrate`: Harmless response win rate (%)
- `avg_winlose_rate`: Overall win-lose rate

### Main Table 작성
1. 각 실험의 최종 round (Round 50) metrics 수집
2. Helpful/Harmless Win-rate 추출
3. 클라이언트 수별로 그룹화
4. 방법별 비교 및 통계 분석

## 주의사항

1. **메모리 관리**: RL 실험은 메모리 사용량이 크므로 GPU당 하나씩만 실행
2. **체크포인트 확인**: RL 실험 실행 전 해당 Selector checkpoint 존재 확인
3. **WandB 동기화**: 실험 결과는 자동 업로드되지만 네트워크 문제 시 수동 동기화 필요
4. **재현성**: 모든 실험에 동일한 seed 사용 권장
5. **로그 백업**: 중요한 실험의 로그는 별도로 백업

## 다음 단계

1. **Config 파일 생성**: 모든 실험 조합에 대한 config 파일 생성
2. **Script 파일 생성**: 실행 스크립트 생성
3. **실험 실행**: Phase 1 (Selector) → Phase 2 (RL) 순서로 실행
4. **결과 수집**: WandB에서 metrics 추출
5. **Main Table 작성**: 결과를 테이블 형식으로 정리

## 참고사항

- 하이퍼파라미터 서치 결과를 바탕으로 FedVPA-GP의 최적 하이퍼파라미터 사용
- Baseline 방법들은 기존 구현 사용
- 모든 실험은 동일한 데이터셋과 split 사용 (재현성 보장)
