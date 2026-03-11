# Local RL Training Only Experiments

로컬 서버에서 selector checkpoint 없이 RL training만 실행하는 baseline 실험입니다.

## 실험 설정

| TID | Client Num | Sample Client Num | GPU | Status |
|-----|------------|-------------------|-----|--------|
| 01000 | 10 | 5 | 2 | Not started |
| 01001 | 50 | 10 | 3 | Not started |
| 01002 | 100 | 10 | 4 | Not started |

## 특징

- Selector checkpoint 불필요 (FedDPO 방식)
- **`--selector-cfg-file` 미사용**: selector를 로드하지 않아 **모델 1개만 사용** (OOM 방지)
- `rlhf_use_variational_selection: False` - Variational selection 사용 안 함
- FedDPO와 유사한 방식으로 직접 preference data에서 학습 (policy 모델로 generation + selection 모두 수행)
- 로컬 서버에서 실행 (SLURM cluster 아님, nohup으로 백그라운드 실행)
- `scripts/feddpo/hrl-10000.sh` 스크립트를 참고하여 작성

## 파일 구조

```
scripts/server/local/
├── README.md (이 파일)
├── run_rl_local.sh        # 개별 실험 실행 스크립트
└── run_all_local_rl.sh    # 모든 실험 실행 스크립트
```

## 사용 방법

### 개별 실험 실행

```bash
# TID, 클라이언트 수, GPU ID를 인자로 전달
bash scripts/server/local/run_rl_local.sh 01000 10 2
bash scripts/server/local/run_rl_local.sh 01001 50 3
bash scripts/server/local/run_rl_local.sh 01002 100 4
```

### 모든 실험 한번에 실행

```bash
# 모든 실험을 순차적으로 시작
bash scripts/server/local/run_all_local_rl.sh
```

## 공통 설정

| 파라미터 | 값 |
|---------|-----|
| Model | `Qwen/Qwen2-0.5B@huggingface_llm` |
| Trainer | `llmdporewardtrainer` (DPO trainer) |
| `rlhf_use_variational_selection` | `False` |
| `rlhf_use_variational_generation` | `False` |
| `reward_coeff` | 0.1 |
| `grad_accum_step` | 4 |
| `max_prompts_for_generation` | 50 |
| `generation_batch_size` | 3 |
| `use_gpt_api_for_winrate` | `True` |
| `use_baseline_model_for_winrate` | `True` |
| `openai_model` | `gpt-4o-mini` |
| Learning rate | 0.00001 (Qwen2) |
| Total rounds | 50 |
| Local update steps | 30 |
| Batch size | 1 |

## 모니터링

### 로그 확인
```bash
# 실시간 로그 확인
tail -f outputs/01000.log
tail -f outputs/01001.log
tail -f outputs/01002.log
```

### GPU 사용량 확인
```bash
nvidia-smi
```

### 프로세스 확인
```bash
ps aux | grep "01000\|01001\|01002" | grep python
```

## 체크포인트 위치

- `/hdd/hdd3/kjb/checkpoints/hhrl_rlhf_qwen2_choice_local_only_t01000.ckpt`
- `/hdd/hdd3/kjb/checkpoints/hhrl_rlhf_qwen2_choice_local_only_t01001.ckpt`
- `/hdd/hdd3/kjb/checkpoints/hhrl_rlhf_qwen2_choice_local_only_t01002.ckpt`

## 평가 지표

- `avg_helpfulness`: Helpfulness score
- `avg_harmlessness`: Harmlessness score
- `helpfulness_winrate`: Helpful response win rate
- `harmlessness_winrate`: Harmless response win rate
- `avg_winlose_rate`: Overall win-lose rate

**Winrate 계산**: `use_gpt_api_for_winrate: true`이면 GPT API(gpt-4o-mini)로 비교합니다.  
`openai` 미설치 또는 `OPENAI_API_KEY` 미설정 시 **내부 모델**로 자동 fallback 되어 0점이 되지 않습니다.  

**GPT API 사용 시**: 프로젝트 루트에 **`.env`** 파일을 만들고 아래 한 줄을 넣으세요 (실제 키로 교체).
```bash
# 프로젝트 루트: /home/kjb/ppfl/.env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxx
```
`.env.example`을 복사해 써도 됩니다: `cp .env.example .env` 후 `OPENAI_API_KEY` 값만 수정.  
또한 `pip install openai` 필요.

## 비교 목적

- Selector training이 있는 경우와 없는 경우의 성능 비교
- Selector의 기여도 정량화
- FedDPO와 동일한 방식으로 직접 preference learning 수행
- 클라이언트 수에 따른 성능 변화 분석
