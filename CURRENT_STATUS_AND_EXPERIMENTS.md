# 현재 상태 및 실험 요약 (2026-01-23)

## 최근 변경사항 (Git Commits)

### 최근 10개 커밋
1. **1aefcb5** - Fix test evaluation caching and variational selector choice key bug
2. **c9c1a16** - Fix feature extractor architecture mismatch and standardize HRL configs
3. **a0ef4df** - Fix client_average_z_dict loading and t-SNE visualization for RL training
4. **c5b03ee** - Add client-specific average z for conditional generation in variational RL
5. **5520f17** - Fix device mismatch in RLHF training and update orthogonal loss settings
6. **50bedc9** - Add experiments 40003, 40103, 50003, 50103 with adjusted loss weights
7. **eba3cae** - Add _hook_on_batch_backward override for separate VPL optimizer
8. **0441e05** - Enable separate training for VPL components (MLP + variational encoder)
9. **f9e98fd** - Remove learnable prototype option, keep only fixed prototypes
10. **9e1a20e** - Add support for both fixed and learnable prototypes

## 현재 실행 중인 실험

### 1. FedBiscuit HHST (20124) - GPU 0
- **상태**: 실행 중 (Round 28 진행 중)
- **설정**: 
  - 50 clients, sample 25 per round
  - Baseline (non-VPL) binary selector training
  - 50 rounds total
- **목적**: VPL-GP와 비교를 위한 baseline
- **로그**: `outputs/20124.log` (798KB, 최근 업데이트: 12:09)
- **체크포인트**: `/hdd/hdd3/kjb/checkpoints/hhrl_choice_gemma_fedbiscuit_u3_20124.ckpt`
- **WandB**: `fvpl-selector` 프로젝트

### 2. VPL-GP HHST Ortho (50124) - GPU 1
- **상태**: 실행 중 (Round 11 진행 중)
- **설정**:
  - 50 clients, sample 10 per round
  - VPL-GP with orthogonal loss (weight: 1.0, orthonorm: 0.1)
  - Manual orthogonal labels (harmlessness/helpfulness)
  - 50 rounds total
- **목적**: 50 clients로 확장한 VPL-GP orthogonal 실험
- **로그**: `outputs/50124.log` (608KB, 최근 업데이트: 12:10)
- **체크포인트**: `/hdd/hdd3/kjb/checkpoints/hhrl_choice_gemma_fedbiscuit_u3_vplgp_ortho_50124.ckpt`
- **WandB**: `fvpl-selector` 프로젝트
- **특징**: 
  - KL loss: ~1.2-2.0
  - Orthogonal loss: ~115-116 (dominant)
  - Reconstruction loss: ~0.01-0.05

### 3. VPL-GP HRL Ortho (51024) - GPU 3
- **상태**: 실행 중 (Generation 완료, Training 진행 중)
- **설정**:
  - RL training with VPL-GP orthogonal selector (50024 checkpoint)
  - Standalone mode (client_num=1)
  - 30 rounds total
- **목적**: VPL-GP orthogonal selector를 사용한 RL fine-tuning
- **로그**: `outputs/51024.log` (609KB, 최근 업데이트: 11:00)
- **체크포인트**: `/hdd/hdd3/kjb/checkpoints/hhrl_rlhf_gemma_choice_vplgp_ortho_51024.ckpt`
- **WandB**: `fvpl-rl` 프로젝트
- **특징**:
  - Standard generation (no z-conditional)
  - Dual selection: harmlessness (client 1, z_1) and helpfulness (client 2, z_2)
  - Conflicting selection statistics
  - GPT API winrate evaluation

## 주요 구현사항

### 1. RL Data Generation 및 Selection

#### Generation Process
- **Prompt 사용**: `HH_RLHF_PROMPT_DICT["generation"]`
  ```
  "Below is a conversation between a human and an AI assistant. "
  "Write a response that is both helpful and harmless.\n\n"
  "### CONVERSATION:\n{prompt}\n\n"
  "### RESPONSE:"
  ```
- **다양성 확보**:
  - `temperature=0.7`: 적절한 다양성
  - `top_p=1.0`: 전체 분포 사용
  - `do_sample=True`: Sampling 활성화
  - `num_return_sequences=2`: 각 prompt당 2개 response 생성
- **Standard Generation**: Z-conditional generation 비활성화 (현재 구현)

#### Selection Process
- **전체 response pair에 대해 dual selection**:
  - Harmlessness selection: Client 1의 z (z_1) 사용
  - Helpfulness selection: Client 2의 z (z_2) 사용
- **Conflicting Selection 통계**:
  - 같은 pair에 대해 harmlessness와 helpfulness가 다른 선택을 하는 경우 식별
  - 통계 계산 및 로깅
  - 처음 5개 conflicting 예시 출력

### 2. Client Average Z Dictionary 로드

#### 로드 시점
- **초기 로드**: `load_pairwise_data()`에서 `use_variational_generation` 또는 `use_variational_selection`이 True일 때
- **Fallback 로드**:
  1. `load_selector_preference_data()`: Selection 단계
  2. `_generate_pairwise_data()`: Generation 단계
  3. `train()`: Training 시작 전

#### 문제점 및 해결
- **문제**: `client_average_z_dict`가 비어있으면 selection과 training에서 문제 발생
- **해결**: `use_variational_selection`이 True일 때도 초기에 로드하도록 수정

### 3. t-SNE 시각화

#### 저장 위치
- **Generation 단계**: `exp/{expname}/sub_exp_{timestamp}/cross_client_z_tsne_generation.png`
- **Training 중**: `exp/{expname}/sub_exp_{timestamp}/cross_client_z_tsne_round_{round_num}.png`
  - 매 5라운드마다 또는 마지막 라운드

#### 생성 조건
- Generation 단계: `is_vpl_model=True`이고 `client_average_z_dict`가 로드된 경우
- Training 중: `z_values_list`에 z 값이 수집된 경우

### 4. Test Evaluation

#### Test Set Loading
- **Prompt만 로드**: Chosen/rejected pair 없음
- **목적**: 모델이 새 response를 생성하여 evaluation 수행

#### Winrate Evaluation
- **GPT API 사용**: Fine-tuned model vs baseline model 비교
- **프로세스**:
  1. Fine-tuned model로 response 생성
  2. Baseline model (adapter 비활성화)로 response 생성
  3. GPT API (gpt-4o-mini)로 두 response 비교
  4. Winrate 계산

### 5. WandB Logging

#### RL Training
- **Train metrics**: `train_loss`, `train_avg_loss`, `train_acc`, `train_total` (winrate 제외)
- **Test metrics**: 모든 test metrics 포함 (winrate, reward model scores)
- **직접 로깅**: `logline_2_wandb_dict` 우회하여 직접 WandB에 로깅

#### Federated Training
- **VPL metrics**: `vpl_total_loss`, `vpl_reconstruction_loss`, `vpl_kl_loss`, `vpl_orthogonal_loss`
- **HRL metrics**: `avg_harmlessness`, `avg_helpfulness`, `helpfulness_winrate`, `harmlessness_winrate`
- **직접 로깅**: `merge_eval_results_from_all_clients`에서 직접 WandB에 로깅

## 실험 설정 비교

| 실험 | TID | GPU | Clients | Method | 목적 |
|------|-----|-----|---------|--------|------|
| FedBiscuit HHST | 20124 | 0 | 50 | Baseline | Baseline 비교 |
| VPL-GP HHST Ortho | 50124 | 1 | 50 | VPL-GP + Ortho | 50 clients 확장 |
| VPL-GP HRL Ortho | 51024 | 3 | 1 (RL) | RL + VPL-GP | RL fine-tuning |

## 코드 주요 변경사항

### 1. `federatedscope/llm/rlhf/standalone_training.py`
- **`load_pairwise_data()`**: 
  - `use_variational_selection`이 True일 때도 `client_average_z_dict` 로드
  - Generation 단계 t-SNE 시각화 추가
- **`load_selector_preference_data()`**:
  - 전체 response pair에 대해 dual selection 수행
  - Conflicting selection 통계 계산 및 로깅
- **`train()`**:
  - Test set은 prompt만 로드
  - Training 중 t-SNE 시각화 (매 5라운드)
  - WandB 직접 로깅

### 2. `federatedscope/llm/rlhf/variational_selector.py`
- **`variational_better_response()`**:
  - `client_average_z_dict` 파라미터 추가
  - Client-specific z 사용 지원
  - `chosen`과 `rejected` 필드 반환

### 3. `federatedscope/llm/metric/winrate_metrics.py`
- **`_get_winrate_scores_with_gpt_api()`**:
  - Fine-tuned model vs baseline model 비교
  - GPT API를 judge로 사용

### 4. `federatedscope/llm/llm_local/server.py`
- **`merge_eval_results_from_all_clients()`**:
  - WandB 직접 로깅 (VPL 및 non-VPL 모두)
  - HRL metrics 로깅

## 문서화 상태

### 업데이트된 문서
- **`docs/RL_DATA_GENERATION_AND_VPL.md`**:
  - Standard Generation 및 Dual Selection 설명
  - Conflicting Selection 통계
  - Test Set Loading 및 Winrate Evaluation
  - Client Average Z Dictionary 로드 및 t-SNE 시각화

### 주요 문서
- `WORK_SUMMARY.md`: 전체 시스템 아키텍처 및 구현 요약
- `docs/RL_DATA_GENERATION_AND_VPL.md`: RL 데이터 생성 및 VPL 가이드
- `docs/VARIATIONAL_RL_EXPLANATION.md`: Variational RL 설명

## 알려진 이슈 및 개선사항

### 해결된 이슈
1. ✅ `client_average_z_dict`가 비어있을 때의 문제
2. ✅ WandB logging이 제대로 안되던 문제
3. ✅ t-SNE 시각화가 generation 단계에서 생성되지 않던 문제
4. ✅ Test set loading이 chosen/rejected pair를 포함하던 문제

### 개선 가능한 사항
1. Generation 다양성: Temperature, top_p 등을 config에서 조정 가능하도록
2. Conflicting selection 비율 모니터링: WandB에 로깅
3. t-SNE 시각화 빈도: Config에서 조정 가능하도록

## 다음 단계

1. **실험 완료 대기**: 20124, 50124, 51024 실험 완료
2. **결과 분석**: WandB에서 metrics 비교
3. **최적화**: Hyperparameter tuning (orthogonal loss weights 등)
4. **문서화**: 실험 결과 및 분석 추가
