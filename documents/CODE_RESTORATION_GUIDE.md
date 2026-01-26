# 코드 복원 가이드

## 개요
이 문서는 50271 로그를 기준으로 손실된 코드를 복원하는 방법을 안내합니다.

## 복원 원칙
1. **50271 로그의 라인 번호와 로그 메시지를 기준으로 복원**
2. **기능 일치 여부가 라인 번호 일치보다 중요**
3. **테스트를 통해 정상 작동 확인**

## 복원이 필요한 파일 목록

### 1. federatedscope/llm/llm_local/server.py
**필요한 함수들:**
- `_collect_vpl_gp_prior_distributions()` (원본: line 382)
- `_compute_balanced_orthogonal_labels()` (원본: line 114, 401)
- `_collect_z_values_for_visualization()` (원본: line 427)
- `_visualize_cross_client_z()`
- `broadcast_model_para()` 오버라이드 (원본: line 623, 632)

**50271 로그에서 확인된 동작:**
```
server:382  - Collected 5 client z distributions for VPL-GP prior. Total clients in prior: 5 (updated: 5, from previous rounds: 0)
server:114  - Assigned manual orthogonal labels: {2: 0, 3: 0, 5: 0, 9: 1, 10: 1}
server:401  - Computed balanced orthogonal labels for 5 clients at round 0: {0: 3, 1: 2}
server:427  - Round 0: Collected z values from 5 clients. Total accumulated: 5 points across 5 clients
server:623  - Broadcasting VPL-GP prior with 5 client distributions at round 1
server:632  - Broadcasting orthogonal labels to clients at round 1
```

**Manual Labels 할당 로직:**
- 참여한 클라이언트만 할당 (train_msg_buffer.keys() 사용)
- 첫 절반은 harmless (0), 나머지는 helpful (1)
- 예: Round 0에서 [2, 3, 5, 9, 10] 참여 → {2:0, 3:0, 5:0, 9:1, 10:1}

### 2. federatedscope/llm/llm_local/client.py
**필요한 기능:**
- z distribution 전송 (get_client_z_distribution() 호출)
- prior 수신 및 업데이트 (update_prior_from_server() 호출)
- orthogonal labels 수신 및 업데이트

### 3. federatedscope/llm/trainer/vpl_reward_choice_trainer.py
**필요한 기능:**
- VPL 기본 구현
- Preference feature extraction
- Variational encoder 통합

### 4. federatedscope/llm/model/variational_encoder.py
**필요한 기능:**
- VariationalEncoder 클래스
- encode(), reparameterize(), forward() 메서드

### 5. federatedscope/llm/model/variational_encoder_gp.py
**필요한 기능:**
- VariationalEncoderGP 클래스 (VariationalEncoder 상속)
- update_prior() - 다른 클라이언트들의 z-distribution으로 prior 업데이트
- kl_divergence() - Mixture prior와의 KL divergence 계산
- sample_prior() - Gumbel Softmax를 사용한 prior 샘플링

### 6. federatedscope/llm/trainer/vpl_gp_reward_choice_trainer.py
**필요한 기능:**
- VPLGPRewardChoiceTrainer 클래스 (VPLRewardChoiceTrainer 상속)
- get_client_z_distribution() - 클라이언트의 z-distribution 반환
- update_prior_from_server() - 서버로부터 받은 prior 업데이트
- update_orthogonal_label_from_server() - 서버로부터 받은 orthogonal label 업데이트

### 7. federatedscope/llm/rlhf/z_visualization.py
**필요한 기능:**
- visualize_cross_client_z() 함수
- t-SNE 시각화
- WandB 로깅

## 복원 검증 방법

### 1. 로그 메시지 확인
50271 로그와 동일한 메시지가 출력되는지 확인:
- "Collected X client z distributions for VPL-GP prior"
- "Assigned manual orthogonal labels: {...}"
- "Computed balanced orthogonal labels for X clients"
- "Round X: Collected z values from X clients"
- "Broadcasting VPL-GP prior with X client distributions"
- "Broadcasting orthogonal labels to clients"

### 2. t-SNE 시각화 확인
- `exp/vplgp_hhst_fd_t50271/sub_exp_*/cross_client_z_tsne_round_*.png` 파일 생성 확인
- WandB에 t-SNE 이미지 업로드 확인

### 3. 기능 테스트
- 50271과 동일한 설정으로 실험 실행
- 각 라운드에서 정상 작동 확인

## 주의사항

1. **라인 번호 불일치**
   - 원본 코드와 복원 코드의 라인 번호가 다를 수 있음
   - 기능 일치 여부가 더 중요

2. **Manual Labels 로직**
   - 참여한 클라이언트만 할당해야 함
   - 모든 클라이언트에 할당하면 안 됨

3. **데이터셋 구조**
   - LLMComparisonDataset 사용 시 win_dataset과 lose_dataset 구조 이해 필요
   - 원본 list_data_dict 접근 방법 확인 필요

4. **캐시 관리**
   - Round별로 캐시 초기화 필요
   - 동일 라운드에서 중복 계산 방지

## 다음 단계

1. 코드 복원 (50271 로그 기준)
2. Manual labels 로직 수정
3. 테스트 실행
4. 로그 확인
5. Git 설정 변경
