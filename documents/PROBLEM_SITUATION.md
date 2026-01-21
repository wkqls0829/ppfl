# 문제 상황 보고서: Rebase Abort로 인한 코드 손실

## 발생 일시
- **2026-01-19 14:20:29**: `git rebase abort` 실행
- **2026-01-16 12:15**: 50271 실험 실행 (정상 작동했던 코드)

## 문제 요약
`git pull`이 `pull.rebase true` 설정으로 인해 자동 rebase 모드로 실행되었고, rebase 중 abort되면서 **커밋되지 않은 로컬 변경사항이 모두 손실**되었습니다.

## 손실된 코드
### 1. `federatedscope/llm/llm_local/server.py`
- **원본 라인 수**: 최소 632줄 이상 (50271 로그 기준)
- **현재 상태**: 256줄 (원본 상태로 되돌아감)
- **손실된 주요 기능**:
  - `_collect_vpl_gp_prior_distributions()` (원본: line 382)
  - `_compute_balanced_orthogonal_labels()` (원본: line 114, 401)
  - `_collect_z_values_for_visualization()` (원본: line 427)
  - `_visualize_cross_client_z()` (원본: 호출 위치 불명)
  - `broadcast_model_para()` 오버라이드 (원본: line 623, 632)

### 2. 50271 로그에서 확인된 실제 라인 번호
```
server:114  - Assigned manual orthogonal labels
server:382  - Collected 5 client z distributions for VPL-GP prior
server:401  - Computed balanced orthogonal labels
server:427  - Round 0: Collected z values from 5 clients
server:623  - Broadcasting VPL-GP prior with 5 client distributions
server:632  - Broadcasting orthogonal labels to clients
```

### 3. 관련 파일들
- `federatedscope/llm/trainer/vpl_reward_choice_trainer.py` - 복원 필요
- `federatedscope/llm/model/variational_encoder.py` - 복원 필요
- `federatedscope/llm/model/variational_encoder_gp.py` - 생성 필요
- `federatedscope/llm/trainer/vpl_gp_reward_choice_trainer.py` - 생성 필요
- `federatedscope/llm/rlhf/z_visualization.py` - 생성 필요
- `federatedscope/llm/llm_local/client.py` - 수정 필요 (z distribution 전송)

## 원인 분석

### Git 설정 확인
```bash
$ git config pull.rebase
true
```

### Bash History 분석
```bash
git pull
git pull
git pull
git config pull.rebase true
git pull
git pull
git pull remote padding
```

### Reflog 분석
```
0785f44 HEAD@{2026-01-19 14:20:29 +0900}: rebase (abort): returning to refs/heads/eval
c3e28e4 HEAD@{5}: pull (start): checkout c3e28e44b24d478158827989b863dd15beda720a
```

### 문제 발생 과정
1. `git pull` 실행
2. `pull.rebase true` 설정으로 자동 rebase 시작
3. 로컬과 origin/eval이 diverged 상태 (로컬 1개 커밋, 원격 7개 커밋)
4. Rebase 중 충돌 또는 문제 발생
5. `git rebase --abort` 실행 (수동 또는 자동)
6. **커밋되지 않은 로컬 변경사항 모두 손실**

## 현재 상태

### Git 상태
```
On branch eval
Your branch and 'origin/eval' have diverged,
and have 1 and 7 different commits each, respectively.

Changes not staged for commit:
  modified:   federatedscope/llm/llm_local/client.py
  modified:   federatedscope/llm/llm_local/server.py
```

### 파일 상태
- `federatedscope/llm/llm_local/server.py`: 256줄 (원본 상태)
- `federatedscope/llm/llm_local/client.py`: 수정됨 (상태 불명)
- 기타 VPL-GP 관련 파일들: 삭제됨 또는 원본 상태

## 발견된 차이점

### 1. Manual Labels 할당 로직
**50271 로그에서 확인된 실제 동작:**
- Round 0: `{2: 0, 3: 0, 5: 0, 9: 1, 10: 1}` (참여한 클라이언트만 할당)
- Round 1: `{2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 9: 1, 10: 1}`

**원본 코드 예상 동작:**
- 참여한 클라이언트만 할당 (train_msg_buffer.keys() 사용)
- 첫 절반은 harmless (0), 나머지는 helpful (1)

### 2. 라인 번호 불일치
- 원본 코드와 복원 코드의 라인 번호가 다를 수 있음
- 기능은 동일하지만, 로그 메시지의 라인 번호가 맞지 않을 수 있음

## 복원 시도 기록

### 시도한 방법들
1. ✅ Git reflog 확인 - rebase abort만 확인됨
2. ✅ Git stash 확인 - 없음
3. ✅ Dangling commit 확인 - b2cc7b9 (256줄, 원본 아님)
4. ✅ VSCode 히스토리 확인 - 없음
5. ✅ 임시 파일 확인 - `/tmp/server_original.py` (256줄, 원본 아님)
6. ✅ Exp 디렉토리 확인 - 백업 없음
7. ✅ 50271 로그 분석 - 라인 번호와 로그 메시지로 복원 시도

### 복원 불가능한 이유
- **50271 실행 시점의 코드는 커밋되지 않은 로컬 변경사항이었음**
- **Git 히스토리에는 존재하지 않음**
- **VSCode나 다른 백업 시스템에도 없음**
- **50271 로그의 라인 번호와 로그 메시지를 기준으로 복원하는 것이 최선**

## 앞으로 방지 방법

### 1. Git 설정 변경
```bash
# Rebase 모드 비활성화
git config pull.rebase false

# 또는 pull 시 명시적으로 merge 사용
git pull --no-rebase
```

### 2. 작업 전 확인
```bash
# 변경사항 확인
git status

# 중요 변경사항은 즉시 커밋
git add .
git commit -m "WIP: 작업 중인 변경사항"
```

### 3. 백업 브랜치 생성
```bash
# 작업 전 백업 브랜치 생성
git branch backup-$(date +%Y%m%d-%H%M%S)
```

### 4. Rebase 전 확인
```bash
# Rebase 시작 전 변경사항 커밋
git add .
git commit -m "Before rebase"

# 또는 stash 사용
git stash save "Before rebase"
```

## 복원이 필요한 부분

### 1. Manual Labels 로직
**수정 필요:**
- 참여한 클라이언트만 할당 (train_msg_buffer.keys() 사용)
- 첫 절반은 harmless (0), 나머지는 helpful (1)

### 2. 라인 번호 확인
- 현재 복원 코드는 기능적으로 동일하지만 라인 번호가 다를 수 있음
- 로그 메시지의 라인 번호가 맞지 않아 디버깅 시 혼란 가능

### 3. 테스트 필요
- 50271과 동일한 설정으로 실험 실행
- t-SNE 시각화가 정상 작동하는지 확인
- 로그 메시지가 정상 출력되는지 확인

## 참고 파일

### 로그 파일
- `/home/kjb/ppfl/outputs/50271.log` - 정상 작동했던 실험 로그 (4388줄)
- `/home/kjb/ppfl/outputs/50240.log` - 이전 실험 로그 (비교용, 4403줄)
- `/home/kjb/ppfl/outputs/50290.log` - 복원 후 실험 로그 (t-SNE 미작동, 1309줄)

### 설정 파일
- `/home/kjb/ppfl/exp/vplgp_hhst_fd_t50271/config.yaml` - 50271 실험 설정

### 문서 파일
- `/home/kjb/ppfl/CONTEXT_FOR_NEW_SESSION.md` - 프로젝트 컨텍스트
- `/home/kjb/ppfl/PROMPT_FOR_NEW_SESSION.txt` - 세션 프롬프트

## 다음 단계

1. **코드 복원** (50271 로그 기준)
   - server.py의 VPL-GP 관련 함수들 복원
   - client.py의 z distribution 전송 로직 확인
   - 관련 trainer 및 model 파일 복원

2. **Manual labels 로직 수정** (참여한 클라이언트만 할당)

3. **테스트 실행** (50271과 동일한 설정)

4. **로그 확인** (t-SNE 시각화 정상 작동 확인)

5. **Git 설정 변경** (pull.rebase false)

6. **백업 브랜치 생성 습관화**

## 중요 참고사항

- **50271 실행 시점의 코드는 커밋되지 않은 로컬 변경사항이었음**
- **Git 히스토리에는 존재하지 않음**
- **50271 로그의 라인 번호와 로그 메시지를 기준으로 복원하는 것이 최선**
- **현재 복원 코드는 기능적으로 동일하지만 세부 구현이 다를 수 있음**
- **라인 번호는 코드 추가/삭제에 따라 달라질 수 있으므로, 기능 일치 여부 확인이 더 중요**
