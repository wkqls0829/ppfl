# Main Table Cluster Test Guide

클러스터에서 main table 실험을 실행하기 전에 환경이 정상 작동하는지 확인하는 테스트 스크립트입니다.

## 목적

- 클러스터 환경 설정 확인
- 데이터 경로 및 체크포인트 경로 확인
- API 키 설정 확인
- Selector 및 RL 실험 파이프라인 검증
- 빠른 실패 감지 (전체 실험 실행 전)

## 사용 방법

### ⚠️ 중요: SLURM을 통해 실행해야 합니다

**클러스터의 로그인 노드에는 GPU가 없습니다.** 반드시 `sbatch`를 사용하여 SLURM job으로 제출해야 합니다.

### 기본 사용법

```bash
# Gemma-2B + FedVPA-GP 테스트
sbatch scripts/main_table/test_cluster.sh gemma-2b fedvpagp

# Qwen 2 + FedDPO 테스트
sbatch scripts/main_table/test_cluster.sh qwen2 feddpo

# FedBiscuit 테스트
sbatch scripts/main_table/test_cluster.sh gemma-2b fedbiscuit
```

### ❌ 잘못된 사용법 (오류 발생)

```bash
# 직접 실행하면 GPU가 없어서 오류 발생
bash scripts/main_table/test_cluster.sh gemma-2b fedvpagp
# 또는
./scripts/main_table/test_cluster.sh gemma-2b fedvpagp
```

스크립트는 SLURM job 내에서 실행되는지 자동으로 확인하며, 그렇지 않으면 명확한 오류 메시지를 출력합니다.

### 지원하는 조합

**Models:**
- `gemma-2b`
- `qwen2`

**Methods:**
- `feddpo`
- `fedbiscuit`
- `fedvpl`
- `fedvpagp`

## 테스트 내용

### 1. Selector Training Test
- **Rounds**: 2 rounds (빠른 테스트)
- **Data**: 100 train samples, 50 test samples
- **Local Steps**: 5 steps per round
- **Checkpoint**: 생성 확인

### 2. RL Training Test
- **Rounds**: 2 rounds (빠른 테스트)
- **Data**: 50 train samples, 20 test samples
- **Local Steps**: 5 steps per round
- **GPT API**: 비활성화 (테스트 속도 향상)
- **Selector Checkpoint**: 사용 (VPL methods)

## 테스트 ID

- **Selector Test TID**: 90001
- **RL Test TID**: 91001

로그 파일:
- `outputs/90001_test.log` (Selector)
- `outputs/91001_test.log` (RL)

## 확인 사항

테스트가 성공하면 다음을 확인하세요:

1. **환경 설정**
   - ✅ `.env` 파일 로드 확인
   - ✅ API 키 설정 확인 (필요한 경우)

2. **데이터 경로**
   - ✅ `$WORK_DIR/data/` 접근 가능
   - ✅ HH-RLHF 데이터셋 존재

3. **체크포인트 경로**
   - ✅ `$WORK_DIR/checkpoints/` 생성 가능
   - ✅ Selector checkpoint 생성 확인

4. **GPU 사용**
   - ✅ GPU 할당 및 사용 확인
   - ✅ 메모리 사용량 확인

5. **로그 출력**
   - ✅ 로그 파일 정상 생성
   - ✅ 에러 없이 완료

## 예상 실행 시간

- **Selector Test**: ~5-10분
- **RL Test**: ~10-15분
- **Total**: ~15-25분

## 실패 시 확인 사항

### Selector Test 실패
1. 데이터 경로 확인: `ls -la $WORK_DIR/data/`
2. Config 파일 확인: `cat cfg/main_table/test/*/hhst_test_*.yaml`
3. 로그 확인: `tail -50 outputs/90001_test.log`

### RL Test 실패
1. Selector checkpoint 확인: `ls -la checkpoints/*test*.ckpt`
2. Config 파일 확인: `cat cfg/main_table/test/*/hrl_test_*.yaml`
3. 로그 확인: `tail -50 outputs/91001_test.log`

### 공통 문제
1. **API 키 없음**: `.env` 파일 생성 및 설정
2. **데이터 없음**: 데이터셋 다운로드 필요
3. **GPU 메모리 부족**: 배치 크기 또는 모델 크기 조정
4. **경로 오류**: `$WORK_DIR` 확인

## 테스트 후 정리

테스트가 완료되면 테스트 체크포인트를 삭제할 수 있습니다:

```bash
# 테스트 체크포인트 삭제
rm -f checkpoints/*test*.ckpt

# 테스트 로그 보관 (선택사항)
mkdir -p logs/test
mv outputs/*test*.log logs/test/
```

## 다음 단계

테스트가 성공하면 main table 실험을 실행할 수 있습니다:

```bash
# Gemma-2B 실험 시작
bash scripts/main_table/submit_all_gemma.sh

# Qwen 2 실험 시작
bash scripts/main_table/submit_all_qwen.sh
```

## 주의사항

- 테스트는 최소한의 데이터와 라운드로 실행됩니다
- 실제 성능은 전체 실험에서 확인해야 합니다
- GPT API는 테스트에서 비활성화됩니다 (비용 절감)
- 테스트 체크포인트는 실제 실험에서 사용하지 마세요
