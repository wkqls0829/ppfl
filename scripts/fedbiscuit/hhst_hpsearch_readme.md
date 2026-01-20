# FedBiscuit HHST Hyperparameter Search Script

이 스크립트는 FedBiscuit HHST (binary selector training)에서 여러 hyperparameter 조합을 병렬로 테스트합니다.

## 사용법

```bash
cd /home/kjb/ppfl
./scripts/fedbiscuit/hhst_hpsearch.sh
```

## Hyperparameter Search Space

- **Learning Rate**: 5e-5, 1e-4, 5e-4 (3 values)
- **Batch Size**: 2, 4 (2 values)
- **Gradient Accumulation Steps**: 2, 4 (2 values)
- **Local Update Steps**: 20, 30 (2 values)

**총 조합 수**: 3 × 2 × 2 × 2 = **24 jobs**

## GPU 할당

- GPU IDs: 0, 1, 4, 5 (4 GPUs)
- 각 GPU당 6 jobs 실행 (round-robin 방식으로 할당)

## 출력 파일

### Checkpoints
- 위치: `/hdd/hdd3/kjb/checkpoints/`
- 파일명: `hhrl_choice_gemma_fedbiscuit_u3_hpsearch_${tid}.ckpt`
- TID 범위: 10200 ~ 10223

### Log Files
- 위치: `outputs/`
- 파일명: `${tid}_hpsearch_lr${lr}_bs${bs}_gas${gas}_lus${lus}.log`
- 예: `10200_hpsearch_lr5e-05_bs2_gas2_lus20.log`

### Job PIDs
- 파일: `outputs/hpsearch_job_pids.txt`
- 모든 job의 PID를 저장 (중단 시 사용)

## WandB

- Project: `ppfl-hhrl`
- Experiment names: `hhst_hpsearch_lr${lr}_bs${bs}_gas${gas}_lus${lus}_t${tid}`
- 각 실험의 hyperparameter가 이름에 포함되어 비교가 용이합니다.

## 모니터링

### 모든 job 로그 확인
```bash
watch -n 1 'tail -n 5 outputs/10200_*_hpsearch*.log'
```

### GPU 사용량 확인
```bash
watch -n 1 nvidia-smi
```

### 특정 job 로그 확인
```bash
tail -f outputs/10200_hpsearch_lr5e-05_bs2_gas2_lus20.log
```

### Job 상태 확인
```bash
ps aux | grep "federatedscope/main.py" | grep hpsearch
```

## Job 중단

### 모든 job 중단
```bash
# PIDs 파일에서 읽기
kill $(cat outputs/hpsearch_job_pids.txt)

# 또는 특정 GPU의 jobs만 중단
pkill -f "CUDA_VISIBLE_DEVICES=0.*hpsearch"
```

### 특정 job 중단
```bash
# PID 확인
ps aux | grep "hpsearch.*tid" | grep -v grep

# 중단
kill <PID>
```

## 결과 분석

### WandB에서 비교
1. WandB 프로젝트 `ppfl-hhrl` 접속
2. Experiment 이름으로 필터링: `hhst_hpsearch_*`
3. 하이퍼파라미터별로 그룹화하여 성능 비교
4. 주요 메트릭: `test_loss`, `test_acc`, `test_avg_harmlessness`, `test_avg_helpfulness`

### 로그에서 직접 확인
```bash
# 각 실험의 최종 결과 확인
grep -r "Round 150" outputs/10200_*_hpsearch*.log

# 특정 메트릭 추출
grep "test_loss" outputs/10200_*_hpsearch*.log | tail -24
```

## 커스터마이징

스크립트를 수정하여 다른 hyperparameter 조합을 테스트할 수 있습니다:

```bash
# 예: 더 많은 learning rate 값 테스트
LEARNING_RATES=(1e-5 5e-5 1e-4 5e-4 1e-3)

# 예: 다른 batch size 범위
BATCH_SIZES=(1 2 4 8)

# 예: GPU 변경
GPU_IDS=(2 3 6 7)
```

## 주의사항

1. **메모리 사용량**: 각 job이 독립적으로 실행되므로, GPU 메모리가 충분한지 확인하세요.
2. **디스크 공간**: 24개의 checkpoint 파일이 생성되므로 충분한 디스크 공간이 필요합니다.
3. **실행 시간**: 모든 job이 완료될 때까지 시간이 걸릴 수 있습니다 (total_round_num=150).
4. **WandB 로깅**: 모든 실험이 같은 프로젝트에 로깅되므로, 실험 이름으로 구분됩니다.
