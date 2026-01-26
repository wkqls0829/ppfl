# SLURM GPU 모니터링 가이드

클러스터에서 제출된 batch job과 GPU 여유 상황을 확인하는 방법입니다.

## 빠른 확인

### 내 작업 확인
```bash
# 내가 제출한 모든 작업
squeue -u $USER

# 실행 중인 작업만
squeue -u $USER -t RUNNING

# 대기 중인 작업만
squeue -u $USER -t PENDING
```

### GPU 파티션별 작업 확인
```bash
# 특정 파티션의 모든 작업
squeue -p A6000
squeue -p RTX4090
squeue -p RTX6000ADA
squeue -p A5000

# 모든 GPU 파티션
squeue -p A6000,RTX4090,RTX6000ADA,A5000
```

## 상세 정보 확인

### 작업 상세 정보
```bash
# 특정 작업의 상세 정보
scontrol show job <job_id>

# 출력 예시:
# JobId=12345 JobName=sel_fedvpagp_n10_62130
# UserId=jbkoo(1000) GroupId=jbkoo(1000)
# Priority=4294901758 Nice=0 Account=(null) QOS=normal
# JobState=RUNNING Reason=None Dependency=(null)
# Requeue=1 Restarts=0 BatchFlag=1 Reboot=0 ExitCode=0:0
# RunTime=01:23:45 TimeLimit=2-00:00:00 TimeMin=N/A
# SubmitTime=2026-01-26T10:00:00 EligibleTime=2026-01-26T10:00:00
# StartTime=2026-01-26T10:05:00 EndTime=2026-01-26T12:05:00
# PreemptTime=None SuspendTime=None SecsPreSuspend=0
# Partition=A6000 AllocNode:Sid=login:12345
# ReqNodeList=(null) ExcNodeList=n27,n33,n42,n72
# NodeList=gpu-node01
# BatchHost=gpu-node01
# NumNodes=1 NumCPUs=8 NumTasks=1 CPUs/Task=8 ReqB:S:C:T=0:0:*:*
# TRES=cpu=8,mem=32G,gres/gpu=1
# Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=*
# MinCPUsNode=8 MinMemoryNode=32G MinTmpDiskNode=0
# Features=(null) DelayBoot=00:00:00
# OverSubscribe=OK Contiguous=0 Licenses=(null) Network=(null)
# Command=/home2/jbkoo/ppfl/scripts/main_table/run_selector_gemma.sh
# WorkDir=/home2/jbkoo/ppfl
# StdOut=/home2/jbkoo/slurm/logs/slurm-12345-run_selector_gemma.out
# StdErr=/home2/jbkoo/slurm/logs/slurm-12345-run_selector_gemma.out
```

### 작업 출력 형식 커스터마이징
```bash
# 상세한 출력 형식
squeue -u $USER -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %N %b"

# 각 필드 의미:
# %i: Job ID
# %P: Partition
# %j: Job Name
# %u: User
# %t: State (RUNNING, PENDING, etc.)
# %M: Time Used
# %D: Number of Nodes
# %N: Node List
# %b: GPU allocation (gres/gpu)
```

## GPU 상태 확인

### 노드별 GPU 상태
```bash
# 모든 GPU 노드 상태
sinfo -o '%N %G %t %e' -p A6000,RTX4090,RTX6000ADA,A5000

# 출력 예시:
# NODELIST           GRES       STATE    END_TIME
# gpu-node01         gpu:a6000:1 idle
# gpu-node02         gpu:rtx4090:1 mixed   2026-01-26T12:00:00
# gpu-node03         gpu:rtx4090:1 alloc
```

### 특정 노드의 GPU 사용률
```bash
# 노드에 SSH 접속 후
ssh gpu-node01
nvidia-smi

# 또는 원격으로
ssh gpu-node01 nvidia-smi
```

### GPU 파티션별 사용 가능한 리소스
```bash
# 각 파티션의 사용 가능한 GPU 수
sinfo -p A6000 -o '%G %t' -h | grep -c idle
sinfo -p RTX4090 -o '%G %t' -h | grep -c idle
sinfo -p RTX6000ADA -o '%G %t' -h | grep -c idle
sinfo -p A5000 -o '%G %t' -h | grep -c idle
```

## 작업 관리

### 작업 취소
```bash
# 특정 작업 취소
scancel <job_id>

# 내 모든 작업 취소
scancel -u $USER

# 특정 이름의 작업 취소
scancel -n "sel_fedvpagp_n10_62130"
```

### 작업 의존성 확인
```bash
# 작업의 의존성 확인
scontrol show job <job_id> | grep Dependency

# 의존성 작업이 완료되기를 기다리는 작업
squeue -u $USER -t PENDING -o "%.18i %.20j %R" | grep Dependency
```

## 실시간 모니터링

### 작업 상태 실시간 모니터링
```bash
# 5초마다 업데이트
watch -n 5 'squeue -u $USER'

# 또는
watch -n 5 'squeue -p A6000,RTX4090,RTX6000ADA,A5000'
```

### GPU 사용률 실시간 모니터링
```bash
# 특정 노드의 GPU 사용률 모니터링
watch -n 2 'ssh gpu-node01 nvidia-smi'
```

## 유용한 스크립트

### check_gpu_status.sh
```bash
# 간편한 상태 확인 스크립트
bash scripts/main_table/check_gpu_status.sh
```

이 스크립트는 다음을 확인합니다:
- 내가 제출한 작업 목록
- GPU 파티션별 작업 수
- 노드별 GPU 상태
- 사용 가능한 노드 정보

## 문제 해결

### 작업이 계속 대기 중인 경우
```bash
# 대기 이유 확인
squeue -u $USER -t PENDING -o "%.18i %.20j %R"

# 일반적인 이유:
# - Resources: GPU가 부족함
# - Dependency: 다른 작업 완료 대기
# - Priority: 우선순위가 낮음
```

### GPU 할당 확인
```bash
# 작업 내에서 GPU 확인 (작업 스크립트 내에서)
echo "Allocated GPUs: $SLURM_GPUS_ON_NODE"
echo "GPU IDs: $CUDA_VISIBLE_DEVICES"
nvidia-smi
```

## 예시 출력

### squeue 출력 예시
```
JOBID   PARTITION  NAME              USER    ST  TIME  NODES  NODELIST(REASON)
12345   A6000      sel_fedvpagp_n10 jbkoo   R   1:23  1      gpu-node01
12346   RTX4090    sel_fedvpagp_n50 jbkoo   R   0:45  1      gpu-node02
12347   A6000      sel_fedvpagp_n100 jbkoo   PD  0:00  1      (Resources)
```

- **R (RUNNING)**: 실행 중
- **PD (PENDING)**: 대기 중
- **CG (COMPLETING)**: 완료 중
- **CD (COMPLETED)**: 완료됨
- **F (FAILED)**: 실패

### sinfo 출력 예시
```
NODELIST           GRES              STATE    END_TIME
gpu-node01         gpu:a6000:1       idle
gpu-node02         gpu:rtx4090:1     mixed    2026-01-26T12:00:00
gpu-node03         gpu:rtx4090:1     alloc
```

- **idle**: 사용 가능 (작업 없음)
- **mixed**: 일부 사용 중
- **alloc**: 모두 사용 중
- **down**: 다운됨

## 팁

1. **작업 이름으로 필터링**
   ```bash
   squeue -u $USER -n "sel_*"  # selector 작업만
   squeue -u $USER -n "rl_*"   # RL 작업만
   ```

2. **특정 TID 작업 확인**
   ```bash
   squeue -u $USER -n "*62130*"
   ```

3. **로그 파일 확인**
   ```bash
   # 실시간 로그
   tail -f outputs/62130.log
   
   # SLURM 출력
   tail -f /home2/jbkoo/slurm/logs/slurm-*.out
   ```
