#!/bin/bash

# GPU 및 SLURM 작업 상태 확인 스크립트

echo "=========================================="
echo "SLURM Job 및 GPU 상태 확인"
echo "=========================================="
echo ""

# 1. 내 작업 확인
echo "1. 내가 제출한 작업:"
echo "----------------------------------------"
squeue -u $USER -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R" 2>/dev/null || echo "squeue 명령어를 사용할 수 없습니다."
echo ""

# 2. GPU 파티션별 작업 수
echo "2. GPU 파티션별 작업 수:"
echo "----------------------------------------"
for partition in A6000 RTX4090 RTX6000ADA A5000; do
    count=$(squeue -p $partition -h 2>/dev/null | wc -l)
    echo "  $partition: $count jobs"
done
echo ""

# 3. 노드별 GPU 상태
echo "3. 노드별 GPU 상태:"
echo "----------------------------------------"
sinfo -o '%N %G %t %e' -p A6000,RTX4090,RTX6000ADA,A5000 2>/dev/null | head -20 || echo "sinfo 명령어를 사용할 수 없습니다."
echo ""

# 4. GPU 파티션별 사용 가능한 노드
echo "4. GPU 파티션별 사용 가능한 노드:"
echo "----------------------------------------"
for partition in A6000 RTX4090 RTX6000ADA A5000; do
    echo "  $partition:"
    sinfo -p $partition -o '%N %G %t' -h 2>/dev/null | grep -E "idle|mixed" | head -5 || echo "    정보 없음"
done
echo ""

# 5. 실행 중인 작업의 GPU 사용률 (가능한 경우)
echo "5. 실행 중인 작업 정보:"
echo "----------------------------------------"
squeue -u $USER -t RUNNING -o "%.18i %.20j %.8u %.10M %.6D %N" 2>/dev/null | head -10 || echo "실행 중인 작업 없음"
echo ""

echo "=========================================="
echo "유용한 명령어:"
echo "=========================================="
echo "  - 작업 상세 정보: scontrol show job <job_id>"
echo "  - 작업 취소: scancel <job_id>"
echo "  - 특정 노드 GPU 확인: ssh <node> nvidia-smi"
echo "  - 작업 로그 확인: tail -f outputs/<tid>.log"
echo ""
