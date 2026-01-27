# 현재 실행 중인 실험 목록

## 실행 중인 실험 (2026-01-28 03:09 기준)

### 1. Ablation Study - RL Training (Qwen 2)

#### TID 63400: VPL + GP (no orthogonal loss) - N=10
- **알고리즘**: VPL + GP (Gumbel-Softmax Prior만, Orthogonal Loss 없음)
- **모델**: Qwen/Qwen2-0.5B
- **클라이언트 수**: 10
- **GPU**: 0
- **상태**: RL training 진행 중
- **Config**: `cfg/vpl-gp-no-ortho/hrl.yaml`
- **Selector**: TID 62400 필요
- **Checkpoint**: `/hdd/hdd3/kjb/checkpoints/hhrl_rlhf_qwen2_ablation_vplgp_n10_t63400.ckpt`

#### TID 63402: VPL + GP (no orthogonal loss) - N=100
- **알고리즘**: VPL + GP (Gumbel-Softmax Prior만, Orthogonal Loss 없음)
- **모델**: Qwen/Qwen2-0.5B
- **클라이언트 수**: 100
- **GPU**: 1
- **상태**: RL training 진행 중 (Round 39)
- **Config**: `cfg/vpl-gp-no-ortho/hrl.yaml`
- **Selector**: TID 62402 필요
- **Checkpoint**: `/hdd/hdd3/kjb/checkpoints/hhrl_rlhf_qwen2_ablation_vplgp_n100_t63402.ckpt`

#### TID 63410: VPL + Ortho (no GP prior) - N=10
- **알고리즘**: VPL + Orthogonal Loss (Standard Normal Prior, GP Prior 없음)
- **모델**: Qwen/Qwen2-0.5B
- **클라이언트 수**: 10
- **GPU**: 2
- **상태**: RL training 진행 중
- **Config**: `cfg/vpl-ortho/hrl.yaml`
- **Selector**: TID 62410 필요
- **Checkpoint**: `/hdd/hdd3/kjb/checkpoints/hhrl_rlhf_qwen2_ablation_vplortho_n10_t63410.ckpt`

#### TID 63412: VPL + Ortho (no GP prior) - N=100
- **알고리즘**: VPL + Orthogonal Loss (Standard Normal Prior, GP Prior 없음)
- **모델**: Qwen/Qwen2-0.5B
- **클라이언트 수**: 100
- **GPU**: 3
- **상태**: RL training 진행 중
- **Config**: `cfg/vpl-ortho/hrl.yaml`
- **Selector**: TID 62412 필요
- **Checkpoint**: `/hdd/hdd3/kjb/checkpoints/hhrl_rlhf_qwen2_ablation_vplortho_n100_t63412.ckpt`

### 2. Unseen Client Experiment

#### TID 70001: FedDPO RL (Unseen)
- **알고리즘**: FedDPO
- **모델**: Qwen/Qwen2-0.5B
- **클라이언트 수**: 20 (10 seen + 10 unseen)
- **GPU**: 7
- **상태**: RL training 진행 중
- **Config**: `cfg/feddpo-unseen/hrl.yaml`
- **Selector**: 불필요 (FedDPO는 selector 없음)
- **Checkpoint**: `/hdd/hdd3/kjb/checkpoints/hhrl_rlhf_qwen2_feddpo_unseen_t70001.ckpt`
- **참고**: Unseen test winrate 로깅 수정 필요 (config에 `unseen_clients_id` 추가 완료)

#### TID 70002: FedVPL Selector Training (Unseen)
- **알고리즘**: FedVPL
- **모델**: Qwen/Qwen2-0.5B
- **클라이언트 수**: 20 (10 training + 10 unseen)
- **GPU**: 6
- **상태**: Selector training 진행 중
- **Config**: `cfg/fedvpl-unseen/hhst.yaml`
- **Checkpoint**: `/hdd/hdd3/kjb/checkpoints/hhrl_choice_qwen2_fedvpl_unseen_t70002.ckpt`
- **참고**: Unseen 클라이언트(11-20)는 training에서 제외됨

#### TID 70010: FedBiscuit RL (Unseen)
- **알고리즘**: FedBiscuit
- **모델**: Qwen/Qwen2-0.5B
- **클라이언트 수**: 20 (10 seen + 10 unseen)
- **GPU**: 4
- **상태**: RL training 진행 중
- **Config**: `cfg/fedbiscuit-unseen/hrl.yaml`
- **Selector**: TID 70000 필요
- **Checkpoint**: `/hdd/hdd3/kjb/checkpoints/hhrl_rlhf_qwen2_fedbiscuit_unseen_t70010.ckpt`

#### TID 70013: FedVPL-GP-Ortho RL (Unseen)
- **알고리즘**: FedVPL-GP-Ortho
- **모델**: Qwen/Qwen2-0.5B
- **클라이언트 수**: 20 (10 seen + 10 unseen)
- **GPU**: 5
- **상태**: RL training 진행 중 (Round 39)
- **Config**: `cfg/fedvpl-gp-ortho-unseen/hrl.yaml`
- **Selector**: TID 70003 필요
- **Checkpoint**: `/hdd/hdd3/kjb/checkpoints/hhrl_rlhf_qwen2_fedvplgp_ortho_unseen_t70013.ckpt`

## GPU 사용 현황

- **GPU 0**: TID 63400 (VPL+GP RL, N=10)
- **GPU 1**: TID 63402 (VPL+GP RL, N=100)
- **GPU 2**: TID 63410 (VPL+Ortho RL, N=10)
- **GPU 3**: TID 63412 (VPL+Ortho RL, N=100)
- **GPU 4**: TID 70010 (FedBiscuit RL, Unseen)
- **GPU 5**: TID 70013 (FedVPL-GP-Ortho RL, Unseen)
- **GPU 6**: TID 70002 (FedVPL Selector, Unseen)
- **GPU 7**: TID 70001 (FedDPO RL, Unseen)

## 실험 분류

### Ablation Study (4개)
- **목적**: GP Prior와 Orthogonal Loss의 개별 효과 확인
- **실험**: VPL+GP (no ortho), VPL+Ortho (no GP)
- **클라이언트 수**: N=10, N=100
- **상태**: 모두 RL training 진행 중

### Unseen Client Experiment (4개)
- **목적**: Unseen 클라이언트에 대한 adaptation 성능 측정
- **실험**: FedDPO, FedBiscuit, FedVPL, FedVPL-GP-Ortho
- **클라이언트 수**: 20 (10 seen + 10 unseen)
- **상태**: 
  - FedDPO RL: 진행 중
  - FedVPL Selector: 진행 중
  - FedBiscuit RL: 진행 중
  - FedVPL-GP-Ortho RL: 진행 중 (Round 39)

## 주의사항

1. **Unseen Experiment 로깅**: TID 70001 (FedDPO)의 unseen test winrate 로깅이 누락되었을 수 있습니다. Config에 `unseen_clients_id` 추가 완료했으므로 재실행 시 정상 작동합니다.

2. **Selector 의존성**: 
   - TID 63400, 63402, 63410, 63412는 각각 selector checkpoint 필요
   - TID 70010은 TID 70000 selector 필요
   - TID 70013은 TID 70003 selector 필요

3. **진행 상황 확인**:
   ```bash
   # 각 실험의 최신 round 확인
   grep "Round #" outputs/{TID}.log | tail -1
   
   # 실시간 모니터링
   tail -f outputs/{TID}.log
   ```
