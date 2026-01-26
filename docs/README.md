# Documentation Index

이 폴더는 **코드 및 시스템 구현에 대한 기술 문서**를 포함합니다.

## 주요 문서

### VPL (Variational Preference Learning) 핵심 문서

- **[VPL_DOCUMENTATION.md](./VPL_DOCUMENTATION.md)**: VPL의 전체 아키텍처와 구현 개요
- **[VPL_CONFIGURATION_OPTIONS.md](./VPL_CONFIGURATION_OPTIONS.md)**: VPL 관련 모든 설정 옵션 설명
- **[VPL_GP_ORTHOGONAL_COMPLETE_IMPLEMENTATION.md](./VPL_GP_ORTHOGONAL_COMPLETE_IMPLEMENTATION.md)**: VPL-GP with Orthogonal Loss 완전 구현 가이드
- **[VPL_ORIGINAL_VS_OUR_IMPLEMENTATION.md](./VPL_ORIGINAL_VS_OUR_IMPLEMENTATION.md)**: 원본 VPL 논문/코드와 우리 구현 비교

### RLHF 및 Variational RL

- **[VARIATIONAL_RL_EXPLANATION.md](./VARIATIONAL_RL_EXPLANATION.md)**: Variational RL (z-conditioned RL training) 구현 설명
- **[RL_DATA_GENERATION_AND_VPL.md](./RL_DATA_GENERATION_AND_VPL.md)**: RL 데이터 생성과 VPL 통합
- **[RLHF_MEMORY_OPTIMIZATION.md](./RLHF_MEMORY_OPTIMIZATION.md)**: RLHF 메모리 최적화 기법

### Loss Functions 및 수학적 설명

- **[KL_LOSS_FORWARD_BACKWARD_EXPLANATION.md](./KL_LOSS_FORWARD_BACKWARD_EXPLANATION.md)**: KL Loss의 forward/backward pass 설명
- **[KL_LOSS_MATHEMATICAL_EXPLANATION.md](./KL_LOSS_MATHEMATICAL_EXPLANATION.md)**: KL Loss의 수학적 유도
- **[ORTHOGONAL_LOSS_DOCUMENTATION.md](./ORTHOGONAL_LOSS_DOCUMENTATION.md)**: Orthogonal Loss (CLOP-based) 구현 및 설명

### GP Prior

- **[GP_PRIOR_DOCUMENTATION.md](./GP_PRIOR_DOCUMENTATION.md)**: Gumbel Softmax Prior (GP Prior) 구현 설명

### 논문 비교

- **[PAPER_IMPLEMENTATION_COMPARISON.md](./PAPER_IMPLEMENTATION_COMPARISON.md)**: 관련 논문들과의 구현 비교

### 최적화

- **[OPTIMIZATION_PARTS.md](./OPTIMIZATION_PARTS.md)**: 최적화를 위한 주요 파츠 구분 및 최적화 포인트

### 실험 및 하이퍼파라미터 탐색

- **[HYPERPARAMETER_SEARCH.md](./HYPERPARAMETER_SEARCH.md)**: VPL-GP 하이퍼파라미터 서치 실험 계획, 진행 상황, 결과 분석 가이드

## 문서 구조

```
docs/
├── README.md (이 파일)
├── VPL_DOCUMENTATION.md
├── VPL_CONFIGURATION_OPTIONS.md
├── VARIATIONAL_RL_EXPLANATION.md
├── ORTHOGONAL_LOSS_DOCUMENTATION.md
├── GP_PRIOR_DOCUMENTATION.md
└── ...
```

## 관련 폴더

- **`documents/`**: 실험 가이드, 문제 상황, 복구 가이드 등 코드 설명이 아닌 문서들
- **`README.md`** (루트): 프로젝트 전체 개요
