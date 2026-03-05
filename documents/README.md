# Documents Index

이 폴더는 **코드/시스템 설명이 아닌 문서들**을 포함합니다:
- 실험 가이드 및 상태
- 문제 상황 보고서
- 코드 복구 가이드
- GPU 할당 가이드
- API 설정 가이드
- 기타 운영 관련 문서

## 주요 문서

### 실험 관련

- **[EXPERIMENT_GUIDE.md](./EXPERIMENT_GUIDE.md)**: 실험 실행 가이드
- **[EXPERIMENT_60000_HYPERPARAMETERS.md](./EXPERIMENT_60000_HYPERPARAMETERS.md)**: 실험 60000번대 하이퍼파라미터 설정

### 문제 상황 및 복구

- **[PROBLEM_SITUATION.md](./PROBLEM_SITUATION.md)**: Git rebase abort로 인한 코드 손실 문제 상황 보고서
- **[CODE_RESTORATION_GUIDE.md](./CODE_RESTORATION_GUIDE.md)**: 손실된 코드 복구 가이드
- **[TSNE_LOGGING_STATUS.md](./TSNE_LOGGING_STATUS.md)**: t-SNE 로깅 상태 확인 및 수정 사항

### 시스템 설정

- **[GPU_ALLOCATION_GUIDE.md](./GPU_ALLOCATION_GUIDE.md)**: GPU 할당 가이드
- **[GPU_ALLOCATION_VERIFICATION.md](./GPU_ALLOCATION_VERIFICATION.md)**: GPU 할당 검증 방법
- **[GPT_API_SETUP.md](./GPT_API_SETUP.md)**: GPT API 설정 가이드
- **[../docs/API_KEY_SETUP.md](../docs/API_KEY_SETUP.md)**: OpenAI API 키 설정 (로컬·클러스터·.env)

### 컨텍스트 및 구현 이력

- **[CONTEXT_FOR_NEW_SESSION.md](./CONTEXT_FOR_NEW_SESSION.md)**: 새 세션을 위한 프로젝트 컨텍스트
- **[Z_EMBEDDING_GENERATION_EXPLANATION.md](./Z_EMBEDDING_GENERATION_EXPLANATION.md)**: Z embedding generation 설명
- **[SIGMA_INITIALIZATION_EXPLANATION.md](./SIGMA_INITIALIZATION_EXPLANATION.md)**: Sigma 초기화 설명
- **[TRAINING_SPEED_OPTIMIZATION.md](./TRAINING_SPEED_OPTIMIZATION.md)**: 학습 속도 최적화
- **[CLOP_ORTHOGONAL_LOSS_EXPLANATION.md](./CLOP_ORTHOGONAL_LOSS_EXPLANATION.md)**: CLOP Orthogonal Loss 설명
- **[GUMBEL_SOFTMAX_DIFFERENTIABILITY.md](./GUMBEL_SOFTMAX_DIFFERENTIABILITY.md)**: Gumbel Softmax 미분가능성
- **[MIXTURE_PRIOR_EXPLANATION.md](./MIXTURE_PRIOR_EXPLANATION.md)**: Mixture Prior 설명
- **[VPL_UNIFIED_IMPLEMENTATION.md](./VPL_UNIFIED_IMPLEMENTATION.md)**: VPL 통합 구현
- **[VPL_IMPLEMENTATION_DETAILED.md](./VPL_IMPLEMENTATION_DETAILED.md)**: VPL 상세 구현
- **[VPL_GP_IMPLEMENTATION.md](./VPL_GP_IMPLEMENTATION.md)**: VPL-GP 구현

## 문서 구조

```
documents/
├── README.md (이 파일)
├── EXPERIMENT_GUIDE.md
├── PROBLEM_SITUATION.md
├── CODE_RESTORATION_GUIDE.md
├── TSNE_LOGGING_STATUS.md
└── ...
```

## 관련 폴더

- **`docs/`**: 코드 및 시스템 구현에 대한 기술 문서
- **`README.md`** (루트): 프로젝트 전체 개요
