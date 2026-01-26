# Meta Documentation

이 문서는 프로젝트의 문서 구조와 각 문서의 목적을 설명하는 메타 문서입니다.

## 문서 구조

프로젝트의 문서는 두 가지 주요 폴더로 구성됩니다:

### 1. `docs/` - 기술 문서 (Technical Documentation)

**목적**: 코드 및 시스템 구현에 대한 기술적 설명

**포함 내용**:
- VPL (Variational Preference Learning) 아키텍처 및 구현
- Loss functions의 수학적 유도 및 구현
- Configuration 옵션 설명
- 논문과의 구현 비교

**대상 독자**: 개발자, 연구자

**주요 문서**:
- `VPL_DOCUMENTATION.md`: VPL 전체 개요
- `VARIATIONAL_RL_EXPLANATION.md`: Variational RL 구현
- `ORTHOGONAL_LOSS_DOCUMENTATION.md`: Orthogonal Loss 구현
- `GP_PRIOR_DOCUMENTATION.md`: GP Prior 구현
- `KL_LOSS_*.md`: KL Loss 수학적 설명

### 2. `documents/` - 운영 문서 (Operational Documentation)

**목적**: 코드 설명이 아닌 운영, 실험, 문제 해결 관련 문서

**포함 내용**:
- 실험 가이드 및 상태
- 문제 상황 보고서
- 코드 복구 가이드
- 시스템 설정 가이드
- GPU 할당 가이드

**대상 독자**: 실험자, 운영자

**주요 문서**:
- `EXPERIMENT_GUIDE.md`: 실험 실행 가이드
- `PROBLEM_SITUATION.md`: 문제 상황 보고서
- `CODE_RESTORATION_GUIDE.md`: 코드 복구 가이드
- `TSNE_LOGGING_STATUS.md`: t-SNE 로깅 상태

## 문서 분류 기준

### `docs/`에 포함되는 문서

✅ **포함**:
- 코드 구현 설명
- 수학적 유도 및 이론
- 아키텍처 설명
- API/Configuration 설명
- 논문 비교

❌ **제외**:
- 실험 실행 가이드
- 문제 상황 보고서
- 코드 복구 가이드
- 시스템 설정 (API 키 등)

### `documents/`에 포함되는 문서

✅ **포함**:
- 실험 가이드 및 상태
- 문제 상황 및 해결 방법
- 코드 복구 가이드
- 시스템 설정 가이드
- GPU 할당 가이드
- 이전 구현 이력 (참고용)

❌ **제외**:
- 현재 코드 구현 설명
- 수학적 유도
- 아키텍처 설명

## 문서 네이밍 규칙

### 기술 문서 (`docs/`)

- `*_DOCUMENTATION.md`: 전체 구현 문서
- `*_EXPLANATION.md`: 특정 개념/구현 설명
- `*_OPTIONS.md`: 설정 옵션 설명
- `*_COMPARISON.md`: 비교 문서

### 운영 문서 (`documents/`)

- `*_GUIDE.md`: 가이드 문서
- `*_STATUS.md`: 상태 확인 문서
- `*_SITUATION.md`: 문제 상황 문서
- `*_SETUP.md`: 설정 가이드

## 문서 유지보수

### 정기적으로 확인할 사항

1. **중복 문서 제거**: 동일한 내용을 다루는 문서 통합
2. **오래된 문서 정리**: 더 이상 유효하지 않은 문서 삭제 또는 `documents/`로 이동
3. **인덱스 업데이트**: `README.md` 파일에 새 문서 추가

### 문서 작성 시 고려사항

1. **분류 확인**: 문서가 `docs/`인지 `documents/`인지 명확히 구분
2. **중복 방지**: 기존 문서와 내용이 겹치지 않는지 확인
3. **인덱스 업데이트**: 새 문서를 해당 폴더의 `README.md`에 추가

## 빠른 참조

### VPL 관련 문서 찾기

- **전체 개요**: `docs/VPL_DOCUMENTATION.md`
- **설정 옵션**: `docs/VPL_CONFIGURATION_OPTIONS.md`
- **GP Prior**: `docs/GP_PRIOR_DOCUMENTATION.md`
- **Orthogonal Loss**: `docs/ORTHOGONAL_LOSS_DOCUMENTATION.md`
- **원본 비교**: `docs/VPL_ORIGINAL_VS_OUR_IMPLEMENTATION.md`

### RL 관련 문서 찾기

- **Variational RL**: `docs/VARIATIONAL_RL_EXPLANATION.md`
- **RL 데이터 생성**: `docs/RL_DATA_GENERATION_AND_VPL.md`
- **메모리 최적화**: `docs/RLHF_MEMORY_OPTIMIZATION.md`

### 실험 관련 문서 찾기

- **실험 가이드**: `documents/EXPERIMENT_GUIDE.md`
- **하이퍼파라미터**: `documents/EXPERIMENT_60000_HYPERPARAMETERS.md`
- **GPU 할당**: `documents/GPU_ALLOCATION_GUIDE.md`

### 문제 해결 문서 찾기

- **문제 상황**: `documents/PROBLEM_SITUATION.md`
- **코드 복구**: `documents/CODE_RESTORATION_GUIDE.md`
- **t-SNE 상태**: `documents/TSNE_LOGGING_STATUS.md`

## 문서 업데이트 이력

- **2026-01-22**: 문서 구조 재정리
  - `docs/`와 `documents/` 폴더 분리
  - 중복 문서 제거
  - Meta documentation 생성
