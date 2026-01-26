# Environment Setup Guide

이 가이드는 클러스터에서 conda 환경을 설정하는 방법을 설명합니다.

## 문제: PyTorch CUDA 빌드 설치

PyTorch의 CUDA 빌드 (`torch==2.5.1+cu121`)는 표준 PyPI에서 제공되지 않습니다. 
PyTorch의 특별한 인덱스에서 설치해야 합니다.

## 해결 방법

### 방법 1: 자동 설치 스크립트 사용 (권장)

```bash
# 1. Conda 환경 생성
conda env create -f environment.yml

# 2. 환경 활성화
conda activate biscuit

# 3. PyTorch 설치 스크립트 실행
bash scripts/setup_pytorch.sh
```

### 방법 2: 수동 설치

```bash
# 1. Conda 환경 생성
conda env create -f environment.yml

# 2. 환경 활성화
conda activate biscuit

# 3. PyTorch CUDA 빌드 설치
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

### 방법 3: Conda로 설치

```bash
# 1. Conda 환경 생성 (PyTorch 제외)
conda env create -f environment.yml

# 2. 환경 활성화
conda activate biscuit

# 3. Conda로 PyTorch 설치
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    pytorch-cuda=12.1 -c pytorch -c nvidia
```

## CUDA 버전 확인

설치 전에 시스템의 CUDA 버전을 확인하세요:

```bash
nvidia-smi
```

출력에서 "CUDA Version"을 확인합니다. 예: `CUDA Version: 12.8`

## PyTorch CUDA 버전 호환성

- **CUDA 12.1**: `--index-url https://download.pytorch.org/whl/cu121`
- **CUDA 11.8**: `--index-url https://download.pytorch.org/whl/cu118`
- **CUDA 12.4**: `--index-url https://download.pytorch.org/whl/cu124`

참고: PyTorch는 하위 호환성을 지원하므로 CUDA 12.8 시스템에서 CUDA 12.1 빌드를 사용할 수 있습니다.

## 설치 확인

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

## 문제 해결

### 오류: "Could not find a version that satisfies the requirement torch==2.5.1+cu121"

**원인**: PyPI에서 CUDA 빌드를 직접 설치할 수 없음

**해결**:
1. `environment.yml`에서 PyTorch 관련 라인을 제거했는지 확인
2. 별도로 PyTorch 인덱스에서 설치:
   ```bash
   pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
   ```

### 오류: "CUDA is not available"

**원인**: CUDA 드라이버 또는 toolkit이 설치되지 않음

**해결**:
1. NVIDIA 드라이버 확인: `nvidia-smi`
2. CUDA toolkit 확인: `nvcc --version`
3. PyTorch CUDA 버전이 시스템 CUDA와 호환되는지 확인

### 오류: "No module named 'torch'"

**원인**: PyTorch가 설치되지 않음

**해결**:
1. Conda 환경이 활성화되었는지 확인: `conda activate biscuit`
2. PyTorch 설치 스크립트 실행: `bash scripts/setup_pytorch.sh`

## 참고 자료

- [PyTorch 공식 설치 가이드](https://pytorch.org/get-started/locally/)
- [PyTorch CUDA 호환성](https://pytorch.org/get-started/previous-versions/)
