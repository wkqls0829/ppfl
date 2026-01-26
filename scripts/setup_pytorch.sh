#!/bin/bash

# PyTorch Installation Script
# Install PyTorch with CUDA support after conda environment is created
# This is needed because PyTorch CUDA builds are not available on standard PyPI

set -e

echo "=========================================="
echo "PyTorch CUDA Installation Script"
echo "=========================================="

# Check if conda environment is active
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "ERROR: No conda environment active."
    echo "Please activate your conda environment first:"
    echo "  conda activate biscuit"
    exit 1
fi

echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo ""

# Detect CUDA version
CUDA_VERSION=${1:-12.1}  # Default to CUDA 12.1
echo "Installing PyTorch for CUDA $CUDA_VERSION"
echo ""

# Install PyTorch from PyTorch index
echo "Installing PyTorch, torchvision, and torchaudio..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "=========================================="
echo "PyTorch installation completed!"
echo "=========================================="
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "If CUDA is not available, check:"
echo "  1. NVIDIA drivers are installed"
echo "  2. CUDA toolkit matches PyTorch CUDA version"
echo "  3. GPU is accessible"
