#!/bin/bash
# ==============================================================================
# Vision in Action (ViA) - Ubuntu Server Automated Environment Setup Script
# ==============================================================================

set -e

echo "=== [1/5] Checking GPU & System Hardware ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "[!] Warning: nvidia-smi not found. Training will fall back to CPU if NVIDIA drivers are missing."
fi

echo "=== [2/5] Ensuring 'uv' Fast Package Manager is Installed ==="
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv version: $(uv --version)"

echo "=== [3/5] Creating Python 3.11 Virtual Environment (.venv) ==="
if [ ! -d ".venv" ]; then
    uv venv .venv --python 3.11
fi

source .venv/bin/activate
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

echo "=== [4/5] Installing PyTorch with CUDA 11.8 ==="
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo "=== [5/5] Installing Exact Pinned Diffusion Policy Dependencies ==="
uv pip install \
    diffusers==0.18.2 \
    accelerate==1.7.0 \
    zarr==2.12.0 \
    numcodecs==0.12.1 \
    huggingface-hub==0.25.1 \
    wandb==0.20.1 \
    "numpy==1.26.4" \
    matplotlib==3.10.3 \
    opencv-python-headless \
    tqdm==4.67.1 \
    PyYAML==6.0.2 \
    scipy

echo "=== Environment Verification ==="
python -c "import torch; print('PyTorch Version:', torch.__version__, '| CUDA Available:', torch.cuda.is_available(), '| Device Count:', torch.cuda.device_count())"

echo "=============================================================================="
echo " Setup complete! To activate the environment in your shell, run:"
echo "     source .venv/bin/activate"
echo "=============================================================================="
