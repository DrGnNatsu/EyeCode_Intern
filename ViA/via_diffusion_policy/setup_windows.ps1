# ==============================================================================
# Vision in Action (ViA) - Windows Environment Setup Script (PowerShell)
# ==============================================================================

Write-Host "=== [1/5] Checking NVIDIA GPU Status ===" -ForegroundColor Cyan
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi
} else {
    Write-Host "[!] Warning: nvidia-smi not detected on PATH." -ForegroundColor Yellow
}

Write-Host "`n=== [2/5] Checking 'uv' Package Manager ===" -ForegroundColor Cyan
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv for Windows..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:PATH = "$HOME\.local\bin;" + $env:PATH
}
Write-Host "uv version: $(uv --version)"

Write-Host "`n=== [3/5] Creating Python 3.11 Virtual Environment (.venv) ===" -ForegroundColor Cyan
if (-not (Test-Path ".venv")) {
    uv venv .venv --python 3.11
}

Write-Host "`n=== [4/5] Installing PyTorch with CUDA 11.8 ===" -ForegroundColor Cyan
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

Write-Host "`n=== [5/5] Installing Exact Pinned Diffusion Policy Dependencies ===" -ForegroundColor Cyan
uv pip install `
    diffusers==0.18.2 `
    accelerate==1.7.0 `
    zarr==2.12.0 `
    numcodecs==0.12.1 `
    huggingface-hub==0.25.1 `
    wandb==0.20.1 `
    "numpy==1.26.4" `
    matplotlib==3.10.3 `
    "opencv-python==4.9.0.80" `
    tqdm==4.67.1 `
    PyYAML==6.0.2 `
    scipy

Write-Host "`n=== Verifying Environment ===" -ForegroundColor Green
& ".\.venv\Scripts\python.exe" -c "import torch; print('PyTorch Version:', torch.__version__, '| CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

Write-Host "`n==============================================================================" -ForegroundColor Green
Write-Host " Setup complete on Windows!" -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Green
