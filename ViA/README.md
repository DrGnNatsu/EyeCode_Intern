# Vision in Action (ViA) Diffusion Policy

> **A Foundation-Model-Guided Diffusion Policy for Bi-Manual Robotic Manipulation (23-DOF Action Space)**  
> Combines Meta's **DINOv2** visual representation with a **1D Conditional U-Net Diffusion Policy** to predict multi-step robot trajectories directly from camera observations.

---

## Table of Contents

- [Overview & Architecture](#overview--architecture)
- [Directory Structure](#directory-structure)
- [1. Environment Setup](#1-environment-setup)
  - [Prerequisites](#prerequisites)
  - [Quick Automated Setup (Windows)](#quick-automated-setup-windows)
  - [Quick Automated Setup (Ubuntu/Linux)](#quick-automated-setup-ubuntulinux)
  - [Manual Setup with `uv` or `pip`](#manual-setup-with-uv-or-pip)
- [2. Pretrained Weights & Dataset](#2-pretrained-weights--dataset)
- [3. Training Policy Models](#3-training-policy-models)
  - [Windows Training](#windows-training)
  - [Ubuntu / Linux Training](#ubuntu--linux-training)
  - [Training Outputs & Separated Logging](#training-outputs--separated-logging)
  - [Key Configuration Parameters](#key-configuration-parameters)
- [4. Benchmarking & Empirical Evaluation](#4-benchmarking--empirical-evaluation)
- [5. Inference & Visualization](#5-inference--visualization)
  - [Single-Sample Inference (6-Panel Dashboard)](#single-sample-inference-6-panel-dashboard)
  - [Understanding the 6 Dashboard Panels](#understanding-the-6-dashboard-panels)
  - [Generating 16-Step Video Clips & Frame Dumps](#generating-16-step-video-clips--frame-dumps)
- [Hardware & Troubleshooting Guide](#hardware--troubleshooting-guide)

---

## Overview & Architecture

ViA maps RGB camera observations to a **16-step future trajectory chunk** across **23 degrees of freedom (DOF)**:

- **Active Perception Neck (7 DOF, Joints 0–6):** Controls gaze tracking and head orientation.
- **Right Arm Manipulation (8 DOF, Joints 7–14):** Primary arm for reaching and grasping (including right gripper).
- **Left Arm Manipulation (8 DOF, Joints 15–22):** Secondary arm for bi-manual coordination (including left gripper).

```text
[ Camera Image ] ──> [ DINOv2 ViT-S/14 ] ──> Visual Feature Condition ──┐
                                                                       v
[ Random Gaussian Noise a_k ~ N(0, I) ] ──> [ Conditional 1D U-Net ] ──> [ 16-Step Action Trajectory ]
                                                    (16 DDIM steps)           (23 DOF per step)
```

---

## Directory Structure

```text
ViA/
├── README.md                      # This comprehensive guide
├── Statistics.md                  # Standardized metrics & benchmark documentation
└── via_diffusion_policy/
    ├── config/
    │   ├── train_windows_small.yaml # Windows training configuration
    │   └── train_ubuntu_small.yaml  # Ubuntu training configuration
    ├── dataset/
    │   ├── cup_train.zarr.zip       # Training Zarr demonstration dataset
    │   └── cup_val.zarr.zip         # Validation Zarr demonstration dataset
    ├── weights/
    │   └── dinov2_vits14_pretrain.pth # Pretrained DINOv2 backbone
    ├── via_model_ckps/              # Model checkpoints directory (*.pth only)
    │   └── via_dinov2_cls_cup_ema_0.pth
    ├── training_logs/               # Training logs, curves & CSV metrics
    │   ├── train.log
    │   ├── training_metrics.csv
    │   ├── training_summary.json
    │   └── loss_curves.png
    ├── inference_output/            # Visual dashboard PNGs, JSONs & MP4 videos
    ├── benchmark_evidence/          # Benchmark reports & distribution plots
    ├── train_windows_small.py       # Training engine for Windows
    ├── train_ubuntu_small.py        # Training engine for Ubuntu / Linux
    ├── benchmark_via.py             # Metric evaluation engine
    ├── infer_single_sample.py       # Visualizer & video generator
    ├── setup_windows.ps1            # Windows environment installer
    └── setup_ubuntu.sh              # Ubuntu environment installer
```

---

## 1. Environment Setup

### Prerequisites

- **OS:** Windows 10/11 (64-bit) or Ubuntu 20.04/22.04 LTS
- **Python:** Python 3.11 (recommended)
- **GPU:** NVIDIA GPU with CUDA support (RTX 3050 8GB, RTX 3060, RTX 4090, A100, etc.)
- **Package Manager:** [`uv`](https://github.com/astral-sh/uv) (fastest) or standard `pip`

### Quick Automated Setup (Windows)

Open PowerShell as Administrator or regular user:

```powershell
cd d:\EyeCode_Intern\ViA\via_diffusion_policy

# Run the automated installer script
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

The script will:

1. Detect NVIDIA GPU via `nvidia-smi`.
2. Install `uv` if missing.
3. Create a clean virtual environment at `.venv` (Python 3.11).
4. Install PyTorch with CUDA 11.8 wheels.
5. Install pinned dependencies (`diffusers==0.18.2`, `zarr==2.12.0`, `opencv-python`, `matplotlib`, `pyyaml`, etc.).

To activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Quick Automated Setup (Ubuntu/Linux)

```bash
cd /path/to/ViA/via_diffusion_policy

# Make script executable and run
chmod +x setup_ubuntu.sh
./setup_ubuntu.sh

# Activate virtual environment
source .venv/bin/activate
```

### Manual Setup with `uv` or `pip`

If you prefer manual setup:

```bash
# 1. Create Python 3.11 virtual environment
uv venv .venv --python 3.11

# 2. Activate
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# 3. Install PyTorch with CUDA 11.8
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Install pinned libraries
uv pip install diffusers==0.18.2 accelerate==1.7.0 zarr==2.12.0 numcodecs==0.12.1 \
               huggingface-hub==0.25.1 wandb==0.20.1 "numpy==1.26.4" \
               matplotlib==3.10.3 "opencv-python==4.9.0.80" tqdm==4.67.1 PyYAML==6.0.2 scipy
```

Verify GPU availability:

```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0))"
```

---

## 2. Pretrained Weights & Dataset

Place the compressed Zarr archives in the `dataset/` directory:

- `via_diffusion_policy/dataset/cup_train.zarr.zip`
- `via_diffusion_policy/dataset/cup_val.zarr.zip`

You can follow this link to download the dataset: [ViA Dataset Download](https://drive.google.com/drive/folders/1qJbXjk8bEvlg3eARPnNTUiw5PcgMhmJy)

---

## 3. Training Policy Models

### Windows Training

Run training with TensorFloat-32 (TF32) acceleration:

```powershell
cd d:\EyeCode_Intern\ViA\via_diffusion_policy
.\.venv\Scripts\Activate.ps1

python train_windows_small.py --config config/train_windows_small.yaml
```

To override directories from the command line:

```powershell
python train_windows_small.py `
    --config config/train_windows_small.yaml `
    --save-dir via_model_ckps `
    --log-dir training_logs
```

### Ubuntu / Linux Training

```bash
cd /path/to/ViA/via_diffusion_policy
source .venv/bin/activate

python train_ubuntu_small.py --config config/train_ubuntu_small.yaml
```

### Training Outputs & Separated Logging

Checkpoints and logs are strictly isolated:

- **Model Weights (`via_model_ckps/`):**
  - `via_dinov2_cls_cup_0.pth` (Raw optimizer weights after epoch 0)
  - `via_dinov2_cls_cup_ema_0.pth` (Exponential Moving Average weights used for inference)
- **Training Artifacts (`training_logs/`):**
  - `train.log`: Detailed runtime log with timestamps and system info.
  - `training_metrics.csv`: Per-epoch metrics table (loss, validation MSE, LR, epoch time).
  - `training_summary.json`: High-level summary (total time, best MSE, final loss).
  - `loss_curves.png`: Training loss vs. Validation MSE plots.
  - `dataset_preview.png`: Sample batch camera observations and trajectory previews.

### Key Configuration Parameters

Inspect or edit `config/train_windows_small.yaml`:

```yaml
# Hardware & Training Schedule
batch_size: 16               # Batch size (set to 8 or 16 for 8GB VRAM)
num_epochs: 10               # Number of training epochs
learning_rate: 1.0e-4        # AdamW learning rate
weight_decay: 1.0e-6         # AdamW weight decay
warmup_epochs: 1             # Warmup period before cosine annealing

# Diffusion Horizon & Action Space
pred_action_horizon: 16      # Number of future steps to predict
action_horizon: 8            # Number of steps executed before replanning (RHC)
obs_horizon: 1               # Number of observation frames
action_dim: 23               # 23 DOF (7 neck + 8 right arm + 8 left arm)

# Denoising Diffusion Parameters
train_diffusion_steps: 50    # DDIM noise schedule timesteps for training
test_diffusion_steps: 16     # Fast DDIM steps for validation & inference
ema_power: 0.75              # EMA weight update rate
```

---

## 4. Benchmarking & Empirical Evaluation

To calculate the full statistical profile according to [`Statistics.md`](Statistics.md) (Overall Action MSE, MAE, Subsystem Errors, Trajectory Jerk, CUDA Latency, and Memory Footprint):

```powershell
python benchmark_via.py `
    --config config/train_windows_small.yaml `
    --checkpoint via_model_ckps/via_dinov2_cls_cup_ema_0.pth `
    --dataset dataset/cup_val.zarr.zip `
    --output-dir benchmark_evidence `
    --num-batches 10
```

### Generated Benchmark Artifacts (`benchmark_evidence/`)

- `benchmark_report.json`: Machine-readable metrics (MSE, MAE in rad & deg, Latency percentiles, Jerk).
- `benchmark_summary.md`: Markdown summary table ready for copy-pasting into reports.
- `benchmark_metrics.csv`: Raw per-sample error distribution.
- `plots/`: Histograms and trajectory comparison curves.

---

## 5. Inference & Visualization

### Single-Sample Inference (6-Panel Dashboard)

To evaluate a single observation frame and generate a high-resolution 6-panel diagnostic dashboard:

```powershell
python infer_single_sample.py `
    --checkpoint via_model_ckps/via_dinov2_cls_cup_ema_0.pth `
    --dataset dataset/cup_val.zarr.zip `
    --sample-idx 0 `
    --output-dir inference_output
```

This generates:

- `inference_output/inference_sample_0.png` (Static 6-panel analysis dashboard)
- `inference_output/inference_sample_0.json` (Exact metrics: MSE, MAE, Latency, Peak VRAM)

### Understanding the 6 Dashboard Panels

| Panel | Name | Description |
| :---: | :--- | :--- |
| **1** | **Camera Observation & HUD** | Shows the input RGB image fed into DINOv2 with overlaid HUD displaying Inference Latency (ms), Control Rate (Hz), and Action MSE. |
| **2** | **Overall Action Trajectory** | Compares the 16-step trajectory mean across all 23 degrees of freedom (Ground Truth vs. Model Prediction with ±1 Std deviation envelope). |
| **3** | **Active Perception Neck (7D)** | Joint-by-joint curves for Pan, Tilt, and Roll camera tracking (Joints 0–6). |
| **4** | **Right Arm Manipulation (8D)** | Joint-by-joint trajectories for the primary reaching and grasping arm (Joints 7–14). |
| **5** | **Left Arm Manipulation (8D)** | Joint-by-joint trajectories for the bi-manual support arm (Joints 15–22). |
| **6** | **Grippers Actuation (2D)** | Right gripper (Joint 14) and Left gripper (Joint 22) open/close status over time. |

### Generating 16-Step Video Clips & Frame Dumps

Add the `--video` and `--save-frames` flags to animate the predicted 16-step trajectory into an MP4 video clip:

```powershell
python infer_single_sample.py `
    --checkpoint via_model_ckps/via_dinov2_cls_cup_ema_0.pth `
    --sample-idx 0 `
    --video `
    --save-frames `
    --output-dir inference_output
```

This produces:

- `inference_output/inference_sample_0.mp4`: A 10 FPS video clip showing the robot camera view with a live HUD and synchronized joint movement indicators.
- `inference_output/sample_0_frames/frame_00.png` ... `frame_15.png`: 16 individual frame dumps suitable for reports or presentations.

---

## Hardware & Troubleshooting Guide

### 1. Enabling TensorFloat-32 (TF32) for NVIDIA RTX 30xx / 40xx / Ada / Hopper

Both `train_windows_small.py` and `infer_single_sample.py` automatically enable TF32 precision:

```python
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
```

This yields a **3x–8x speedup** on matrix multiplications with zero loss in training convergence.

### 2. Out of Memory (OOM) on 8GB GPUs

If you experience CUDA OOM on 8GB cards (e.g., RTX 3050/3060):

- Reduce `batch_size: 8` in `config/train_windows_small.yaml`.
- Ensure other GPU applications are closed.
- Keep `test_diffusion_steps: 16` during evaluation instead of 50.

### 3. Windows Terminal Character Encoding

All inference and benchmark scripts use pure ASCII terminal log format (`[+]`, `[-]`, `[OK]`, `deg`, `rad^2`) to prevent `UnicodeEncodeError` crashes on Windows `cp1252` code pages.
