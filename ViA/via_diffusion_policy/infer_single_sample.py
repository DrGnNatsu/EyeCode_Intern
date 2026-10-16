#!/usr/bin/env python3
"""
Vision in Action (ViA) - Single Sample Inference and Visualization Tool

Features:
- Loads a single observation (RGB image + 23D proprioception) from evaluation dataset.
- Executes Diffusion Policy DDIM inference to predict 16-step action trajectory (23D).
- Computes comprehensive robotics metrics (Action MSE, MAE, Subsystem breakdown, Jerk, Latency).
- Generates a publication-quality 6-panel visualization dashboard:
  1. Camera RGB Observation
  2. Subsystem Error Bar Chart (Neck vs. Right Arm vs. Left Arm vs. Grippers)
  3. Joint-by-Timestep Absolute Error Heatmap (23 joints x 16 timesteps)
  4. Neck Joints Trajectory Tracking (Ground Truth vs. Predicted)
  5. Right Arm Trajectory Tracking (Ground Truth vs. Predicted)
  6. Left Arm Trajectory Tracking (Ground Truth vs. Predicted)
- Exports high-resolution PNG image and structured JSON results.
"""

import os
import sys
import yaml
import json
import time
import argparse
import datetime
import collections
import numpy as np
import zarr
import cv2
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Headless backend; interactive display handled via --show
import matplotlib.pyplot as plt

from models import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from utils import (
    normalize_data,
    unnormalize_data,
    normalize_image_imagenet,
    get_data_stats,
)


def load_from_zarr(save_path):
    """Load all keys from a Zarr archive into memory."""
    zarr_store = zarr.open(save_path, mode='r')
    data = {}
    for key in zarr_store:
        data[key] = zarr_store[key][:]
    return data


def compute_trajectory_jerk(trajectory: np.ndarray) -> float:
    """Computes mean squared jerk (acceleration variation) over a trajectory."""
    if len(trajectory) < 3:
        return 0.0
    # Second-order discrete difference
    accel_diff = np.diff(trajectory, n=2, axis=0)
    return float(np.mean(np.linalg.norm(accel_diff, axis=-1) ** 2))


def render_sample_video_frame(fig, ax_cam, ax_neck, ax_rarm, ax_larm,
                              image_rgb, step_t, total_h,
                              gt_action, pred_action, cur_mse,
                              sample_idx, r_grip, l_grip):
    """Renders one synchronized video frame of the single-sample trajectory."""
    # 1. Camera View
    ax_cam.clear()
    ax_cam.imshow(image_rgb)
    ax_cam.set_title(f"Sample {sample_idx} - Step {step_t}/{total_h - 1}", fontsize=11, fontweight='bold')
    ax_cam.axis('off')

    # HUD telemetry badge
    ax_cam.text(0.03, 0.93, f"Step Error: {cur_mse:.4f} rad^2", transform=ax_cam.transAxes,
                fontsize=9, color='yellow', weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))
    ax_cam.text(0.03, 0.06, f"R-Grip: {r_grip:.2f} | L-Grip: {l_grip:.2f}", transform=ax_cam.transAxes,
                fontsize=9, color='white', weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))

    timesteps = np.arange(total_h)
    cmap = matplotlib.colormaps['tab10']

    # 2. Neck
    ax_neck.clear()
    for j in range(7):
        ax_neck.plot(timesteps, gt_action[:, j], color=cmap(j), linestyle='-', alpha=0.35, linewidth=1.2)
        ax_neck.plot(timesteps[:step_t + 1], pred_action[:step_t + 1, j], color=cmap(j), linestyle='--', linewidth=1.8)
    ax_neck.axvline(step_t, color='red', linestyle='-', linewidth=1.5)
    ax_neck.set_title("Neck (Joints 0-6)  [Solid: GT | Dashed: Pred]", fontsize=10, fontweight='bold')
    ax_neck.set_xlim(0, total_h - 1)
    ax_neck.set_ylabel("Rad", fontsize=8)
    ax_neck.grid(True, linestyle='--', alpha=0.4)

    # 3. Right Arm & Gripper
    ax_rarm.clear()
    for j in range(8):
        joint_idx = 7 + j
        is_g = (joint_idx == 14)
        lw = 2.0 if is_g else 1.2
        ax_rarm.plot(timesteps, gt_action[:, joint_idx], color=cmap(j % 10), linestyle='-', alpha=0.35, linewidth=lw)
        ax_rarm.plot(timesteps[:step_t + 1], pred_action[:step_t + 1, joint_idx], color=cmap(j % 10), linestyle='--', linewidth=lw + 0.5)
    ax_rarm.axvline(step_t, color='red', linestyle='-', linewidth=1.5)
    ax_rarm.set_title("Right Arm & Gripper (Joints 7-14)  [Thick line: Gripper]", fontsize=10, fontweight='bold')
    ax_rarm.set_xlim(0, total_h - 1)
    ax_rarm.set_ylabel("Rad", fontsize=8)
    ax_rarm.grid(True, linestyle='--', alpha=0.4)

    # 4. Left Arm & Gripper
    ax_larm.clear()
    for j in range(8):
        joint_idx = 15 + j
        is_g = (joint_idx == 22)
        lw = 2.0 if is_g else 1.2
        ax_larm.plot(timesteps, gt_action[:, joint_idx], color=cmap(j % 10), linestyle='-', alpha=0.35, linewidth=lw)
        ax_larm.plot(timesteps[:step_t + 1], pred_action[:step_t + 1, joint_idx], color=cmap(j % 10), linestyle='--', linewidth=lw + 0.5)
    ax_larm.axvline(step_t, color='red', linestyle='-', linewidth=1.5)
    ax_larm.set_title("Left Arm & Gripper (Joints 15-22)  [Thick line: Gripper]", fontsize=10, fontweight='bold')
    ax_larm.set_xlim(0, total_h - 1)
    ax_larm.set_xlabel("Prediction Horizon Step (t)", fontsize=9)
    ax_larm.set_ylabel("Rad", fontsize=8)
    ax_larm.grid(True, linestyle='--', alpha=0.4)

    fig.canvas.draw()
    rgba_buffer = fig.canvas.buffer_rgba()
    img_bgr = cv2.cvtColor(np.asarray(rgba_buffer), cv2.COLOR_RGBA2BGR)
    return img_bgr


def run_single_inference(args):
    # Setup device
    device_str = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    # Enable TF32 for fast matmuls on Ampere GPUs
    if torch.cuda.is_available() and device.type == "cuda":
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print(" VISION IN ACTION (ViA) - SINGLE SAMPLE INFERENCE & VISUALIZATION")
    print("=" * 70)
    print(f"Device:           {device}")
    if torch.cuda.is_available() and device.type == "cuda":
        print(f"GPU Model:        {torch.cuda.get_device_name(0)}")
    print(f"Config:           {args.config}")
    print(f"Checkpoint:       {args.checkpoint}")
    print(f"Eval Dataset:     {args.dataset}")
    print(f"Sample Index:     {args.sample_idx}")
    print(f"DDIM Timesteps:   {args.ddim_steps}")
    print("=" * 70)

    # 1. Load normalization statistics from training dataset
    train_dataset_path = args.train_dataset or config.get("dataset_path", "dataset/cup_train.zarr.zip")
    print(f"[*] Loading normalization statistics from: {train_dataset_path}")
    train_zarr = load_from_zarr(train_dataset_path)
    stats = {
        'agent_pos': get_data_stats(np.array(train_zarr["proprio"]).astype(np.float32)),
        'action': get_data_stats(np.array(train_zarr["action"]).astype(np.float32))
    }

    # 2. Load evaluation dataset and extract target sample
    print(f"[*] Loading evaluation dataset: {args.dataset}")
    val_zarr = load_from_zarr(args.dataset)
    total_val_frames = len(val_zarr["action"])
    pred_horizon = config.get("pred_horizon", 16)
    action_horizon = config.get("action_horizon", 8)

    sample_idx = args.sample_idx
    if sample_idx >= total_val_frames:
        raise ValueError(f"Requested sample_idx {sample_idx} exceeds dataset length {total_val_frames}")

    # Extract single observation frame
    raw_img = np.array(val_zarr["main-image"][sample_idx]).astype(np.float32)
    # Check if image is in [0, 255] or [0, 1]
    if raw_img.max() > 1.0:
        display_img = raw_img / 255.0
    else:
        display_img = raw_img.copy()

    agent_pos = np.array(val_zarr["proprio"][sample_idx]).astype(np.float32)

    # Extract ground truth action trajectory
    max_h = min(pred_horizon, total_val_frames - sample_idx)
    gt_action = np.array(val_zarr["action"][sample_idx : sample_idx + max_h]).astype(np.float32)

    # 3. Load Vision Encoder & Noise Prediction Net
    print(f"[*] Loading DINOv2 vision encoder...")
    vision_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    vision_encoder = vision_encoder.to(device)

    print(f"[*] Loading 1D Diffusion Conditional U-Net...")
    noise_pred_net = ConditionalUnet1D(
        input_dim=config["action_dim"],
        global_cond_dim=config["obs_dim"] * config["obs_horizon"]
    ).to(device)

    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': noise_pred_net
    })

    print(f"[*] Loading model weights from: {args.checkpoint}")
    checkpoint_state = torch.load(args.checkpoint, map_location=device)
    nets.load_state_dict(checkpoint_state)
    nets.eval()

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=config["num_train_DDIM_timesteps"],
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type='epsilon'
    )

    # 4. Prepare Observation Tensor
    img_for_net = display_img[None, ...].copy()  # (1, H, W, 3)
    img_norm = normalize_image_imagenet(img_for_net)
    img_norm = np.moveaxis(img_norm, -1, 1)  # (1, 3, H, W)
    t_img = torch.from_numpy(img_norm).to(device, dtype=torch.float32)

    nagent_pos = normalize_data(agent_pos[None, :], stats=stats['agent_pos'])
    t_agent = torch.from_numpy(nagent_pos).to(device, dtype=torch.float32)

    # 5. Execute Timed Inference
    print(f"[*] Running DDIM diffusion inference ({args.ddim_steps} denoising steps)...")

    if device.type == "cuda":
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        t0 = time.perf_counter()

    with torch.no_grad():
        # Encode image features
        embeddings = nets['vision_encoder'].forward_features(t_img)
        main_feat = embeddings["x_norm_clstoken"]  # (1, 384)
        obs_cond = torch.cat([main_feat, t_agent], dim=-1).flatten(start_dim=1)  # (1, 407)

        # Sample initial Gaussian noise
        noisy_action = torch.randn((1, pred_horizon, config["action_dim"]), device=device)
        naction = noisy_action

        # Denoise with DDIM
        noise_scheduler.set_timesteps(args.ddim_steps)
        for k in noise_scheduler.timesteps:
            noise_pred = nets['noise_pred_net'](sample=naction, timestep=k, global_cond=obs_cond)
            naction = noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample

    if device.type == "cuda":
        end_event.record()
        torch.cuda.synchronize()
        latency_ms = start_event.elapsed_time(end_event)
    else:
        latency_ms = (time.perf_counter() - t0) * 1000.0

    control_hz = 1000.0 / latency_ms if latency_ms > 0 else 0.0

    # Unnormalize predicted action
    naction_np = naction.detach().to('cpu').numpy()[0]  # (pred_horizon, 23)
    pred_action = unnormalize_data(naction_np, stats=stats['action'])

    # Align comparison horizon
    eval_h = min(len(gt_action), pred_horizon)
    gt_eval = gt_action[:eval_h]
    pred_eval = pred_action[:eval_h]

    # 6. Calculate Detailed Quantitative Metrics
    diff = pred_eval - gt_eval
    abs_diff = np.abs(diff)
    sq_diff = diff ** 2

    overall_mse = float(np.mean(sq_diff))
    overall_mae = float(np.mean(abs_diff))

    # Subsystem slices
    # Neck: 0..6 (7 DOF)
    # Right Arm: 7..14 (8 DOF, joint 14 is right gripper)
    # Left Arm: 15..22 (8 DOF, joint 22 is left gripper)
    # Grippers: [14, 22] (2 DOF)
    subsystems = {
        "Neck (7D)": {"slice": slice(0, 7), "color": "#1f77b4"},
        "Right Arm (8D)": {"slice": slice(7, 15), "color": "#2ca02c"},
        "Left Arm (8D)": {"slice": slice(15, 23), "color": "#d62728"},
        "Grippers (2D)": {"indices": [14, 22], "color": "#9467bd"},
    }

    subsystem_metrics = {}
    for name, meta in subsystems.items():
        if "slice" in meta:
            sub_gt = gt_eval[:, meta["slice"]]
            sub_pred = pred_eval[:, meta["slice"]]
        else:
            sub_gt = gt_eval[:, meta["indices"]]
            sub_pred = pred_eval[:, meta["indices"]]

        sub_mse = float(np.mean((sub_pred - sub_gt) ** 2))
        sub_mae = float(np.mean(np.abs(sub_pred - sub_gt)))
        subsystem_metrics[name] = {
            "mse": sub_mse,
            "mae": sub_mae
        }

    # Jerk / Smoothness
    gt_jerk = compute_trajectory_jerk(gt_eval)
    pred_jerk = compute_trajectory_jerk(pred_eval)
    jerk_ratio = pred_jerk / (gt_jerk + 1e-8)

    # Per-step Euclidean error
    step_errors = np.linalg.norm(diff, axis=-1)

    print("\n" + "=" * 70)
    print(" INFERENCE RESULTS & METRICS SUMMARY")
    print("=" * 70)
    print(f"[-] Execution Latency:     {latency_ms:.2f} ms ({control_hz:.1f} Hz query rate)")
    print(f"[-] Overall Action MSE:    {overall_mse:.6f} rad^2")
    print(f"[-] Overall Action MAE:    {overall_mae:.6f} rad ({np.degrees(overall_mae):.2f} deg)")
    print(f"[-] Mean Step Distance:    {float(np.mean(step_errors)):.4f} rad")
    print(f"[-] Trajectory Jerk:       Pred={pred_jerk:.4f} | GT={gt_jerk:.4f} (Ratio={jerk_ratio:.2f}x)")
    print("-" * 70)
    print("Subsystem Breakdown:")
    for name, m in subsystem_metrics.items():
        print(f"  - {name:<18} MSE: {m['mse']:.6f} rad^2  |  MAE: {m['mae']:.4f} rad ({np.degrees(m['mae']):.2f} deg)")
    print("=" * 70)

    # 7. Generate Multi-Panel Visual Dashboard
    os.makedirs(args.output_dir, exist_ok=True)
    fig = plt.figure(figsize=(19, 11), dpi=150)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.32, wspace=0.25)

    # Top Super Title
    ckpt_name = os.path.basename(args.checkpoint)
    plt.suptitle(
        f"ViA Diffusion Policy Inference Sample (Index: {sample_idx})\n"
        f"Checkpoint: {ckpt_name}  |  Latency: {latency_ms:.1f}ms ({control_hz:.1f} Hz)  |  "
        f"Overall MSE: {overall_mse:.6f} rad^2  |  MAE: {overall_mae:.4f} rad",
        fontsize=13, fontweight='bold', y=0.98
    )

    timesteps = np.arange(eval_h)

    # Panel 1: Camera Observation
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(display_img)
    ax1.set_title(f"1. Robot Camera Observation\nFrame {sample_idx} (224x224 RGB)", fontsize=11, fontweight='bold')
    ax1.axis('off')

    # Panel 2: Subsystem Error Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    names = list(subsystem_metrics.keys())
    mses = [subsystem_metrics[k]["mse"] for k in names]
    maes = [subsystem_metrics[k]["mae"] for k in names]
    x_pos = np.arange(len(names))
    width = 0.35

    bar1 = ax2.bar(x_pos - width/2, mses, width, label='MSE (rad^2)', color='#3498db', alpha=0.85)
    bar2 = ax2.bar(x_pos + width/2, maes, width, label='MAE (rad)', color='#e67e22', alpha=0.85)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    ax2.set_ylabel("Error Magnitude", fontsize=10)
    ax2.set_title("2. Subsystem Error Breakdown", fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax2.legend(fontsize=9)

    for b in bar1:
        h = b.get_height()
        ax2.annotate(f"{h:.4f}", xy=(b.get_x() + b.get_width()/2, h),
                     xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=7)
    for b in bar2:
        h = b.get_height()
        ax2.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width()/2, h),
                     xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=7)

    # Panel 3: Joint Absolute Error Heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    im = ax3.imshow(abs_diff.T, aspect='auto', cmap='plasma', interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("Absolute Error |Pred - GT| (rad)", fontsize=8)
    ax3.set_title("3. 23D Joint Error Heatmap", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Prediction Horizon Step", fontsize=9)
    ax3.set_ylabel("Joint Dimension (0..22)", fontsize=9)
    ax3.set_yticks([0, 6, 7, 14, 15, 22])
    ax3.axhline(6.5, color='cyan', linestyle='--', linewidth=1.2, label='Neck / R-Arm Boundary')
    ax3.axhline(14.5, color='yellow', linestyle='--', linewidth=1.2, label='R-Arm / L-Arm Boundary')
    ax3.grid(False)

    # Panel 4: Neck Joints Trajectory (Joints 0..6)
    ax4 = fig.add_subplot(gs[1, 0])
    cmap_neck = matplotlib.colormaps['tab10']
    for j in range(7):
        ax4.plot(timesteps, gt_eval[:, j], color=cmap_neck(j), linestyle='-', linewidth=1.8, label=f'J{j} GT' if j < 4 else None)
        ax4.plot(timesteps, pred_eval[:, j], color=cmap_neck(j), linestyle='--', linewidth=1.8, label=f'J{j} Pred' if j < 4 else None)
    ax4.axvline(action_horizon - 1, color='gray', linestyle=':', label='Action Horizon' if action_horizon <= eval_h else None)
    ax4.set_title("4. Neck Tracking (Joints 0-6)\nSolid: Ground Truth | Dashed: Predicted", fontsize=10, fontweight='bold')
    ax4.set_xlabel("Horizon Step (t)", fontsize=9)
    ax4.set_ylabel("Joint Angle (rad)", fontsize=9)
    ax4.grid(True, linestyle='--', alpha=0.5)

    # Panel 5: Right Arm Trajectory (Joints 7..14)
    ax5 = fig.add_subplot(gs[1, 1])
    cmap_rarm = matplotlib.colormaps['tab10']
    for j in range(8):
        joint_idx = 7 + j
        is_gripper = (joint_idx == 14)
        lw = 2.2 if is_gripper else 1.5
        label_gt = "R-Gripper GT" if is_gripper else None
        label_pred = "R-Gripper Pred" if is_gripper else None
        ax5.plot(timesteps, gt_eval[:, joint_idx], color=cmap_rarm(j % 10), linestyle='-', linewidth=lw, label=label_gt)
        ax5.plot(timesteps, pred_eval[:, joint_idx], color=cmap_rarm(j % 10), linestyle='--', linewidth=lw, label=label_pred)
    ax5.axvline(action_horizon - 1, color='gray', linestyle=':', label='Action Horizon' if action_horizon <= eval_h else None)
    ax5.set_title("5. Right Arm & Gripper (Joints 7-14)\nSolid: Ground Truth | Dashed: Predicted", fontsize=10, fontweight='bold')
    ax5.set_xlabel("Horizon Step (t)", fontsize=9)
    ax5.set_ylabel("Joint Angle (rad)", fontsize=9)
    ax5.grid(True, linestyle='--', alpha=0.5)
    if any(joint_idx == 14 for joint_idx in range(7, 15)):
        ax5.legend(loc='upper right', fontsize=8)

    # Panel 6: Left Arm Trajectory (Joints 15..22)
    ax6 = fig.add_subplot(gs[1, 2])
    cmap_larm = matplotlib.colormaps['tab10']
    for j in range(8):
        joint_idx = 15 + j
        is_gripper = (joint_idx == 22)
        lw = 2.2 if is_gripper else 1.5
        label_gt = "L-Gripper GT" if is_gripper else None
        label_pred = "L-Gripper Pred" if is_gripper else None
        ax6.plot(timesteps, gt_eval[:, joint_idx], color=cmap_larm(j % 10), linestyle='-', linewidth=lw, label=label_gt)
        ax6.plot(timesteps, pred_eval[:, joint_idx], color=cmap_larm(j % 10), linestyle='--', linewidth=lw, label=label_pred)
    ax6.axvline(action_horizon - 1, color='gray', linestyle=':', label='Action Horizon' if action_horizon <= eval_h else None)
    ax6.set_title("6. Left Arm & Gripper (Joints 15-22)\nSolid: Ground Truth | Dashed: Predicted", fontsize=10, fontweight='bold')
    ax6.set_xlabel("Horizon Step (t)", fontsize=9)
    ax6.set_ylabel("Joint Angle (rad)", fontsize=9)
    ax6.grid(True, linestyle='--', alpha=0.5)
    if any(joint_idx == 22 for joint_idx in range(15, 23)):
        ax6.legend(loc='upper right', fontsize=8)

    # Save output image
    save_png_path = os.path.join(args.output_dir, f"inference_sample_{sample_idx}.png")
    plt.savefig(save_png_path, bbox_inches='tight', dpi=150)
    print(f"\n[+] Visualization dashboard saved to: {save_png_path}")

    # Optional interactive display
    if args.show:
        plt.show()
    plt.close(fig)

    # 8. Export Structured JSON Results
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "sample_index": sample_idx,
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "device": str(device),
        "ddim_timesteps": args.ddim_steps,
        "inference_latency_ms": round(latency_ms, 2),
        "control_frequency_hz": round(control_hz, 2),
        "overall_metrics": {
            "action_mse": round(overall_mse, 6),
            "action_mae": round(overall_mae, 6),
            "action_mae_degrees": round(float(np.degrees(overall_mae)), 2),
            "mean_step_euclidean_error": round(float(np.mean(step_errors)), 4),
            "gt_jerk": round(gt_jerk, 6),
            "pred_jerk": round(pred_jerk, 6),
            "jerk_ratio": round(jerk_ratio, 2)
        },
        "subsystem_metrics": {
            name: {
                "mse": round(m["mse"], 6),
                "mae": round(m["mae"], 6),
                "mae_degrees": round(float(np.degrees(m["mae"])), 2)
            }
            for name, m in subsystem_metrics.items()
        },
        "ground_truth_action": gt_eval.tolist(),
        "predicted_action": pred_eval.tolist(),
        "visualization_path": save_png_path
    }

    # 9. Optional: Generate Trajectory Video for this Sample
    if args.video:
        print(f"[*] Generating {eval_h}-frame trajectory video for sample {sample_idx}...")
        sample_video_path = os.path.join(args.output_dir, f"inference_sample_{sample_idx}.mp4")
        frames_dir = os.path.join(args.output_dir, f"sample_{sample_idx}_frames")
        if args.save_frames:
            os.makedirs(frames_dir, exist_ok=True)

        vid_fig = plt.figure(figsize=(15, 8), dpi=100)
        vid_gs = vid_fig.add_gridspec(3, 2, width_ratios=[1.1, 1], hspace=0.35, wspace=0.22)
        v_cam = vid_fig.add_subplot(vid_gs[:, 0])
        v_neck = vid_fig.add_subplot(vid_gs[0, 1])
        v_rarm = vid_fig.add_subplot(vid_gs[1, 1])
        v_larm = vid_fig.add_subplot(vid_gs[2, 1])
        vid_fig.suptitle(f"ViA Policy 16-Step Action Trajectory Prediction (Sample {sample_idx})",
                         fontsize=13, fontweight='bold', y=0.98)

        # Sequence of horizon frames from the dataset
        horizon_images = np.array(val_zarr["main-image"][sample_idx : sample_idx + eval_h]).astype(np.float32)
        if horizon_images.max() > 1.0:
            horizon_images /= 255.0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = None

        for t_step in range(eval_h):
            step_img = horizon_images[t_step]
            step_mse = float(np.mean((gt_eval[t_step] - pred_eval[t_step]) ** 2))
            r_g = float(pred_eval[t_step, 14])
            l_g = float(pred_eval[t_step, 22])

            fbgr = render_sample_video_frame(
                vid_fig, v_cam, v_neck, v_rarm, v_larm,
                step_img, t_step, eval_h,
                gt_eval, pred_eval, step_mse,
                sample_idx, r_g, l_g
            )

            if vw is None:
                vh, vw_w, _ = fbgr.shape
                vw = cv2.VideoWriter(sample_video_path, fourcc, args.fps, (vw_w, vh))

            vw.write(fbgr)

            if args.save_frames:
                f_path = os.path.join(frames_dir, f"frame_{t_step:02d}.png")
                cv2.imwrite(f_path, fbgr)

        if vw is not None:
            vw.release()
        plt.close(vid_fig)

        results["video_path"] = sample_video_path
        if args.save_frames:
            results["frames_dir"] = frames_dir

        print(f"[+] Sample trajectory video saved to: {sample_video_path}")
        if args.save_frames:
            print(f"[+] Exported {eval_h} individual frames to: {frames_dir}")

    save_json_path = os.path.join(args.output_dir, f"inference_sample_{sample_idx}.json")
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"[+] Structured JSON metrics saved to:  {save_json_path}\n")

    return save_png_path, save_json_path


def main():
    parser = argparse.ArgumentParser(description="ViA Diffusion Policy Single Sample Inference & Visualization")
    parser.add_argument('--config', type=str, default="config/train_windows_small.yaml", help="Path to config YAML")
    parser.add_argument('--checkpoint', type=str, default="via_model_ckps/via_dinov2_cls_cup_ema_0.pth", help="Path to model checkpoint (.pth)")
    parser.add_argument('--dataset', type=str, default="dataset/cup_val.zarr.zip", help="Path to evaluation dataset .zarr.zip")
    parser.add_argument('--train-dataset', type=str, default=None, help="Path to training dataset for normalization stats (optional)")
    parser.add_argument('--sample-idx', type=int, default=0, help="Timestep index to infer and visualize (default: 0)")
    parser.add_argument('--ddim-steps', type=int, default=16, help="DDIM diffusion denoising steps (default: 16)")
    parser.add_argument('--video', action='store_true', default=False, help="Generate MP4 video of the 16-step trajectory")
    parser.add_argument('--fps', type=int, default=4, help="Frames per second for output video (default: 4)")
    parser.add_argument('--save-frames', action='store_true', default=False, help="Save individual video frame PNGs to sample_<idx>_frames/")
    parser.add_argument('--output-dir', type=str, default="inference_output", help="Directory to save PNG, MP4, & JSON results")
    parser.add_argument('--device', type=str, default=None, help="Target device (cuda or cpu)")
    parser.add_argument('--show', action='store_true', help="Open interactive matplotlib window")
    args = parser.parse_args()

    run_single_inference(args)


if __name__ == "__main__":
    main()

