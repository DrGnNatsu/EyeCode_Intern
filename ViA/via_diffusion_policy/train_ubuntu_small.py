#!/usr/bin/env python3
"""
Vision in Action (ViA) Diffusion Policy Training Script for Ubuntu Server.

Features:
- Optimized for Linux/Ubuntu (multi-worker DataLoader, persistent workers, pin_memory).
- Mixed precision support (FP16/BF16/FP32) via Accelerate for high-throughput training.
- Headless execution (no X11 / GUI required).
- Configurable via command-line arguments (override epochs, batch size, learning rate, workers).
- Signal handling (catches SIGINT/SIGTERM to save emergency checkpoint before termination).
- Offline or online WandB logging support.
"""

import os
import sys
import yaml
import signal
import argparse
import datetime
import time
import json
import csv
import collections
import numpy as np
import zarr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from tqdm import tqdm
import wandb
import matplotlib
matplotlib.use('Agg')  # Headless backend for Ubuntu server
import matplotlib.pyplot as plt

from models import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from utils import (
    normalize_data,
    unnormalize_data,
    normalize_image_imagenet,
    create_sample_indices,
    sample_sequence,
    get_data_stats,
)

# Global reference for graceful shutdown handler
CURRENT_TRAINER = None


def signal_handler(sig, frame):
    print(f"\n[!] Signal {sig} received. Performing graceful shutdown...")
    if CURRENT_TRAINER is not None:
        CURRENT_TRAINER.save_emergency_checkpoint()
    sys.exit(0)


def load_from_zarr(save_path):
    """Load all keys from a Zarr archive into memory."""
    zarr_store = zarr.open(save_path, mode='r')
    data = {}
    for key in zarr_store:
        data[key] = zarr_store[key][:]
    return data


class ViAImageDataset(Dataset):
    def __init__(self, dataset_path: str, pred_horizon: int, obs_horizon: int, action_horizon: int):
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Loading training dataset: {dataset_path}")
        loaded_dataset = load_from_zarr(dataset_path)
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Dataset successfully loaded into RAM.")

        train_main_image_data = np.array(loaded_dataset["main-image"]).astype(np.float32)
        train_main_image_data /= 255.0
        train_main_image_data = np.moveaxis(train_main_image_data, -1, 1)  # (N, H, W, 3) -> (N, 3, H, W)
        train_main_image_data = normalize_image_imagenet(train_main_image_data)

        train_data = {
            'agent_pos': np.array(loaded_dataset["proprio"]).astype(np.float32),
            'action': np.array(loaded_dataset["action"]).astype(np.float32)
        }
        episode_ends = np.array(loaded_dataset["episode_ends"])
        print(f"Total demonstration episodes: {len(episode_ends)}, Total frames: {len(train_main_image_data)}")

        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1
        )

        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        normalized_train_data['main-image'] = train_main_image_data

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        b_start, b_end, s_start, s_end = self.indices[idx]
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=b_start,
            buffer_end_idx=b_end,
            sample_start_idx=s_start,
            sample_end_idx=s_end
        )
        nsample['main-image'] = nsample['main-image'][:self.obs_horizon, :]
        nsample['agent_pos'] = nsample['agent_pos'][:self.obs_horizon, :]
        return nsample


def unnormalize_imagenet(image):
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    image = image * std + mean
    image = image[[2, 1, 0], :, :]
    return np.clip(image, 0, 1)


def save_dataset_preview(dataloader, save_path="dataset_preview.png", n_samples=4):
    sample_batch = next(iter(dataloader))
    sample_images = sample_batch["main-image"][:n_samples]
    plt.figure(figsize=(4 * n_samples, 4))
    for i in range(n_samples):
        img = sample_images[i, 0].cpu().numpy()
        img = unnormalize_imagenet(img).transpose(1, 2, 0)
        plt.subplot(1, n_samples, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Sample {i}")
    plt.tight_layout()
    plt.suptitle("First Timestep of main-image (Training Split)", y=1.05)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved dataset preview image to: {save_path}")


class UbuntuPolicyTrainer:
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.accelerator = Accelerator(mixed_precision=args.mixed_precision)
        self.device = self.accelerator.device

        # Setup paths & output directories
        self.save_dir = args.save_dir or config.get("model_save_dir", "via_model_ckps")
        os.makedirs(self.save_dir, exist_ok=True)

        self.log_dir = args.log_dir or config.get("training_log_dir", "training_logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # File logger
        self.log_file = os.path.join(self.log_dir, "train.log")
        def log_msg(msg):
            print(msg)
            with open(self.log_file, "a", encoding="utf-8") as lf:
                lf.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
        self.log_msg = log_msg

        self.log_msg(f"Checkpoints directory: {self.save_dir}")
        self.log_msg(f"Training logs directory: {self.log_dir}")

        # Initialize CSV metrics log
        self.csv_metrics_path = os.path.join(self.log_dir, "training_metrics.csv")
        with open(self.csv_metrics_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_action_mse", "learning_rate", "epoch_time_sec", "peak_vram_mb", "timestamp"])
        self.metrics_history = []

        # Enable TensorFloat32 (TF32) for fast matmul & convs on Ampere+ GPUs (RTX 30-series, A100, H100)
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('high')
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.log_msg("TF32 enabled: torch.set_float32_matmul_precision('high')")

        self.log_msg(f"Accelerator initialized | Device: {self.device} | Mixed precision: {self.accelerator.mixed_precision}")

        # Load Dataset
        self.dataset = ViAImageDataset(
            dataset_path=config["dataset_path"],
            pred_horizon=config["pred_horizon"],
            obs_horizon=config["obs_horizon"],
            action_horizon=config["action_horizon"]
        )

        num_workers = args.num_workers if args.num_workers is not None else config.get("num_workers", 4)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=args.batch_size or config["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0)
        )
        self.stats = self.dataset.stats
        self.log_msg(f"DataLoader configured with {num_workers} workers, batch_size={args.batch_size or config['batch_size']}")

        save_dataset_preview(self.dataloader, save_path=os.path.join(self.log_dir, "dataset_preview.png"))

        # Load Validation Data
        val_path = config["val_dataset_path"]
        self.log_msg(f"Loading validation dataset: {val_path}")
        val_zarr = load_from_zarr(val_path)
        val_ep_ends = np.array(val_zarr["episode_ends"])
        self.val_data = {
            'agent_pos': np.array(val_zarr["proprio"][:val_ep_ends[0]]).astype(np.float32),
            'action': np.array(val_zarr["action"][:val_ep_ends[0]]).astype(np.float32),
            'main-image': np.array(val_zarr["main-image"][:val_ep_ends[0]]).astype(np.float32) / 255.0
        }
        self.log_msg(f"Validation episode loaded: {len(self.val_data['action'])} timesteps")

        # Models
        self.log_msg("Loading DINOv2 vision encoder...")
        vision_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        vision_encoder = vision_encoder.to(self.device)

        noise_pred_net = ConditionalUnet1D(
            input_dim=config["action_dim"],
            global_cond_dim=config["obs_dim"] * config["obs_horizon"]
        )

        self.nets = nn.ModuleDict({
            'vision_encoder': vision_encoder,
            'noise_pred_net': noise_pred_net
        })

        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=config["num_train_DDIM_timesteps"],
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='epsilon'
        )

        self.ema = EMAModel(parameters=self.nets.parameters(), power=0.75)
        lr = args.lr or config["learning_rate"]
        self.optimizer = torch.optim.AdamW(self.nets.parameters(), lr=lr, weight_decay=1e-6)

        total_epochs = args.epochs or config["num_epochs"]
        self.lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=self.optimizer,
            num_warmup_steps=min(500, len(self.dataloader) // 2),
            num_training_steps=len(self.dataloader) * total_epochs
        )

        # Prepare for Accelerate
        self.ema, self.optimizer, self.dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.ema, self.optimizer, self.dataloader, self.lr_scheduler
        )
        self.nets.to(self.device)
        self.ema.to(self.device)

        # Resume from checkpoint if specified
        if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
            self.log_msg(f"Resuming weights from: {args.resume_checkpoint}")
            state_dict = torch.load(args.resume_checkpoint, map_location=self.device)
            unwrapped = self.accelerator.unwrap_model(self.nets)
            unwrapped.load_state_dict(state_dict)
            self.ema.copy_to(unwrapped.parameters())

        # WandB Setup
        wandb_mode = "disabled" if args.no_wandb else os.environ.get("WANDB_MODE", "offline")
        wandb.init(
            project=config.get("wandb_project", "via_diffusion_policy"),
            dir=self.log_dir,
            mode=wandb_mode,
            config={
                "epochs": total_epochs,
                "batch_size": args.batch_size or config["batch_size"],
                "learning_rate": lr,
                "dataset": config["dataset_path"],
                "mixed_precision": args.mixed_precision
            }
        )

        self.current_epoch = 0

    def save_emergency_checkpoint(self):
        ckpt_path = os.path.join(self.save_dir, f"{self.config['model_name']}_emergency_epoch_{self.current_epoch}.pth")
        unwrapped = self.accelerator.unwrap_model(self.nets)
        torch.save(unwrapped.state_dict(), ckpt_path)
        print(f"[!] Emergency checkpoint safely saved to: {ckpt_path}")

    def evaluate(self, epoch_idx):
        self.log_msg(f"\n--- Running Validation at Epoch {epoch_idx} ---")
        unwrapped = self.accelerator.unwrap_model(self.nets)
        unwrapped.eval()

        val_loss = []
        obs = {
            'main-image': self.val_data['main-image'][0],
            'agent_pos': self.val_data['agent_pos'][0]
        }
        obs_deque = collections.deque([obs] * self.config["obs_horizon"], maxlen=self.config["obs_horizon"])

        step_idx = 0
        total_eval_steps = int(len(self.val_data['main-image']) / 8) - 2

        with torch.no_grad():
            for _ in range(total_eval_steps):
                main_img = np.stack([x['main-image'] for x in obs_deque])
                main_img = normalize_image_imagenet(main_img)
                main_img = np.moveaxis(main_img, -1, 1)
                t_img = torch.from_numpy(main_img).to(self.device, dtype=torch.float32)

                agent_poses = np.stack([x['agent_pos'] for x in obs_deque])
                nagent_poses = normalize_data(agent_poses, stats=self.stats['agent_pos'])
                t_agent = torch.from_numpy(nagent_poses).to(self.device, dtype=torch.float32)

                embeddings = unwrapped['vision_encoder'].forward_features(t_img)
                main_feat = embeddings["x_norm_clstoken"]
                obs_cond = torch.cat([main_feat, t_agent], dim=-1).unsqueeze(0).flatten(start_dim=1)

                noisy_action = torch.randn((1, self.config["pred_horizon"], self.config["action_dim"]), device=self.device)
                naction = noisy_action

                self.noise_scheduler.set_timesteps(self.config["num_test_DDIM_timesteps"])
                for k in self.noise_scheduler.timesteps:
                    noise_pred = unwrapped['noise_pred_net'](sample=naction, timestep=k, global_cond=obs_cond)
                    naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample

                naction_np = naction.detach().to('cpu').numpy()[0]
                action_pred = unnormalize_data(naction_np, stats=self.stats['action'])

                start = self.config["obs_horizon"] - 1
                end = start + self.config["action_horizon"]
                action = action_pred[start:end, :]

                for i in range(len(action)):
                    gt_act = self.val_data['action'][step_idx]
                    cur_act = action[i]
                    val_loss.append(nn.functional.mse_loss(torch.tensor(gt_act), torch.tensor(cur_act)).item())
                    step_idx += 1
                    obs_deque.append({
                        "main-image": self.val_data['main-image'][step_idx],
                        "agent_pos": self.val_data['agent_pos'][step_idx]
                    })

        avg_loss = float(np.mean(val_loss))
        self.log_msg(f"Validation Action MSE: {avg_loss:.6f}")
        wandb.log({"val_epoch_action_error": avg_loss, "epoch": epoch_idx})

        ckpt_path = os.path.join(self.save_dir, f"{self.config['model_name']}_ema_{epoch_idx}.pth")
        torch.save(unwrapped.state_dict(), ckpt_path)
        self.log_msg(f"Checkpoint saved: {ckpt_path}\n")
        return avg_loss

    def train(self):
        total_epochs = self.args.epochs or self.config["num_epochs"]
        val_freq = self.config.get("validate_every_epoch", 1)
        total_start_time = time.time()

        self.log_msg(f"Commencing training for {total_epochs} epochs...")

        for epoch_idx in range(total_epochs):
            self.current_epoch = epoch_idx
            epoch_start_time = time.time()
            val_loss = None

            # Periodic validation
            if epoch_idx % val_freq == 0:
                val_loss = self.evaluate(epoch_idx)

            epoch_loss = []
            self.nets.train()

            progress_bar = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch_idx}/{total_epochs}",
                leave=True,
                dynamic_ncols=True,
                file=sys.stdout
            )

            for step_i, nbatch in enumerate(progress_bar):
                n_main_image = nbatch['main-image'][:, :self.config["obs_horizon"]].to(self.device)
                nagent_pos = nbatch['agent_pos'][:, :self.config["obs_horizon"]].to(self.device)
                naction = nbatch['action'].to(self.device)
                B = nagent_pos.shape[0]

                n_main_image = n_main_image.flatten(end_dim=1)
                embeddings = self.nets['vision_encoder'].forward_features(n_main_image)
                main_image_features = embeddings["x_norm_clstoken"]
                nagent_pos = nagent_pos.flatten(end_dim=1)

                obs_cond = torch.cat([main_image_features, nagent_pos], dim=-1)

                noise = torch.randn(naction.shape, device=self.device)
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps,
                    (B,), device=self.device
                ).long()

                noisy_actions = self.noise_scheduler.add_noise(naction, noise, timesteps)
                noise_pred = self.nets['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)

                loss = nn.functional.mse_loss(noise_pred, noise)

                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                self.ema.step(self.nets.parameters())

                loss_val = loss.item()
                epoch_loss.append(loss_val)
                progress_bar.set_postfix(batch_loss=f"{loss_val:.4f}")
                wandb.log({"train_batch_loss": loss_val, "epoch": epoch_idx})
                if (step_i + 1) % 25 == 0:
                    sys.stdout.flush()

            avg_epoch_loss = float(np.mean(epoch_loss))
            epoch_duration = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

            # Record per-epoch metrics to CSV
            with open(self.csv_metrics_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch_idx,
                    f"{avg_epoch_loss:.6f}",
                    f"{val_loss:.6f}" if val_loss is not None else "",
                    f"{current_lr:.6e}",
                    f"{epoch_duration:.2f}",
                    f"{peak_vram:.2f}",
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ])

            self.metrics_history.append({
                "epoch": epoch_idx,
                "train_loss": avg_epoch_loss,
                "val_loss": val_loss,
                "lr": current_lr,
                "epoch_time_sec": epoch_duration,
                "peak_vram_mb": peak_vram
            })

            wandb.log({"train_epoch_loss": avg_epoch_loss, "epoch": epoch_idx})
            self.log_msg(f"Epoch {epoch_idx} completed | Avg Loss: {avg_epoch_loss:.6f} (Time: {epoch_duration:.1f}s, Peak VRAM: {peak_vram:.1f}MB)")

        # Final checkpoint
        final_ckpt = os.path.join(self.save_dir, f"{self.config['model_name']}_ema_final.pth")
        unwrapped = self.accelerator.unwrap_model(self.nets)
        torch.save(unwrapped.state_dict(), final_ckpt)
        self.log_msg(f"Training complete. Final checkpoint saved to: {final_ckpt}")

        # Save summary JSON
        total_duration = time.time() - total_start_time
        summary_data = {
            "model_name": self.config.get("model_name", "via_dinov2_cls_cup"),
            "timestamp": datetime.datetime.now().isoformat(),
            "config": {
                "num_epochs": total_epochs,
                "batch_size": self.args.batch_size or self.config["batch_size"],
                "learning_rate": self.args.lr or self.config["learning_rate"],
                "pred_horizon": self.config["pred_horizon"],
                "obs_horizon": self.config["obs_horizon"],
                "action_horizon": self.config["action_horizon"],
                "mixed_precision": self.args.mixed_precision
            },
            "total_training_time_sec": round(total_duration, 2),
            "final_train_loss": avg_epoch_loss if 'avg_epoch_loss' in locals() else None,
            "best_val_loss": min([m["val_loss"] for m in self.metrics_history if m["val_loss"] is not None], default=None),
            "model_save_dir": self.save_dir,
            "training_log_dir": self.log_dir,
            "epochs_history": self.metrics_history
        }
        summary_json_path = os.path.join(self.log_dir, "training_summary.json")
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=4)
        self.log_msg(f"Training summary saved to: {summary_json_path}")

        # Plot loss curves
        if self.metrics_history:
            epochs = [m["epoch"] for m in self.metrics_history]
            train_losses = [m["train_loss"] for m in self.metrics_history]
            val_records = [(m["epoch"], m["val_loss"]) for m in self.metrics_history if m["val_loss"] is not None]

            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(epochs, train_losses, 'b-o', label='Train Loss (MSE)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss per Epoch')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()

            plt.subplot(1, 2, 2)
            if val_records:
                v_epochs, v_losses = zip(*val_records)
                plt.plot(v_epochs, v_losses, 'r-s', label='Val Action MSE')
                plt.xlabel('Epoch')
                plt.ylabel('MSE')
                plt.title('Validation Action MSE')
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.legend()
            plt.tight_layout()
            plot_path = os.path.join(self.log_dir, "loss_curves.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            self.log_msg(f"Loss curves saved to: {plot_path}")

        self.log_msg(f"All outputs organized:\n  - Checkpoints: {self.save_dir}\n  - Logs & Metrics: {self.log_dir}")


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(description="ViA Diffusion Policy Training for Ubuntu Server")
    parser.add_argument('--config', type=str, default="config/train_ubuntu_small.yaml", help="Path to config YAML")
    parser.add_argument('--epochs', type=int, default=None, help="Override number of training epochs")
    parser.add_argument('--batch-size', type=int, default=None, help="Override batch size")
    parser.add_argument('--lr', type=float, default=None, help="Override learning rate")
    parser.add_argument('--num-workers', type=int, default=None, help="DataLoader worker processes (default 4 on Linux)")
    parser.add_argument('--mixed-precision', type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Mixed precision mode")
    parser.add_argument('--save-dir', type=str, default=None, help="Directory to save checkpoints")
    parser.add_argument('--log-dir', type=str, default=None, help="Directory to save training logs, metrics CSV, and plots")
    parser.add_argument('--resume-checkpoint', type=str, default=None, help="Path to checkpoint .pth to resume from")
    parser.add_argument('--no-wandb', action='store_true', help="Disable WandB logging entirely")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    global CURRENT_TRAINER
    CURRENT_TRAINER = UbuntuPolicyTrainer(config, args)
    CURRENT_TRAINER.train()


if __name__ == "__main__":
    main()

