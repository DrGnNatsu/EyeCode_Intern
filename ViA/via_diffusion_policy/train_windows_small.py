import os
import yaml
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import wandb
from models import ConditionalUnet1D, get_resnet, replace_bn_with_gn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
import argparse
import torch.nn as nn
import collections
import numpy as np
from torch.utils.data import Dataset
import zarr
from utils import normalize_data, unnormalize_data, normalize_image_imagenet, create_sample_indices, sample_sequence, get_data_stats
import matplotlib.pyplot as plt
import time
import json
import csv
import datetime



def load_from_zarr(save_path):
    # Open the Zarr file
    zarr_store = zarr.open(save_path, mode='r')
    
    # Load data from the store
    data = {}
    for key in zarr_store:
        data[key] = zarr_store[key][:]
        
    return data


class ViAImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int):

        print("loading training dataset...")
        print(dataset_path)

        loaded_dateset = load_from_zarr(dataset_path)
        print("loaded!")
        train_main_image_data = np.array(loaded_dateset["main-image"]).astype(np.float32)

        train_main_image_data /= 255.0

        # Move from (N, H, W, 3) → (N, 3, H, W)
        train_main_image_data = np.moveaxis(train_main_image_data, -1, 1)

        train_main_image_data = normalize_image_imagenet(train_main_image_data)

        # (N, D)
        train_data = {
            'agent_pos': np.array(loaded_dateset["proprio"]).astype(np.float32),
            'action': np.array(loaded_dateset["action"]).astype(np.float32)
        }
        episode_ends = np.array(loaded_dateset["episode_ends"])

        print("demo numbers:", len(episode_ends))

        # compute start and end of each state-action sequence also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        # compute statistics and normalize data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        # images are already normalized
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
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get normalized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        nsample['main-image'] = nsample['main-image'][:self.obs_horizon, :]
        nsample['agent_pos'] = nsample['agent_pos'][:self.obs_horizon, :]
        return nsample


def unnormalize_imagenet(image):
    """Undo ImageNet normalization and convert RGB → BGR for correct display."""
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    image = image * std + mean
    image = image[[2, 1, 0], :, :]  # RGB → BGR
    return np.clip(image, 0, 1)


def visualize_main_images(dataloader, n_samples=4, save_path="sample_preview.png"):
    """Visualizes the first observation frame and saves to file."""
    sample_batch = next(iter(dataloader))
    sample_images = sample_batch["main-image"][:n_samples]  # (B, T, 3, H, W)
    
    B, T, C, H, W = sample_images.shape
    print(f"Saving visualization of {n_samples} samples to {save_path}...")

    plt.figure(figsize=(4 * n_samples, 4))
    for i in range(n_samples):
        img = sample_images[i, 0].cpu().numpy()
        img = unnormalize_imagenet(img)
        img = img.transpose(1, 2, 0)
        plt.subplot(1, n_samples, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Sample {i}")
    plt.tight_layout()
    plt.suptitle("First Timestep of main-image", y=1.05)
    plt.savefig(save_path)
    plt.close()
    print("Saved preview:", save_path)


def main():
    parser = argparse.ArgumentParser(description="Train Vision in Action Diffusion Policy (Small-scale)")
    parser.add_argument('--config', type=str, default="config/train_windows_small.yaml", help="Path to config YAML")
    parser.add_argument('--log-dir', type=str, default=None, help="Directory to save training logs, metrics CSV, and plots")
    parser.add_argument('--save-dir', type=str, default=None, help="Directory to save model checkpoints")
    args = parser.parse_args()

    # Load the YAML config
    with open(args.config, "r") as f:
        CONFIG = yaml.safe_load(f)

    # Setup directories
    model_save_dir = args.save_dir or CONFIG.get("model_save_dir", "via_model_ckps")
    os.makedirs(model_save_dir, exist_ok=True)

    training_log_dir = args.log_dir or CONFIG.get("training_log_dir", "training_logs")
    os.makedirs(training_log_dir, exist_ok=True)

    log_file_path = os.path.join(training_log_dir, "train.log")
    def log_print(msg):
        print(msg)
        with open(log_file_path, "a", encoding="utf-8") as lf:
            lf.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

    log_print(f"Checkpoints directory: {model_save_dir}")
    log_print(f"Training logs directory: {training_log_dir}")

    # Initialize CSV metrics log
    csv_metrics_path = os.path.join(training_log_dir, "training_metrics.csv")
    with open(csv_metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_action_mse", "learning_rate", "epoch_time_sec", "peak_vram_mb", "timestamp"])

    accelerator = Accelerator()
    device = accelerator.device

    # Enable TensorFloat32 (TF32) for fast matrix multiplication on Ampere+ GPUs (RTX 30-series, RTX 40-series)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        log_print("TF32 enabled: torch.set_float32_matmul_precision('high')")

    log_print(f"Running on device: {device}")

    # Load dataset
    dataset = ViAImageDataset(
        dataset_path=CONFIG["dataset_path"],
        pred_horizon=CONFIG["pred_horizon"],
        obs_horizon=CONFIG["obs_horizon"],
        action_horizon=CONFIG["action_horizon"]
    )
    
    num_workers = CONFIG.get("num_workers", 0)
    dataloader = DataLoader(
        dataset, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    stats = dataset.stats
    log_print(f"Length of training dataset: {len(dataset)}")

    # Save initial sample preview inside training_log_dir
    visualize_main_images(dataloader, save_path=os.path.join(training_log_dir, "dataset_preview.png"))

    # Load validation data
    log_print(f"Loading validation dataset: {CONFIG['val_dataset_path']}")
    loaded_dateset = load_from_zarr(CONFIG["val_dataset_path"])
    episode_ends = np.array(loaded_dateset["episode_ends"])
    val_data = {
        'agent_pos': np.array(loaded_dateset["proprio"][0:episode_ends[0]]).astype(np.float32),
        'action': np.array(loaded_dateset["action"][0:episode_ends[0]]).astype(np.float32),
        'main-image': np.array(loaded_dateset["main-image"][0:episode_ends[0]]).astype(np.float32) / 255.0,        
    }
    log_print(f"Validation samples loaded: {len(val_data['action'])}")

    # Load noise prediction network
    noise_pred_net = ConditionalUnet1D(
        input_dim=CONFIG["action_dim"],
        global_cond_dim=CONFIG["obs_dim"] * CONFIG["obs_horizon"]
    )

    # DINOv2 vision encoder
    log_print("Loading DINOv2 encoder...")
    head_dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    head_dinov2_vits14 = head_dinov2_vits14.to(device)

    nets = nn.ModuleDict({
        'vision_encoder': head_dinov2_vits14,
        'noise_pred_net': noise_pred_net
    })

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=CONFIG["num_train_DDIM_timesteps"],
        beta_start=.0001,
        beta_end=0.02,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type='epsilon'
    )

    ema = EMAModel(parameters=nets.parameters(), power=0.75)
    optimizer = torch.optim.AdamW(nets.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-6)

    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=min(500, len(dataloader) // 2),
        num_training_steps=len(dataloader) * CONFIG["num_epochs"]
    )

    ema, optimizer, dataloader, lr_scheduler = accelerator.prepare(ema, optimizer, dataloader, lr_scheduler)
    nets.to(device)
    ema.to(device)

    # Initialize wandb in offline mode so it does not block for credentials
    os.environ.setdefault("WANDB_MODE", "offline")
    wandb.init(
        project="via",
        dir=training_log_dir,
        mode=os.environ.get("WANDB_MODE", "offline"),
        config={
            "epochs": CONFIG["num_epochs"],
            "batch_size": CONFIG["batch_size"],
            "learning_rate": CONFIG["learning_rate"],
        }
    )

    metrics_history = []
    total_start_time = time.time()
    log_print(f"Starting training for {CONFIG['num_epochs']} epochs...")
    with tqdm(range(CONFIG["num_epochs"]), desc='Epoch') as tglobal:
        for epoch_idx in tglobal:
            epoch_start_time = time.time()
            avg_val_loss = None

            # Validation loop
            if epoch_idx % CONFIG["validate_every_epoch"] == 0:
                log_print(f"\nRunning validation at epoch {epoch_idx}...")
                ema_nets_unwrapped = accelerator.unwrap_model(nets)
                ema_nets_unwrapped.eval()
                val_loss = list()

                obs = dict()
                obs['main-image'] = val_data['main-image'][0]
                obs["agent_pos"] = val_data["agent_pos"][0]

                obs_deque = collections.deque(
                    [obs] * CONFIG["obs_horizon"], maxlen=CONFIG["obs_horizon"])

                step_idx = 0
                val_steps = int(len(val_data['main-image']) / 8) - 2
                for _ in range(val_steps):
                    B = 1
                    main_image = np.stack([x['main-image'] for x in obs_deque])
                    main_image = normalize_image_imagenet(main_image)
                    main_image = np.moveaxis(main_image, -1, 1)

                    nmain_image = torch.from_numpy(main_image).to(device, dtype=torch.float32)

                    agent_poses = np.stack([x['agent_pos'] for x in obs_deque])
                    nagent_poses = normalize_data(agent_poses, stats=stats['agent_pos'])
                    nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)

                    with torch.no_grad():
                        embeddings = ema_nets_unwrapped['vision_encoder'].forward_features(nmain_image)
                        main_image_features = embeddings["x_norm_clstoken"]
                        obs_features = torch.cat([main_image_features, nagent_poses], dim=-1)
                        obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                        noisy_action = torch.randn(
                            (B, CONFIG["pred_horizon"], CONFIG["action_dim"]), device=device)
                        naction = noisy_action

                        noise_scheduler.set_timesteps(CONFIG["num_test_DDIM_timesteps"])
                        for k in noise_scheduler.timesteps:
                            noise_pred = ema_nets_unwrapped['noise_pred_net'](
                                sample=naction,
                                timestep=k,
                                global_cond=obs_cond
                            )
                            naction = noise_scheduler.step(
                                model_output=noise_pred,
                                timestep=k,
                                sample=naction
                            ).prev_sample

                        naction = naction.detach().to('cpu').numpy()[0]
                        action_pred = unnormalize_data(naction, stats=stats['action'])

                        start = CONFIG["obs_horizon"] - 1
                        end = start + CONFIG["action_horizon"]
                        action = action_pred[start:end, :]

                        for i in range(len(action)):
                            gt_action = val_data['action'][step_idx]
                            cur_action = action[i]
                            val_loss.append(nn.functional.mse_loss(torch.tensor(gt_action), torch.tensor(cur_action)).item())
                            step_idx += 1
                            obs = dict()
                            obs["main-image"] = val_data['main-image'][step_idx]
                            obs["agent_pos"] = val_data["agent_pos"][step_idx]
                            obs_deque.append(obs)

                avg_val_loss = float(np.mean(val_loss))
                log_print(f"Epoch {epoch_idx} Validation Action MSE: {avg_val_loss:.6f}")
                wandb.log({"val_epoch_action_error": avg_val_loss, "epoch": epoch_idx})

                # Save EMA checkpoint into model_save_dir
                ckpt_path = os.path.join(model_save_dir, f"{CONFIG['model_name']}_ema_{epoch_idx}.pth")
                torch.save(ema_nets_unwrapped.state_dict(), ckpt_path)
                log_print(f"Checkpoint saved: {ckpt_path}")

            # Training batch loop
            epoch_loss = list()
            nets.train()
            with tqdm(dataloader, desc=f'Epoch {epoch_idx} Batches', leave=False) as tepoch:
                for nbatch in tepoch:
                    n_main_image = nbatch['main-image'][:, :CONFIG["obs_horizon"]].to(device)
                    nagent_pos = nbatch['agent_pos'][:, :CONFIG["obs_horizon"]].to(device)
                    naction = nbatch['action'].to(device)
                    B = nagent_pos.shape[0]

                    n_main_image = n_main_image.flatten(end_dim=1)

                    embeddings = nets['vision_encoder'].forward_features(n_main_image)
                    main_image_features = embeddings["x_norm_clstoken"]
                    nagent_pos = nagent_pos.flatten(end_dim=1)

                    obs_features = torch.cat([main_image_features, nagent_pos], dim=-1)
                    obs_cond = obs_features

                    noise = torch.randn(naction.shape, device=device)
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)
                    noise_pred = noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)

                    loss = nn.functional.mse_loss(noise_pred, noise)

                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

                    ema.step(nets.parameters())

                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=f"{loss_cpu:.4f}")
                    wandb.log({"train_batch_loss": loss_cpu, "epoch": epoch_idx})

            avg_epoch_loss = float(np.mean(epoch_loss))
            epoch_duration = time.time() - epoch_start_time
            current_lr = optimizer.param_groups[0]['lr']
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

            # Record per-epoch metrics to CSV
            with open(csv_metrics_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch_idx,
                    f"{avg_epoch_loss:.6f}",
                    f"{avg_val_loss:.6f}" if avg_val_loss is not None else "",
                    f"{current_lr:.6e}",
                    f"{epoch_duration:.2f}",
                    f"{peak_vram:.2f}",
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ])

            metrics_history.append({
                "epoch": epoch_idx,
                "train_loss": avg_epoch_loss,
                "val_loss": avg_val_loss,
                "lr": current_lr,
                "epoch_time_sec": epoch_duration,
                "peak_vram_mb": peak_vram
            })

            tglobal.set_postfix(loss=f"{avg_epoch_loss:.4f}")
            wandb.log({"train_epoch_loss": avg_epoch_loss, "epoch": epoch_idx})
            log_print(f"Epoch {epoch_idx} Complete. Average Training Loss: {avg_epoch_loss:.6f} (Time: {epoch_duration:.1f}s, Peak VRAM: {peak_vram:.1f}MB)")

    # Final validation and final model save
    log_print("\nTraining completed! Saving final model checkpoint...")
    final_ckpt_path = os.path.join(model_save_dir, f"{CONFIG['model_name']}_ema_final.pth")
    ema_nets_unwrapped = accelerator.unwrap_model(nets)
    torch.save(ema_nets_unwrapped.state_dict(), final_ckpt_path)
    log_print(f"Final checkpoint saved: {final_ckpt_path}")

    # Save summary JSON
    total_duration = time.time() - total_start_time
    summary_data = {
        "model_name": CONFIG.get("model_name", "via_dinov2_cls_cup"),
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {
            "num_epochs": CONFIG["num_epochs"],
            "batch_size": CONFIG["batch_size"],
            "learning_rate": CONFIG["learning_rate"],
            "pred_horizon": CONFIG["pred_horizon"],
            "obs_horizon": CONFIG["obs_horizon"],
            "action_horizon": CONFIG["action_horizon"],
        },
        "total_training_time_sec": round(total_duration, 2),
        "final_train_loss": avg_epoch_loss if 'avg_epoch_loss' in locals() else None,
        "best_val_loss": min([m["val_loss"] for m in metrics_history if m["val_loss"] is not None], default=None),
        "model_save_dir": model_save_dir,
        "training_log_dir": training_log_dir,
        "epochs_history": metrics_history
    }
    summary_json_path = os.path.join(training_log_dir, "training_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)
    log_print(f"Training summary saved to: {summary_json_path}")

    # Plot loss curves
    if metrics_history:
        epochs = [m["epoch"] for m in metrics_history]
        train_losses = [m["train_loss"] for m in metrics_history]
        val_records = [(m["epoch"], m["val_loss"]) for m in metrics_history if m["val_loss"] is not None]

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
        plot_path = os.path.join(training_log_dir, "loss_curves.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        log_print(f"Loss curves saved to: {plot_path}")

    log_print(f"All outputs organized:\n  - Checkpoints: {model_save_dir}\n  - Logs & Metrics: {training_log_dir}")



if __name__ == "__main__":
    main()
