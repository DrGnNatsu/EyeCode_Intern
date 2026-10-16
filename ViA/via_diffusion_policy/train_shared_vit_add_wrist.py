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
import torch
import torch.nn as nn
import collections
import numpy as np
import torch
from torch.utils.data import Dataset
import zarr
from utils import normalize_data, unnormalize_data, normalize_image_imagenet, create_sample_indices, sample_sequence, get_data_stats




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

        loaded_dateset = load_from_zarr(dataset_path)
        print("loaded!")
        train_main_image_data = np.array(loaded_dateset["main-image"]).astype(np.float32)
        train_right_image_data = np.array(loaded_dateset["right-image"]).astype(np.float32)
        train_left_image_data = np.array(loaded_dateset["left-image"]).astype(np.float32)
        # train_pointcloud_data = np.array(loaded_dateset["pointcloud"]).astype(np.float32)

        train_main_image_data /= 255.0
        train_right_image_data /= 255.0
        train_left_image_data  /= 255.0



        # Move from (N, H, W, 3) → (N, 3, H, W)
        train_main_image_data = np.moveaxis(train_main_image_data, -1, 1)
        train_right_image_data = np.moveaxis(train_right_image_data, -1, 1)
        train_left_image_data  = np.moveaxis(train_left_image_data, -1, 1)

        train_main_image_data = normalize_image_imagenet(train_main_image_data)
        train_right_image_data = normalize_image_imagenet(train_right_image_data)
        train_left_image_data  = normalize_image_imagenet(train_left_image_data)

        # train_pointcloud_data = np.array(loaded_dateset["pointcloud"]).astype(np.float32)
        # print("train_pointcloud_data", train_pointcloud_data.shape)




        # (N, D)
        train_data = {
            # first two dims of state vector are agent (i.e. gripper) locations
            'agent_pos': np.array(loaded_dateset["proprio"]).astype(np.float32),
            'action': np.array(loaded_dateset["action"]).astype(np.float32)
        }
        episode_ends = np.array(loaded_dateset["episode_ends"])

        print("demo numbers", len(episode_ends))

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            # print(key)
            stats[key] = get_data_stats(data)
            # print(stats[key])

            # normalized_train_data[key] = data  # normalize_data(data, stats[key])
            normalized_train_data[key] = normalize_data(data, stats[key])

        # images are already normalized
        normalized_train_data['main-image'] = train_main_image_data
        normalized_train_data['right-image'] = train_right_image_data
        normalized_train_data['left-image'] = train_left_image_data

        # normalized_train_data['pointcloud'] = train_pointcloud_data
        print("main-image shape:", normalized_train_data['main-image'].shape)
        print("right-image shape:", normalized_train_data['right-image'].shape)
        print("left-image shape:", normalized_train_data['left-image'].shape)

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        # print(len(self.indices))
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        nsample['main-image'] = nsample['main-image'][:self.obs_horizon,:]
        nsample['right-image'] = nsample['right-image'][:self.obs_horizon,:]
        nsample['left-image'] = nsample['left-image'][:self.obs_horizon,:]
        # nsample['pointcloud'] = nsample['pointcloud'][:self.obs_horizon,:]

        nsample['agent_pos'] = nsample['agent_pos'][:self.obs_horizon,:]
        return nsample


accelerator = Accelerator()
device = accelerator.device
# Load config from YAML
# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
args = parser.parse_args()

# Load the YAML config
with open(args.config, "r") as f:
    CONFIG = yaml.safe_load(f)


# Load dataset
dataset = ViAImageDataset(
    dataset_path=CONFIG["dataset_path"], pred_horizon=CONFIG["pred_horizon"], obs_horizon=CONFIG["obs_horizon"], action_horizon=CONFIG["action_horizon"])
dataloader = DataLoader(
    dataset, batch_size=CONFIG["batch_size"], shuffle=True,
    num_workers=4, pin_memory=True, persistent_workers=True)
stats = dataset.stats
print("length of taining dataset:", len(dataset))


# Load the dictionary from the pickle file
loaded_dateset = load_from_zarr(CONFIG["val_dataset_path"])
episode_ends = np.array(loaded_dateset["episode_ends"])
val_data = {
    # first two dims of state vector are agent (i.e. gripper) locations
    'agent_pos': np.array(loaded_dateset["proprio"][0:episode_ends[0]]).astype(np.float32),
    'action': np.array(loaded_dateset["action"][0:episode_ends[0]]).astype(np.float32),
    'main-image': np.array(loaded_dateset["main-image"][0:episode_ends[0]]).astype(np.float32),        
    'right-image': np.array(loaded_dateset["right-image"][0:episode_ends[0]]).astype(np.float32),        
    'left-image': np.array(loaded_dateset["left-image"][0:episode_ends[0]]).astype(np.float32),        
}

print("main-image shape:", val_data['main-image'].shape)
print("right-image shape:", val_data['right-image'].shape)
print("left-image shape:", val_data['left-image'].shape)

# Load resnet encoder
# resnet_encoder = get_resnet('resnet18', weights="IMAGENET1K_V1")
# resnet_encoder = replace_bn_with_gn(main_encoder)

# Load noise prediction network
noise_pred_net = ConditionalUnet1D(
    input_dim=CONFIG["action_dim"],
    global_cond_dim=CONFIG["obs_dim"] * CONFIG["obs_horizon"]
)

# DINOv2 encoder (placeholder)
head_dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
head_dinov2_vits14 = head_dinov2_vits14.to(device)

vision_encoder_right = get_resnet(
    name='resnet18',
    weights="IMAGENET1K_V1"
)
vision_encoder_right = replace_bn_with_gn(vision_encoder_right)

vision_encoder_left = get_resnet(
    name='resnet18',
    weights="IMAGENET1K_V1"
)
vision_encoder_left = replace_bn_with_gn(vision_encoder_left)

# the final arch has 2 parts
nets = nn.ModuleDict({
    'vision_encoder': head_dinov2_vits14,
    'vision_encoder_right': vision_encoder_right,
    'vision_encoder_left': vision_encoder_left,


    'noise_pred_net': noise_pred_net
})

noise_scheduler = DDIMScheduler(
    num_train_timesteps=CONFIG["num_train_DDIM_timesteps"],
    beta_start = .0001,
    beta_end=0.02,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    set_alpha_to_one=True,
    steps_offset=0,
    prediction_type='epsilon'
)


ema = EMAModel(parameters=nets.parameters(), power=0.75)
optimizer = torch.optim.AdamW(nets.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-6)
# Cosine LR schedule with linear warmup
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * CONFIG["num_epochs"]
)

ema, optimizer, dataloader, lr_scheduler = accelerator.prepare(ema, optimizer, dataloader, lr_scheduler)
nets.to(device)
ema.to(device)

# Initialize wandb project
wandb.init(project="via", config={
    "epochs": CONFIG["num_epochs"],
    "batch_size": CONFIG["batch_size"],
    "learning_rate": CONFIG["learning_rate"],
})




with tqdm(range(CONFIG["num_epochs"]), desc='Epoch') as tglobal:
    # epoch loop
    for epoch_idx in tglobal:
        # Validation loop
        if (epoch_idx) % CONFIG["validate_every_epoch"] ==0:
            # Unwrap the model from accelerator
            ema_nets_unwrapped = accelerator.unwrap_model(nets)
            ema_nets_unwrapped.eval()  # Ensure the model is in evaluation mode
            val_loss = list()
            # get first observation
            obs = dict()
            obs['main-image'] = val_data['main-image'][0]
            obs['right-image'] = val_data['right-image'][0]
            obs['left-image'] = val_data['left-image'][0]
            obs["agent_pos"] = val_data["agent_pos"][0]
            # keep a queue of last 2 steps of observations
            obs_deque = collections.deque(
                [obs] * CONFIG["obs_horizon"], maxlen=CONFIG["obs_horizon"])

            step_idx = 0
            for _ in range(int(len(val_data['main-image'])/8) -2 ):
                B = 1
                main_image = np.stack([x['main-image'] for x in obs_deque])
                right_image = np.stack([x['right-image'] for x in obs_deque])
                left_image = np.stack([x['left-image'] for x in obs_deque])

                main_image = normalize_image_imagenet(main_image)
                main_image = np.moveaxis(main_image, -1,1)
                right_image = normalize_image_imagenet(right_image)
                right_image = np.moveaxis(right_image, -1,1)
                left_image = normalize_image_imagenet(left_image)
                left_image = np.moveaxis(left_image, -1,1)


                nmain_image = torch.from_numpy(main_image).to(device, dtype=torch.float32)
                nright_images = torch.from_numpy(right_image).to(device, dtype=torch.float32)
                nleft_images = torch.from_numpy(left_image).to(device, dtype=torch.float32)

                agent_poses = np.stack([x['agent_pos'] for x in obs_deque])
                nagent_poses = normalize_data(agent_poses, stats=stats['agent_pos'])
                nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
        



                # infer action
                with torch.no_grad():
                    # get image features
                    embeddings = ema_nets_unwrapped['vision_encoder'].forward_features(nmain_image)
                    main_image_features = embeddings["x_norm_clstoken"] # [1, 256, 384])

                    right_image_features = ema_nets_unwrapped['vision_encoder_right'](nright_images)
                    left_image_features = ema_nets_unwrapped['vision_encoder_left'](nleft_images)

                    obs_features = torch.cat([main_image_features, right_image_features, left_image_features, nagent_poses], dim=-1)


                    # concat with low-dim observations
                    # obs_features = torch.cat([main_image_features, nagent_poses], dim=-1)
                    # reshape observation to (B,CONFIG["obs_horizon"]*obs_dim)
                    obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)
                    # initialize action from Guassian noise
                    noisy_action = torch.randn(
                        (B, CONFIG["pred_horizon"], CONFIG["action_dim"]), device=device)
                    naction = noisy_action

                    # init scheduler
                    noise_scheduler.set_timesteps(CONFIG["num_test_DDIM_timesteps"])
                    for k in noise_scheduler.timesteps:
                        # predict noise
                        noise_pred = ema_nets_unwrapped['noise_pred_net'](
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond
                        )
                        # inverse diffusion step (remove noise)
                        naction = noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        ).prev_sample


                    # unnormalize action
                    naction = naction.detach().to('cpu').numpy()
                    # (B, pred_horizon, action_dim)
                    naction = naction[0]
                    action_pred = unnormalize_data(naction, stats=stats['action'])

                    # only take action_horizon number of actions
                    start = CONFIG["obs_horizon"] - 1
                    end = start + CONFIG["action_horizon"]
                    action = action_pred[start:end,:]

                    for i in range(len(action)):
                        print(step_idx)                       
                        # compute errors
                        gt_action = val_data['action'][step_idx]
                        naction =  action[i]
                        val_loss.append(nn.functional.mse_loss(torch.tensor(gt_action), torch.tensor(naction)))
                        step_idx += 1
                        obs = dict()
                        obs["main-image"] = val_data['main-image'][step_idx]
                        obs["right-image"] = val_data['right-image'][step_idx]
                        obs["left-image"] = val_data['left-image'][step_idx]
                        obs["agent_pos"] = val_data["agent_pos"][step_idx]
                        obs_deque.append(obs)

                        
            

            avg_val_loss = np.mean(val_loss)
            wandb.log({"val_epoch_action_error": avg_val_loss, "epoch": epoch_idx})
            # Directory where to save the models
            os.makedirs(CONFIG["model_save_dir"], exist_ok=True)
            # Save the EMA model's state dict
            torch.save(ema_nets_unwrapped.state_dict(), os.path.join(CONFIG["model_save_dir"], CONFIG["model_name"]+f"_ema_{epoch_idx}.pth"))

            print("EMA model saved!")



        epoch_loss = list()
        # batch loop
        with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
            for nbatch in tepoch:
                # data normalized in dataset
                # device transfer
                n_main_image = nbatch['main-image'][:,:CONFIG["obs_horizon"]].to(device)
                n_right_image = nbatch['right-image'][:,:CONFIG["obs_horizon"]].to(device)
                n_left_image = nbatch['left-image'][:,:CONFIG["obs_horizon"]].to(device)

                nagent_pos = nbatch['agent_pos'][:,:CONFIG["obs_horizon"]].to(device)
                naction = nbatch['action'].to(device)
                B = nagent_pos.shape[0]


                n_main_image = n_main_image.flatten(end_dim=1)
                n_right_image = n_right_image.flatten(end_dim=1)
                n_left_image = n_left_image.flatten(end_dim=1)

                # encoder rgb features
                embeddings = nets['vision_encoder'].forward_features(n_main_image)
                main_image_features = embeddings["x_norm_clstoken"]

                right_image_features = nets['vision_encoder_right'](n_right_image)
                left_image_features = nets['vision_encoder_left'](n_left_image)

                nagent_pos = nagent_pos.flatten(end_dim=1)
                # concat with low-dim observations

                obs_features = torch.cat([main_image_features, right_image_features, left_image_features, nagent_pos], dim=-1)


                obs_cond = obs_features#.unsqueeze(0).flatten(start_dim=1)


                # sample noise to add to actions
                noise = torch.randn(naction.shape, device=device)

                # sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (B,), device=device
                ).long()

                # add noise to the clean images according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)
                noisy_actions = noise_scheduler.add_noise(
                    naction, noise, timesteps)

                # predict the noise residual
                noise_pred = noise_pred_net(
                    noisy_actions, timesteps, global_cond=obs_cond)

                # L2 loss
                loss = nn.functional.mse_loss(noise_pred, noise)

                # optimize
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                # step lr scheduler every batch
                # this is different from standard pytorch behavior
                lr_scheduler.step()

                # update Exponential Moving Average of the model weights
                ema.step(nets.parameters())

                # logging
                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)

                wandb.log({"train_batch_loss": loss_cpu, "epoch": epoch_idx})



        avg_epoch_loss = np.mean(epoch_loss)
        tglobal.set_postfix(loss=avg_epoch_loss)

        # Log average loss per epoch to wandb
        wandb.log({"train_epoch_loss": avg_epoch_loss, "epoch": epoch_idx})



     