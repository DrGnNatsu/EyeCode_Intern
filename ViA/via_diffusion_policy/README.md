# Policy Training

Download training data from [this link](https://drive.google.com/drive/folders/1qJbXjk8bEvlg3eARPnNTUiw5PcgMhmJy?usp=sharing), and move them to <code>~/Vision-in-Action/via_diffusion_policy/dataset</code>.

``` sh
cd via_diffusion_policy

python train_via.py --config config/train_via_bag.yaml 

``` 

# Policy Deployment

Download checkpoints from [this link](https://drive.google.com/drive/folders/1bMppBPUzdCGkUPAHtgmV9bW6bR0T5Rth?usp=sharing),:


```bash
cd via_diffusion_policy

# Step 1: Bring up CAN interface
chmod +x bringup_can.sh
./bringup_can.sh

# Step 2: Start ROS core (in a new terminal)
chmod +x roscore.sh
./roscore.sh

# Step 3: Launch robot controller (in a new terminal)
chmod +x robot_controller.sh
./robot_controller.sh

# Step 4: Start iPhone camera stream (in a new terminal)
chmod +x iphone_camera.sh
./iphone_camera.sh

# Step 5: Start other external cameras (in a new terminal)
chmod +x other_cameras.sh
./other_cameras.sh

# Step 6: Start Policy Rollout (in a new terminal)

chmod +x policy_rollout.sh 
./policy_rollout.sh 