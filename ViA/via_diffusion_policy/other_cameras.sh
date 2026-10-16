#!/bin/bash

# Load mamba into the shell
eval "$(mamba shell hook --shell bash)"

# Activate the via environment
mamba activate via

# Set ROS environment variables
export ROS_MASTER_URI=http://localhost:11311
export ROS_HOSTNAME=127.0.0.1
export ROS_IP=127.0.0.1

cd ../src

# Wait for 1 second
sleep 1

# Open a new tab in the existing terminal and run the second Python script
gnome-terminal --tab -- bash -c "export ROS_MASTER_URI=http://localhost:11311; export ROS_HOSTNAME=127.0.0.1; export ROS_IP=127.0.0.1; eval \"\$(mamba shell hook --shell bash)\"; mamba activate via; cd ../src; python read_right_wrist_cam.py; exec bash"

gnome-terminal --tab -- bash -c "export ROS_MASTER_URI=http://localhost:11311; export ROS_HOSTNAME=127.0.0.1; export ROS_IP=127.0.0.1; eeval \"\$(mamba shell hook --shell bash)\"; mamba activate via; cd ../src; python read_left_wrist_cam.py; exec bash"

gnome-terminal --tab -- bash -c "export ROS_MASTER_URI=http://localhost:11311; export ROS_HOSTNAME=127.0.0.1; export ROS_IP=127.0.0.1; eval \"\$(mamba shell hook --shell bash)\"; mamba activate via; cd ../src; python read_chest_cam.py; exec bash"

