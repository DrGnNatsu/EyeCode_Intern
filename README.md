# Robotics & AI Internship Project

This repository contains the complete implementation and technical documentation for the two-part Robotics & AI Internship assignment:

1. **Part 1 — ROSOrin Pro Simulation (ROS 2):** Gazebo-based mobile robot simulation on ROS 2 Humble, sensor bridging, LiDAR visualization, SLAM mapping with `slam_toolbox`, and autonomous waypoint navigation with `Nav2`.
2. **Part 2 — Research Paper Reading & Implementation (Active Perception):** Deep-dive comparative analysis between **ViA (CoRL 2025)** and **ActiveVLA (CVPR 2026)**, along with a complete, reproducible implementation of the **Vision in Action (ViA) Diffusion Policy** for 23-DOF bi-manual manipulation with active perception.

---

## Table of Contents

- [Part 1 — ROSOrin Pro Simulation (`ROSOrin_Simulation/`)](#part-1--rosorin-pro-simulation)
- [Part 2 — Research Paper Reading & Implementation (`ViA/`)](#part-2--research-paper-reading--implementation)
- [Repository Structure](#repository-structure)
- [Final Submission Deliverables](#final-submission-deliverables)

---

## Part 1 — ROSOrin Pro Simulation

**Directory:** [`ROSOrin_Simulation/`](ROSOrin_Simulation/)  
**User Guide & Documentation:** [`ROSOrin_Simulation/README.md`](ROSOrin_Simulation/README.md)

Part 1 implements the complete **ROS 2** simulation pipeline for the **ROSOrin Pro** mobile robot in Gazebo, from building workspace packages to autonomous waypoint patrol without requiring physical robot hardware.

> [!NOTE]  
> **Virtual Machine Credentials:** Username: `ubuntu` (or `root`) | Password: `ubuntu`

### Key Covered Milestones & Features

- **ROS 2 Environment & Build:** Setup in `~/ros2_ws` using `colcon build --symlink-install` for `robot_gazebo` and `rosorin_description`.
- **Gazebo Simulation & Spawning:** Launching simulated room environments (`room_worlds.launch.py`) and spawning the ROSOrin Pro model (`spawn_model.launch.py`).
- **Gazebo–ROS 2 Topic Bridge:** Relaying `/scan`, `/odom`, and `/cmd_vel` topics into ROS 2 via `ros_ign_bridge.launch.py`.
- **LiDAR Verification & RViz 2:** Diagnosing 10–30 Hz `/scan` publish frequency, setting up `RobotModel` (`/robot_description`), and configuring `LaserScan` with `Best Effort` reliability in RViz 2.
- **SLAM Mapping (`slam_toolbox`):** Constructing real-time occupancy grid maps with simulation time (`use_sim_time:=True`), teleoperating with remapped `/controller/cmd_vel`, and exporting maps with `nav2_map_server map_saver_cli` (`my_sim_map.yaml`, `my_sim_map.pgm`).
- **Autonomous Navigation (`Nav2`):**
  - Bridging Nav2 velocity commands to the Gazebo controller via `topic_tools relay /cmd_vel /controller/cmd_vel`.
  - Launching the full Nav2 bringup stack with the generated map.
  - Setting AMCL initial localization (`2D Pose Estimate`) aligning LiDAR scans with map boundaries.
  - Executing single-goal path planning and multi-point waypoint patrol entirely within RViz 2.
- **Official Documentation:** [Hiwonder ROSOrin Pro Wiki — Section 9: Gazebo Simulation](https://wiki.hiwonder.com/projects/rosorin-pro/en/latest/docs/9_Gazebo_Simulation.html#anther9.3.1)

👉 **For the complete terminal-by-terminal setup instructions, launch commands, configuration files, and demo links, please refer to [`ROSOrin_Simulation/README.md`](ROSOrin_Simulation/README.md).**

---

## Part 2 — Research Paper Reading & Implementation

**Directory:** [`ViA/`](ViA/)  
**User Guide & Training Manual:** [`ViA/README.md`](ViA/README.md)

Part 2 explores the topic of **Active Perception** in robotics through academic literature analysis and empirical reproduction of diffusion-based manipulation policies.

### Literature Covered

- **Paper 1:** [Vision in Action: Learning Active Perception from Human Demonstrations](https://proceedings.mlr.press/v305/xiong25a.html) (CoRL 2025)
- **Paper 2:** [ActiveVLA: Injecting Active Perception into Vision-Language-Action Models for Precise 3D Robotic Manipulation](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_ActiveVLA_Injecting_Active_Perception_into_Vision-Language-Action_Models_for_Precise_3D_CVPR_2026_paper.html) (CVPR 2026)

### Implemented Model: Vision in Action (ViA) Diffusion Policy

Under [`ViA/`](ViA/), we provide a complete, standalone reproduction of the ViA policy:

- **Architecture:** Pretrained **DINOv2 ViT-S/14** visual backbone combined with a **1D Conditional U-Net Diffusion Policy** to predict 16-step trajectories across 23 continuous degrees of freedom (7D active perception neck + 8D right arm + 8D left arm).
- **Environment Setup:** Automated setup scripts for both Windows (`setup_windows.ps1`) and Linux (`setup_ubuntu.sh`) with CUDA acceleration.
- **Model Training:** Training pipelines (`train_windows_small.py`, `train_ubuntu_small.py`) with TensorFloat-32 (TF32) support and separated logging.
- **Benchmarking & Evaluation:** Quantitative evaluation script (`benchmark_via.py`) measuring policy MSE, MAE, jerk smoothness, latency, and VRAM usage.
- **Inference & Visualization:** Single-sample inference tool (`infer_single_sample.py`) generating 6-panel diagnostic dashboards and animated 16-step MP4 video clips.

👉 **For detailed instructions on environment setup, dataset downloads, training commands, evaluation, and troubleshooting, please refer to [`ViA/README.md`](ViA/README.md).**  

Additional analytical documentation:

- [`ViA/Statistics.md`](ViA/Statistics.md) — Standardized academic metric definitions and benchmark tables.
- [`ViA/RESULTS.md`](ViA/RESULTS.md) — Detailed empirical evaluation, convergence curves, and failure analysis.

---

## Repository Structure

```text
EyeCode_Intern/
├── README.md                     # Root overview and navigation guide
├── ROSOrin_Simulation/           # Part 1: ROSOrin Pro simulation module (ROS 2)
│   └── README.md                 # Complete ROS 2 guide (Gazebo, LiDAR, SLAM, Nav2)
├── ViA/                          # Part 2: Vision in Action (ViA) reproduction
│   ├── README.md                 # Part 2 training guide & user manual
│   ├── Statistics.md             # Benchmark metric definitions & statistics
│   ├── RESULTS.md                # Empirical results & error analysis
│   └── via_diffusion_policy/     # Python code, configs, dataset & scripts
└── research_paper_prompts/       # Prompt pack for the 5–7 page technical report
```

---

## Final Submission Deliverables

- **GitHub Repository / Source Code:** Contains both `ROSOrin_Simulation/` and `ViA/via_diffusion_policy/`.
- **Reproduction Documentation:** Detailed instructions in [`ROSOrin_Simulation/README.md`](ROSOrin_Simulation/README.md) and [`ViA/README.md`](ViA/README.md).
- **Part 1 Demo Video:** 3–5 minute video demonstrating Gazebo simulation, LiDAR visualization, SLAM, and Nav2 navigation.
- **Part 2 Demo Video & Visuals:** 16-step rollout MP4 and 6-panel diagnostic dashboards in `ViA/via_diffusion_policy/inference_output/`.
- **Technical Report:** 5–7 page academic report covering both parts.
