# Robotics & AI Internship Project

This repository contains the complete implementation and technical documentation for the two-part Robotics & AI Internship assignment:

1. **Part 1 — ROSOrin Pro Simulation:** Gazebo-based mobile robot simulation, LiDAR sensor integration, SLAM mapping, and autonomous navigation.
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

Part 1 focuses on simulating the **ROSOrin Pro** mobile robot in Gazebo, integrating sensors, performing SLAM, and executing autonomous navigation without requiring physical hardware access.

### Key Objectives & Covered Tasks
- **Gazebo Simulation Setup:** Launching and controlling the simulated robot in a virtual environment.
- **Sensor Integration:** Visualizing simulated LiDAR, RGB-D camera, IMU, and odometry in RViz.
- **SLAM & Mapping:** Generating, inspecting, and saving occupancy grid maps.
- **Autonomous Navigation:** Path planning, obstacle avoidance, and executing single-goal as well as multi-point waypoint navigation.
- **Official Documentation:** [Hiwonder ROSOrin Pro Wiki](https://wiki.hiwonder.com/projects/rosorin-pro/en/latest/)

👉 **For the complete setup instructions, launch commands, configuration files, and demo links, please refer to [`ROSOrin_Simulation/README.md`](ROSOrin_Simulation/README.md).**

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
├── ROSOrin_Simulation/           # Part 1: ROSOrin Pro simulation module
│   └── README.md                 # Part 1 instructions, configs, and demo links
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
- **Part 1 Demo Video:** 3–5 minute video demonstrating simulation, SLAM, and navigation.
- **Part 2 Demo Video & Visuals:** 16-step rollout MP4 and 6-panel diagnostic dashboards in `ViA/via_diffusion_policy/inference_output/`.
- **Technical Report:** 5–7 page academic report covering both parts.
