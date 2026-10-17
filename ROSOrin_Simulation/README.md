# Part 1: ROSOrin Pro Simulation Guide (ROS 2)

A concise, step-by-step guide to launch the **ROSOrin Pro** simulation, bridge sensor topics, visualize LiDAR in RViz 2, and run SLAM/Nav2.

> [!NOTE]  
> **Official Documentation:** [Hiwonder ROSOrin Pro — Gazebo Simulation](https://wiki.hiwonder.com/projects/rosorin-pro/en/latest/docs/9_Gazebo_Simulation.html#anther9.3.1)  
> **Environment:** ROS 2 Humble (`colcon`, `ros2 launch`, `~/ros2_ws`)  
> **Virtual Machine Credentials:**  
> - **Username:** `ubuntu` (or `root`)  
> - **Password:** `ubuntu`

---

## 1. Build & Setup Workspace

```bash
cd ~/ros2_ws
colcon build --symlink-install --packages-select robot_gazebo rosorin_description
source install/setup.bash
```

---

## 2. Launch Simulation & Robot

### Step 1: Launch Gazebo Simulation

```bash
ros2 launch robot_gazebo room_worlds.launch.py
```

> If the robot does not appear automatically, open a new terminal and run:

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch robot_gazebo spawn_model.launch.py
```

### Step 2: Enable Gazebo–ROS 2 Topic Bridge

In a new terminal:

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch robot_gazebo ros_ign_bridge.launch.py
```

*Bridges `/scan`, `/odom`, and `/cmd_vel` from Gazebo to ROS 2.*

---

## 3. Verify & Visualize LiDAR in RViz 2

### Step 3: Verify LiDAR Topic Signal

In a new terminal:

```bash
# Check if /scan topic exists
ros2 topic list | grep scan

# Check publish frequency (should be ~10 Hz – 30 Hz)
ros2 topic hz /scan
```

### Step 4: Launch & Configure RViz 2

```bash
source ~/ros2_ws/install/setup.bash
rviz2
```

Configure RViz 2 settings:

1. **Global Options -> Fixed Frame:** Set to `odom` (or `base_footprint`).
2. **Add RobotModel:** Click **Add** (bottom-left) $\to$ **RobotModel** $\to$ set **Description Topic** to `/robot_description`.
3. **Add LaserScan:** Click **Add** $\to$ **LaserScan** $\to$ set **Topic** to `/scan`.
   - Set **Size (m)** to `0.05`.
   - Set **Reliability Policy** to `Best Effort` (required if points are not rendering).

### Step 5: Visual & Motion Verification

Open keyboard teleop in a new terminal:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap /cmd_vel:=/controller/cmd_vel
```

*Drive the robot (`i`, `j`, `l`, `,`). As the robot turns and moves, LiDAR scan points in RViz 2 dynamically update against wall contours and obstacles.*

---

## 4. SLAM Mapping (`slam_toolbox`)

### 4.1 Install SLAM & Map Tools

```bash
sudo apt-get update
sudo apt-get install -y ros-humble-slam-toolbox ros-humble-nav2-map-server
```

### 4.2 Keep Simulation & Bridge Running

Ensure the Gazebo world and bridge node are active in separate terminals:

- **Gazebo World:** `ros2 launch robot_gazebo room_worlds.launch.py`
- **Topic Bridge:** `source ~/ros2_ws/install/setup.bash && ros2 launch robot_gazebo ros_ign_bridge.launch.py`

### 4.3 Launch SLAM Toolbox

Open a new terminal to start the SLAM node (generates `/map` topic):

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch slam_toolbox online_async_launch.py use_sim_time:=True
```

*Verification:* Run `ros2 topic list | grep map` to confirm `/map` is active.

### 4.4 Visualize Real-Time Map in RViz 2

In a new terminal:

```bash
source ~/ros2_ws/install/setup.bash
rviz2
```

Configure displays in RViz 2:

- **Fixed Frame:** Set to `map`.
- **Add Map:** Click **Add** $\to$ **Map** $\to$ set **Topic** to `/map`, and set **Durability Policy** to `Transient Local`.
- **Add LaserScan (optional):** Click **Add** $\to$ **LaserScan** $\to$ set **Topic** to `/scan` with **Size (m)** `0.05`.

### 4.5 Drive Robot to Map the Arena

Open another terminal to teleoperate:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap /cmd_vel:=/controller/cmd_vel
```

*Drive slowly (`i`, `j`, `k`, `l`, `,`) around the room until all walls and obstacles are fully drawn in RViz 2.*

### 4.6 Save the Completed Map

Once the map is complete, save it to disk in a separate terminal:

```bash
mkdir -p ~/ros2_ws/src/maps
ros2 run nav2_map_server map_saver_cli -f ~/ros2_ws/src/maps/my_sim_map
```

*Verification:* Run `ls ~/ros2_ws/src/maps/` to confirm `my_sim_map.yaml` and `my_sim_map.pgm` were generated successfully.

---

## 5. Autonomous Navigation (Nav2)

A complete step-by-step procedure to run Nav2 autonomous navigation and multi-point waypoint patrol without node/topic missing errors.

### 5.1 Prerequisites & Topic Tools Relay

Nav2 publishes velocity to `/cmd_vel`, whereas Gazebo expects commands on `/controller/cmd_vel`. Install and run `topic_tools relay`:

```bash
sudo apt-get install -y ros-humble-topic-tools ros-humble-navigation2 ros-humble-nav2-bringup
```

### 5.2 Terminal-by-Terminal Execution

#### Terminal 1 — Gazebo Simulation

```bash
ros2 launch robot_gazebo room_worlds.launch.py
```

*(If robot is not present: `ros2 launch robot_gazebo spawn_model.launch.py`)*

#### Terminal 2 — Gazebo–ROS 2 Topic Bridge

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch robot_gazebo ros_ign_bridge.launch.py
```

*Verify `/scan` and `/odom` with `ros2 topic list | grep scan`.*

#### Terminal 3 — Velocity Command Relay

```bash
ros2 run topic_tools relay /cmd_vel /controller/cmd_vel
```

*Relays Nav2 velocity commands `/cmd_vel` directly to the Gazebo controller `/controller/cmd_vel`. Keep this terminal active.*

#### Terminal 4 — Launch Nav2 Stack

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch nav2_bringup bringup_launch.py \
  use_sim_time:=True \
  autostart:=True \
  map:=$HOME/ros2_ws/src/maps/my_sim_map.yaml
```

*Verification:* Wait ~10–15s until log prints:

```text
[lifecycle_manager_navigation]: Managed nodes are active
```

*(Or check in another terminal: `ros2 lifecycle get /bt_navigator` $\to$ `active [3]`)*.

#### Terminal 5 — Launch Pre-Configured Nav2 RViz 2

```bash
source ~/ros2_ws/install/setup.bash
rviz2 -d /opt/ros/humble/share/nav2_bringup/rviz/nav2_default_view.rviz
```

*Verification:* RViz 2 loads with Nav2 map, global/local costmaps, and the Navigation 2 panel status shows `active` or `idle`.

---

### 5.3 Initial Localization (2D Pose Estimate)

AMCL requires an initial pose estimate to align LiDAR readings with the saved map:

1. Observe the robot's physical location and heading inside **Gazebo**.
2. In **RViz 2**, click **2D Pose Estimate** on the top toolbar.
3. Click on the corresponding map position, hold and drag the green arrow to match the robot's heading, then release.
4. **Verification:**
   - A green particle cloud appears around the robot.
   - Red LiDAR `/scan` points align flush against black map walls.
   - Global and local costmaps render clearly without orange warning indicators.

---

### 5.4 Execute Navigation in RViz 2

#### A. Single-Goal Navigation

1. In the **RViz 2** window connected to Nav2, select the **Nav2 Goal** tool on the top toolbar.
2. Select your destination on the map:
   - **Click** on the desired $(x, y)$ target coordinates.
   - **Hold and drag** the arrow in the direction you want the robot to face upon arrival (Orientation $\theta$), then release.
3. **Verification:** A solid global path line appears connecting the robot to the goal pose. The robot in Gazebo starts driving along the path, dynamically dodging obstacles, and stops precisely at the target location with the desired heading.

#### B. Multi-Point / Waypoint Patrol Navigation

1. In the **Navigation 2** panel on the left side of RViz 2, switch mode from **Navigate** to **Waypoint mode**.
2. Select the **Nav2 Goal** tool on the top toolbar:
   - Click and drag to set **Waypoint 1**.
   - Click and drag to set **Waypoint 2**.
   - Continue adding waypoints (**Waypoint 3**, **Waypoint 4**, etc.). Each point is marked with an indexed numbered circle on the map.
3. Click the green **Start Waypoint Following** button in the left Navigation 2 panel.
4. **Verification:** The robot autonomously navigates sequentially through Waypoint 1 $\to$ Waypoint 2 $\to$ Waypoint 3... and completes the entire patrol cycle.

---
