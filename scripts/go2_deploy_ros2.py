#!/usr/bin/env python3
# Description: Modular Vision-based Parkour Policy Deployment

import os
import sys
import time
import numpy as np
import torch as th
import copy
import mujoco

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, Image
from dls2_msgs.msg import BaseStateMsg, BlindStateMsg, TrajectoryGeneratorMsg
from cv_bridge import CvBridge


from scripts.utils import load_local_cfg
from core.deployment_player import DeploymentPlayer
from mujoco_deploy.mujoco_wrapper import MujocoWrapper
import core

# Set priority for real-time safety
pid = os.getpid()
os.system(f"renice -n -21 -p {pid}")
os.system(f"echo -20 > /proc/{pid}/autogroup")


class RealGo2Env(MujocoWrapper):
    """
    Subclasses the simulation wrapper to inject real hardware states and
    bypass the physics engine, while keeping all observation/history logic intact.
    """
    def __init__(self, env_cfg, agent_cfg, model_xml_path, use_camera, ros_node):
        super().__init__(env_cfg, agent_cfg, model_xml_path, use_camera)
        self.ros_node = ros_node
        self.real_depth_tensor = th.zeros(
            self.sensor_cfg.pattern_cfg.height,
            self.sensor_cfg.pattern_cfg.width,
            dtype=th.float32,
            device=self.device
        )

    def sync_mujoco_state(self, qpos, qvel):
        """Forces the internal MuJoCo model to perfectly match the real robot."""
        self._mujoco_env.data.qpos[:] = qpos
        self._mujoco_env.data.qvel[:] = qvel
        mujoco.mj_forward(self._mujoco_env.model, self._mujoco_env.data)

    def step(self, actions: th.Tensor | None = None):
        """Overrides the simulation step to send commands to the real hardware instead."""
        self._actions = actions.clone()
        # Uses the inherited logic for delays, clipping, scaling, and offset
        self._process_actions()
        self.common_step_counter += 1
        self.episode_length_buf += 1

        # Publish the processed actions to the real robot
        self.ros_node.publish_actions(self._processed_actions[0].detach().cpu().numpy())

        # Bypass the physical `self._mujoco_env.step()`!

        # Grab updated observations (which will read the MuJoCo state we synced earlier)
        obs, extras = self.get_observations()
        termination = self._termination()
        time_out_buf = self.episode_length_buf >= 100000 # Keep alive
        return obs, termination, time_out_buf, extras

    def _get_depth_image(self, is_reset: bool = False):
        """Overrides the MuJoCo renderer to ingest the real ROS depth frame."""
        # Process using the inherited crop/resize/normalize logic
        processed_image = self._process_depth_image(self.real_depth_tensor)

        if is_reset:
            self.depth_buffer[0] = th.stack([processed_image]* 2, dim=0)

        if self.common_step_counter % 5 == 0:
            self.depth_buffer[0] = th.cat([self.depth_buffer[0, 1:],
                                    processed_image.to(self.device).unsqueeze(0)], dim=0)
        return self.depth_buffer[:, -2].to(self.device)


class RealDeploymentPlayer(DeploymentPlayer):
    """Subclass DeploymentPlayer to inject our RealGo2Env"""
    def __init__(self, env_cfg, agent_cfg, logs_path, ros_node):
        super().__init__(env_cfg, agent_cfg, 'lo', logs_path)
        # Overwrite the initialized `self.env` with our hardware-bound environment
        model_path = os.path.join(core.__path__[0], 'go2/scene_parkour.xml')
        self.env = RealGo2Env(env_cfg, agent_cfg, model_path, self._use_camera, ros_node)


class Parkour_Go2_Deployment_Node(Node):
    def __init__(self):
        super().__init__('Parkour_Go2_Deployment_Node')

        self.bridge = CvBridge()

        # Paths
        exp_id = "2025-09-03_12-07-56" # Update as needed
        logs_path = f"/home/edelia-iit.local/git/IsaacLab/Isaaclab_Parkour/logs/rsl_rl/unitree_go2_parkour/{exp_id}"
        cfgs_path = os.path.join(logs_path, 'params')

        self.get_logger().info("Loading Configurations...")
        env_cfg = load_local_cfg(cfgs_path, 'env')
        agent_cfg = load_local_cfg(cfgs_path, 'agent')
        env_cfg.scene.num_envs = 1

        # Initialize the modular player with our ROS Node
        self.player = RealDeploymentPlayer(env_cfg, agent_cfg, logs_path, self)

        # State tracking
        self.first_base_arrived = False
        self.first_joints_arrived = False

        self.qpos = np.zeros(19)
        self.qvel = np.zeros(18)

        # ROS2 Subscriptions
        self.sub_base = self.create_subscription(BaseStateMsg, "/dls2/base_state", self.base_state_callback, 1)
        self.sub_blind = self.create_subscription(BlindStateMsg, "/dls2/blind_state", self.blind_state_callback, 1)
        self.sub_joy = self.create_subscription(Joy, "joy", self.joy_callback, 1)
        self.sub_depth = self.create_subscription(Image, "/camera/depth/image_rect_raw", self.depth_callback, 1)

        # ROS2 Publishers
        self.pub_traj = self.create_publisher(TrajectoryGeneratorMsg, "dls2/trajectory_generator", 1)

        # Control Loop Timer (50 Hz) -> Match RL_FREQ
        self.timer = self.create_timer(1.0 / 50.0, self.compute_rl_control)

    def joy_callback(self, msg):
        # Feed directly into the Env's mock joystick
        self.player.env._joystick.velocity_cmd[0, 0] = msg.axes[1] / 3.5  # Forward/Backward
        self.player.env._joystick.velocity_cmd[0, 1] = msg.axes[0] / 3.5  # Left/Right
        self.player.env._joystick.velocity_cmd[0, 2] = msg.axes[3] / 2.0  # Yaw

        if msg.buttons[8] == 1:
            self.get_logger().info("Kill switch pressed. Shutting down.")
            os.system("kill -9 $(ps -u | grep -m 1 hal | grep -o '^[^ ]* *[0-9]*' | grep -o '[0-9]*')")
            os.system("pkill -f modular_parkour_ros2.py")
            exit(0)

    def base_state_callback(self, msg):
        self.qpos[0:3] = np.array(msg.position)
        # Quat: MuJoCo uses [w, x, y, z], DLS2 uses [x, y, z, w]
        self.qpos[3:7] = np.roll(np.array(msg.orientation), 1)
        self.qvel[0:3] = np.array(msg.linear_velocity)
        self.qvel[3:6] = np.array(msg.angular_velocity)
        self.first_base_arrived = True

    def blind_state_callback(self, msg):
        jp = np.array(msg.joints_position)
        jv = np.array(msg.joints_velocity)

        # Fix DLS2 Hip Signs
        jp[0] = -jp[0]; jp[6] = -jp[6]
        jv[0] = -jv[0]; jv[6] = -jv[6]

        self.qpos[7:19] = jp
        self.qvel[6:18] = jv
        self.first_joints_arrived = True

    def depth_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            depth_m = th.tensor(cv_image, dtype=th.float32, device=self.player.env.device) / 1000.0

            # Send raw depth tensor directly to the environment
            self.player.env.real_depth_tensor = depth_m
        except Exception as e:
            self.get_logger().warn(f"Depth conversion failed: {e}")

    def publish_actions(self, target_joints):
        """Called internally by RealGo2Env.step()"""
        # Re-fix signs for DLS2 convention before sending to hardware
        target_joints[0] = -target_joints[0]
        target_joints[6] = -target_joints[6]

        msg = TrajectoryGeneratorMsg()
        msg.joints_position = target_joints.flatten().tolist()
        msg.joints_velocity = np.zeros(12).tolist()
        self.pub_traj.publish(msg)

    def compute_rl_control(self):
        if not (self.first_base_arrived and self.first_joints_arrived):
            return

        # 1. Sync the real hardware state into the MuJoCo internal kinematic tree
        self.player.env.sync_mujoco_state(self.qpos, self.qvel)

        # 2. Execute the existing DeploymentPlayer step (forward pass + publish via our override)
        _ = self.player.play()


def main(args=None):
    rclpy.init(args=args)
    node = Parkour_Go2_Deployment_Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()