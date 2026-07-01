import mujoco
import torch as th
from core.utils import get_entity_name, get_entity_id, ISAAC_JOINT_NAMES, mujoco_to_isaac, isaac_to_mujoco
import numpy as np
from typing import Tuple
import re
from scipy.spatial.transform import Rotation
from dataclasses import dataclass

@dataclass
class TimestampedBuffer:
    """A buffer class containing data and its timestamp.

    This class is a simple data container that stores a tensor and its timestamp. The timestamp is used to
    track the last update of the buffer. The timestamp is set to -1.0 by default, indicating that the buffer
    has not been updated yet. The timestamp should be updated whenever the data in the buffer is updated. This
    way the buffer can be used to check whether the data is outdated and needs to be refreshed.

    The buffer is useful for creating lazy buffers that only update the data when it is outdated. This can be
    useful when the data is expensive to compute or retrieve. For example usage, refer to the data classes in
    the :mod:`isaaclab.assets` module.
    """

    data: th.Tensor = None  # type: ignore
    """The data stored in the buffer. Default is None, indicating that the buffer is empty."""

    timestamp: float = -1.0
    """Timestamp at the last update of the buffer. Default is -1.0, indicating that the buffer has not been updated."""


@th.jit.script
def quat_apply_inverse(quat: th.Tensor, vec: th.Tensor) -> th.Tensor:
    """Apply an inverse quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    # store shape
    shape = vec.shape
    # reshape to (N, 3) for multiplication
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    # extract components from quaternions
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec - quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)

import logging
logger = logging.getLogger(__name__)

class MujocoArticulation():
    """
    Wrapping Mujoco data into IsaacLab style
    """
    def __init__(self, env_cfg ,model: mujoco.MjModel, data:mujoco.MjData ):
        self._model = model
        self._data = data
        self._env_cfg = env_cfg
        actuator_consistency = self._check_actuator_consistency()
        assert actuator_consistency, "Only support that all the actuator use the same control type."
        self._sim_timestamp = 0.0
        self._num_motor = self._model.nu
        self._dim_motor_sensor = 3 * self._num_motor
        self._has_free_joint = self._model.nv != self._model.nq
        self.joint_pos_offset = self._model.nq - self._model.nu  # Because positions are in generalized coordinates
        self.joint_vel_offset = self._model.nv - self._model.nu  # Because velocities include the free joint
        self._device = 'cuda' if th.cuda.is_available() else 'cpu'
        self._init_buffer()
        self._init_joint_info()

    def _init_joint_info(self):
        self._joint_stiffness = th.zeros(self._num_motor).to(self._device)
        self._joint_dampings = th.zeros(self._num_motor).to(self._device)

        self._saturation_effort = th.zeros((1, self._num_motor), device=self.device)
        self._velocity_limit = th.zeros((1, self._num_motor), device=self.device)
        self._effort_limit = th.zeros((1, self._num_motor), device=self.device)
        
        # New arrays for PaceDCMotor
        self._encoder_bias = th.zeros(self._num_motor).to(self._device)
        self.torque_delay = 0

        # Loop through all actuator groups dynamically (e.g., 'hip', 'thigh', 'calf')
        for actuator_group_name, actuator_cfg in self._env_cfg.scene.robot.actuators.items():
            
            # Find which joints belong to this specific actuator group
            group_joints = []
            exprs = actuator_cfg.joint_names_expr if isinstance(actuator_cfg.joint_names_expr, list) else [actuator_cfg.joint_names_expr]
            for expr in exprs:
                compiled_expr = re.compile(expr)
                for idx, real_joint_name in enumerate(ISAAC_JOINT_NAMES):
                    if compiled_expr.match(real_joint_name):
                        group_joints.append((idx, real_joint_name))

            # 1. Parse Stiffness
            if hasattr(actuator_cfg.stiffness, 'items'):
                for pattern, value in actuator_cfg.stiffness.items():
                    compiled_pattern = re.compile(pattern)
                    for idx, real_joint_name in group_joints:
                        if compiled_pattern.match(real_joint_name):
                            self._joint_stiffness[idx] = value
            else:
                for idx, _ in group_joints:
                    self._joint_stiffness[idx] = actuator_cfg.stiffness

            # 2. Parse Damping
            if hasattr(actuator_cfg.damping, 'items'):
                for pattern, value in actuator_cfg.damping.items():
                    compiled_pattern = re.compile(pattern)
                    for idx, real_joint_name in group_joints:
                        if compiled_pattern.match(real_joint_name):
                            self._joint_dampings[idx] = value
            else:
                for idx, _ in group_joints:
                    self._joint_dampings[idx] = actuator_cfg.damping

            # 3. Parse Saturation Effort
            if hasattr(actuator_cfg.saturation_effort, 'items'):
                for pattern, value in actuator_cfg.saturation_effort.items():
                    compiled_pattern = re.compile(pattern)
                    for idx, real_joint_name in group_joints:
                        if compiled_pattern.match(real_joint_name):
                            self._saturation_effort[0, idx] = value
            else:
                for idx, _ in group_joints:
                    self._saturation_effort[0, idx] = actuator_cfg.saturation_effort

            # 4. Parse Velocity Limit
            if hasattr(actuator_cfg.velocity_limit, 'items'):
                for pattern, value in actuator_cfg.velocity_limit.items():
                    compiled_pattern = re.compile(pattern)
                    for idx, real_joint_name in group_joints:
                        if compiled_pattern.match(real_joint_name):
                            self._velocity_limit[0, idx] = value
            else:
                for idx, _ in group_joints:
                    self._velocity_limit[0, idx] = actuator_cfg.velocity_limit

            # 5. Parse Effort Limit
            if hasattr(actuator_cfg.effort_limit, 'items'):
                for pattern, value in actuator_cfg.effort_limit.items():
                    compiled_pattern = re.compile(pattern)
                    for idx, real_joint_name in group_joints:
                        if compiled_pattern.match(real_joint_name):
                            self._effort_limit[0, idx] = value
            else:
                for idx, _ in group_joints:
                    self._effort_limit[0, idx] = actuator_cfg.effort_limit
                    
            # 6. Parse Encoder Bias (For PaceDCMotor)
            if hasattr(actuator_cfg, 'encoder_bias') and actuator_cfg.encoder_bias is not None:
                if hasattr(actuator_cfg.encoder_bias, 'items'):
                    for pattern, value in actuator_cfg.encoder_bias.items():
                        compiled_pattern = re.compile(pattern)
                        for idx, real_joint_name in group_joints:
                            if compiled_pattern.match(real_joint_name):
                                self._encoder_bias[idx] = value
                else:
                    for idx, _ in group_joints:
                        self._encoder_bias[idx] = actuator_cfg.encoder_bias

            # 7. Parse Torque Delay (For PaceDCMotor)
            if hasattr(actuator_cfg, 'max_delay') and actuator_cfg.max_delay is not None:
                self.torque_delay = max(self.torque_delay, int(actuator_cfg.max_delay))

        # 8. Mass and COM
        self._body_com = np.zeros(3)
        self._total_mass = 0.0
        base_link_id = self.get_body_ids(['base_link'])['base_link']
        self._body_mass = self._model.body_mass[base_link_id]
        self._com = self._data.subtree_com[base_link_id]
        self._body_com += self._body_mass * self._com
        self._total_mass += self._body_mass
        self._body_com = th.from_numpy(self._body_com).to(self._device)/ self._total_mass
        self._body_mass = th.tensor([self._total_mass]).to(self._device)
        self._joint_efforts = th.zeros((1, self._num_motor), dtype=th.float, device=self.device)
        self._control_joint_velocities = th.zeros((1, self._num_motor), dtype=th.float, device=self.device)
        self._zeros_effort = th.zeros((1, self._num_motor), device=self.device)

    @property
    def saturation_effort(self):
        return self._saturation_effort

    @property
    def velocity_limit(self):
        return self._velocity_limit

    @property
    def control_joint_velocities(self):
        return self._control_joint_velocities

    @property
    def zeros_effort(self):
        return self._zeros_effort

    @property
    def effort_limit(self):
        return self._effort_limit

    @property
    def joint_efforts(self):
        return self._joint_efforts

    @joint_efforts.setter
    def joint_efforts(self, value: th.Tensor):
        self._joint_efforts[:] = value

    @property
    def body_com(self):
        return self._body_com

    @property
    def body_mass(self):
        return self._body_mass

    @property
    def joint_stiffness(self):
        return self._joint_stiffness

    @property
    def joint_dampings(self):
        return self._joint_dampings

    @property
    def body_names(self) -> list[str]:
        return [get_entity_name(self._model, "body", i) for i in range(1, self._model.nbody)]

    def get_body_ids(self, body_names: list[str] | None = None, free_joint_offset: int = 1) -> dict[str, int]:
        body_names_ = body_names if body_names else self.body_names
        body_ids = {}
        for name in body_names_:
            id_ = get_entity_id(self._model, "body", name)
            if id_ > 0:
                body_ids[name] = id_ - free_joint_offset
            else:
                body_ids[name] = id_
        return body_ids

    @property
    def joint_names(self) -> list[str]:
        offset = 0
        if self._has_free_joint:
            offset = 1
        return [get_entity_name(self._model, "joint", i) for i in range(offset, self._model.njnt)]

    def get_joint_ids(self, joint_names: list[str] | None = None, free_joint_offset: int = 1) -> dict[str, int]:
        joint_name_ = joint_names if joint_names else self.joint_names
        joint_ids = {}
        for name in joint_name_:
            id_ = get_entity_id(self._model, "joint", name)
            if id_ > 0:
                joint_ids[name] = id_ - free_joint_offset
            else:
                joint_ids[name] = id_
        return joint_ids

    def _check_actuator_consistency(self):
        """Check whether all the actuators share the same control mode."""
        actuator_type_system = None
        for actuator_id in range(self._model.nu):
            actuator_type = self._model.actuator_trntype[actuator_id]
            if actuator_type_system is None:
                actuator_type_system = actuator_type
            else:
                if actuator_type_system != actuator_type:
                    return False
        return True

    def update(self, dt: float):
        self._sim_timestamp += dt

    def _init_buffer(self):
        self._root_state_w = TimestampedBuffer()
        self._body_state_w = TimestampedBuffer()
        self._joint_pos = TimestampedBuffer()
        self._joint_vel = TimestampedBuffer()


    @property
    def root_state_w(self):
        bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
        if bid < 0:
            raise ValueError(f"Body 'base_link' not found")
        pos_w   = th.from_numpy(self._data.xpos[bid].copy()).to(device=self._device)
        quat_w  = th.from_numpy(self._data.xquat[bid].copy()).to(device=self._device)

        cvel_local = self._data.cvel[bid].copy()
        R = self._data.xmat[bid].reshape(3, 3).copy()
        ang_w = R @ cvel_local[0:3]
        lin_w = R @ cvel_local[3:6]

        ang_w = th.from_numpy(ang_w).to(device=self._device)
        lin_w = th.from_numpy(lin_w).to(device=self._device)

        root_state = th.cat([pos_w, quat_w, lin_w, ang_w], dim=0).unsqueeze(0)
        return root_state


    @property
    def joint_pos(self):
        if self._joint_pos.timestamp < self._sim_timestamp:
            # read data from simulation and set the buffer data and timestamp
            self._joint_pos.data = th.from_numpy(self._data.sensordata[:self._num_motor].copy())\
                                    .to(dtype=th.float32, device=self._device).expand(1, -1)
            self._joint_pos.timestamp = self._sim_timestamp
        return self._joint_pos.data[:,mujoco_to_isaac]

    @property
    def joint_vel(self):
        if self._joint_vel.timestamp < self._sim_timestamp:
            # read data from simulation and set the buffer data and timestamp
            self._joint_vel.data = th.from_numpy(self._data.sensordata[self._num_motor:self._num_motor+self._num_motor].copy())\
                                    .to(dtype=th.float32, device=self._device).expand(1, -1)
            self._joint_vel.timestamp = self._sim_timestamp
        return self._joint_vel.data[:,mujoco_to_isaac]

    @property
    def root_quat_w(self) -> th.Tensor:
        return self.root_state_w[:, 3:7]

    @property
    def root_ang_vel_w(self) ->th.Tensor:
        return self.root_state_w[:, 10:13]

    @property
    def root_ang_vel_b(self) -> th.Tensor:
        return quat_apply_inverse(self.root_quat_w, self.root_ang_vel_w)

    @property
    def device(self):
        return self._device

    @property
    def num_motor(self):
        return self._num_motor

    @joint_vel.setter
    def joint_vel(self, value: th.Tensor):
        assert value.shape[-1] == self._num_motor, \
            f"joint_vel must have shape (1, {self._num_motor}), but got {value.shape}"
        value_np = value.squeeze(0).detach().cpu().numpy()
        self._data.sensordata[:self._num_motor] = value_np
        self._joint_vel.data = value.clone().detach()
        self._joint_vel.timestamp = self._sim_timestamp