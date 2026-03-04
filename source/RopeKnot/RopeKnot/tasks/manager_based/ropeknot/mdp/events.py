
from __future__ import annotations

import math
import random
import torch
from typing import TYPE_CHECKING

from isaacsim.core.utils.extensions import enable_extension

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, AssetBase
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_rope_joints(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    std_min: float,
    std_max: float,
    angle_rad: float,
    repeats: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    Randomizes the rope pose by modifying the "z" DOF rotation.
    The angles are modified such that they have the shape of a sum of gaussians.
    """
    rope: Articulation = env.scene[asset_cfg.name]

    # Add gaussian noise to joint states
    joint_pos = rope.data.default_joint_pos[env_ids].clone()
    joint_vel = rope.data.default_joint_vel[env_ids].clone()

    z_dofs = joint_pos[:, 2::3]
    device = z_dofs.device
    N = len(z_dofs[0])

    for _ in range(repeats):
        means = torch.rand(len(env_ids))
        stds = torch.rand(len(env_ids)) * (std_max - std_min) + std_min
        diff_to_mean = torch.linspace(0, 1, N).repeat(len(env_ids), 1) - means.unsqueeze(1)
        z_dofs += (angle_rad * torch.exp(
            -(diff_to_mean / stds.unsqueeze(1))**2).to(device)
        )

    joint_pos[:, 2::3] = torch.clip(z_dofs, 0, 0.5)

    joint_pos_limits = rope.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    
    rope.set_joint_position_target(joint_pos, env_ids=env_ids)
    rope.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    rope.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # It might happen that the rope is rotated and moved,
    # therefore its root state needs to be reset.
    root_state = rope.data.default_root_state.clone()

    # Add environment origins
    root_state[:, 0:3] += env.scene.env_origins[env_ids]

    rope.write_root_state_to_sim(root_state, env_ids=env_ids)
