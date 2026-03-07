
from __future__ import annotations

import math
import random
import torch
from typing import TYPE_CHECKING

from isaacsim.core.utils.extensions import enable_extension

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, AssetBase, AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaacsim.core.prims import RigidPrim
import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def bend(d, target_angle, pos, orientation):
    angles = torch.linspace(0, target_angle, pos.shape[0], device=pos.device)

    pos[:, 0] = torch.cumsum(d * torch.cos(angles), 0)
    pos[:, 1] = torch.cumsum(d * torch.sin(angles), 0)
    orientation *= 0
    orientation[:, 0] = torch.cos(angles*0.5)
    orientation[:, 3] = torch.sin(angles*0.5)


def randomize_rope_joints(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    angle_min: float,
    angle_max: float,
    capsule_subpath: str,
    rope_path: str,
):
    """
    Randomizes the rope pose by modifying the "z" DOF rotation.
    The angles are modified such that they have the shape of a sum of gaussians.
    """
    # ropes = sim_utils.find_matching_prims("")
    ropes = [ f"/World/envs/env_{id}/{rope_path}" for id in env_ids ]

    for rope_path in ropes:
        prims = RigidPrim(prim_paths_expr=rope_path + capsule_subpath, name="rigid_prim_view")

        pos, orient = prims.get_local_poses()

        d = torch.norm(pos[:-1, :] - pos[1:, :], dim=1).mean()

        center = pos.shape[0] //2
        angles = torch.distributions.uniform.Uniform(torch.tensor([angle_min]), torch.tensor([angle_max]))
        bend(d, angles.sample().item(), pos[(center+1):, :], orient[(center+1):, :])
        bend(-d, -angles.sample().item(), pos[:center, :], orient[:center, :])
        pos[center, :2] *= 0 # reset center position

        # correct ordering.
        pos[:center, :] = torch.flip(pos[:center, :], dims=(0,))
        orient[:center, :] = torch.flip(orient[:center, :], dims=(0,))

        prims.set_local_poses(pos, orient)
        velocities = prims.get_velocities() * 0.0
        prims.set_velocities(velocities)

