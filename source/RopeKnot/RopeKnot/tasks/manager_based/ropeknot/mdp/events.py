
from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaacsim.core.prims import RigidPrim

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def bend(d, target_angle, pos, orientation):
    angles = torch.linspace(0, target_angle, pos.shape[0], device=pos.device)

    pos[:, 0] = torch.cumsum(d * torch.cos(angles), 0)
    pos[:, 1] = torch.cumsum(d * torch.sin(angles), 0)
    orientation *= 0
    orientation[:, 0] = torch.cos(angles * 0.5)
    orientation[:, 3] = torch.sin(angles * 0.5)


def randomize_rope_joints(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    angle_min: float,
    angle_max: float,
    capsule_subpath: str,
    rope_path: str,
    capsule_distance: float = 0.02,
):
    """
    Randomizes the rope pose by modifying the "z" DOF rotation.
    The angles are modified such that they have the shape of a sum of gaussians.
    """
    prims = None
    if hasattr(env, "_cache_rope_rigidprim"):
        prims = env._cache_rope_rigidprim
    else:
        prims = RigidPrim(prim_paths_expr=f"/World/envs/env_.*/{rope_path}" + capsule_subpath, name="rigid_prim_view")
        env._cache_rope_rigidprim = prims

    num_envs = len(env_ids)
    ropes = [f"/World/envs/env_{id}/{rope_path}" for id in env_ids]
    paths = prims.prim_paths

    all_pos, all_orient = None, None
    d = capsule_distance  # distance between capsules.
    angles = torch.distributions.uniform.Uniform(torch.tensor([angle_min]), torch.tensor([angle_max]))

    target_ids = []
    N = None  # number of capsules per rope.
    for rope_index, rope_path in enumerate(ropes):

        ids = [i for i in range(prims.count) if paths[i].startswith(rope_path)]  # ids in prim of this rope.

        if all_pos is None or all_orient is None:
            N = len(ids)  # number of capsules per rope.
            all_orient = torch.zeros((num_envs * N, 4))
            all_pos = torch.zeros((num_envs * N, 3))

        start_idx = rope_index * N  # start index in all_orient, all_pos tensors.
        end_idx = (rope_index + 1) * N
        pos = all_pos[start_idx:end_idx]
        orient = all_orient[start_idx:end_idx]

        center = pos.shape[0] // 2
        bend(d, angles.sample().item(), pos[center:, :], orient[center:, :])
        bend(-d, -angles.sample().item(), pos[:center, :], orient[:center, :])
        pos[center, :2] *= 0  # reset center position

        # correct ordering.
        pos[:center, :] = torch.flip(pos[:center, :], dims=(0,))
        orient[:center, :] = torch.flip(orient[:center, :], dims=(0,))

        all_pos[start_idx:end_idx, :] = pos
        all_orient[start_idx:end_idx, :] = orient
        target_ids.extend(ids)

    prims.set_local_poses(all_pos, all_orient, indices=target_ids)
    prims.set_velocities(torch.zeros((num_envs * N, 6)), indices=target_ids)


def clean_cache(env: ManagerBasedEnv, env_ids: torch.Tensor):
    if hasattr(env, "_cached_mask_size"):
        env._cached_mask_size[env_ids] = 0.0
    
    if hasattr(env, "_cached_masks"):
        env._cached_masks[env_ids] *= 0.0
    if hasattr(env, "last_rewards"):
        env.last_rewards[env_ids] *= 0.0

