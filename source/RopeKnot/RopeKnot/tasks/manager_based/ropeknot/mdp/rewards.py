# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, transform_points
from isaaclab.sensors import TiledCamera
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import numpy as np
import cv2


# ---------- Main model ----------
class ResNet18_Features(nn.Module):
    def __init__(self, pretrained=True, freeze_encoder=True):
        super().__init__()

        resnet = models.resnet18(pretrained=pretrained)

        # ----- Encoder -----
        self.enc0 = nn.Sequential(
            resnet.conv1,  # 64, 112x112
            resnet.bn1,
            resnet.relu
        )
        self.pool = resnet.maxpool  # -> 56x56

        self.enc1 = resnet.layer1   # 64, 56x56
        self.enc2 = resnet.layer2   # 128, 28x28
        self.enc3 = resnet.layer3   # 256, 14x14
        self.enc4 = resnet.layer4   # 512, 7x7

        # Freeze encoder if requested
        if freeze_encoder:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x):

        # ----- Encoder -----
        x0 = self.enc0(x)      # 64, 112x112
        x1 = self.pool(x0)     # 64, 56x56
        x1 = self.enc1(x1)     # 64, 56x56
        x2 = self.enc2(x1)     # 128, 28x28
        x3 = self.enc3(x2)     # 256, 14x14
        #x4 = self.enc4(x3)     # 512, 7x7

        return x0, x1, x2, x3, None


feature_encoder = ResNet18_Features()
feature_encoder.eval()
segmentation_model = torch.jit.load("segmentation2.pt")
segmentation_model.eval()
reward_model = torch.jit.load("reward-8.pt")
reward_model.eval()


def model_reward(env: ManagerBasedRLEnv, camera_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    camera: TiledCamera = env.scene[camera_cfg.name]
    data = camera.data.output["rgb"]  # (envs, H, W, C) in format (0,255)
    
    model_device = env.device

    with torch.no_grad():
        feature_encoder.to(model_device)
        segmentation_model.to(model_device)
        # move the image to the model device
        image_proc = data.to(model_device)
        # permute the image to (num_envs, channel, height, width)
        image_proc = image_proc.permute(0, 3, 1, 2).float() / 255.0
        # normalize the image
        mean = torch.tensor([0.485, 0.456, 0.406], device=model_device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=model_device).view(1, 3, 1, 1)
        image_proc = (image_proc - mean) / std

        # forward the images through the model
        image_features = feature_encoder(image_proc)  # (x0, x1, x2, x3, x4).
        env._cached_image_features = image_features

        masks = torch.sigmoid(segmentation_model(*image_features))
        if hasattr(env, "_last_masks"):
            beta = 0.9
            masks = (1 - beta) * masks + beta * env._last_masks
        env._last_masks = masks
        rewards = reward_model(masks)
        env._cached_masks = masks

        last_rewards = torch.zeros(env.num_envs, device=env.device)
        if hasattr(env, "last_rewards"):
            last_rewards = env.last_rewards
        beta = 0.0
        result = (1 - beta) * rewards + beta * last_rewards
        env.last_rewards = result

        return result


def mask_size(env):
    mask = env._cached_masks

    if not hasattr(env, "_cached_mask_size"):
        env._cached_mask_size = env._cached_masks.flatten(start_dim=1).sum(dim=1)

    # check which environments have been reset and reset their mask.
    masks_to_update = env._cached_mask_size == 0.0
    env._cached_mask_size[masks_to_update] = env._cached_masks[masks_to_update].flatten(start_dim=1).sum(dim=1)

    masks_flattened = mask.flatten(start_dim=1)

    return masks_flattened.sum(dim=1) / env._cached_mask_size


def close_to_mask(env, camera_cfg: SceneEntityCfg, ee_cfg: SceneEntityCfg):
    mask = env._cached_masks

    B, C, H, W = mask.shape
    mask = mask[:, 0]  # assume single-channel mask -> (B,H,W)

    camera: TiledCamera = env.scene[camera_cfg.name]
    depth = camera.data.output["depth"]

    num_samples = 64

    # --------------------------------------------------
    # create pixel coordinate grid
    # --------------------------------------------------

    ys = torch.arange(H, device=mask.device)
    xs = torch.arange(W, device=mask.device)

    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    coords = torch.stack((grid_x, grid_y), dim=-1)  # (H,W,2)
    coords = coords.view(-1, 2)                     # (H*W,2)

    coords = coords.unsqueeze(0).expand(B, -1, -1)  # (B,H*W,2)

    mask_flat = mask.view(B, -1)

    sampled_coords = []

    for b in range(B):
        valid_coords = coords[b][mask_flat[b] > 0.95]  # only mask pixels

        if valid_coords.shape[0] == 0:
            idx = torch.randint(0, H * W, (num_samples,), device=mask.device)
            sampled = coords[b, idx]
        else:
            idx = torch.randint(
                0,
                valid_coords.shape[0],
                (num_samples,),
                device=mask.device
            )
            sampled = valid_coords[idx]

        sampled_coords.append(sampled)

    sampled_coords = torch.stack(sampled_coords)  # (B,num_samples,2)

    u = sampled_coords[:, :, 0]
    v = sampled_coords[:, :, 1]

    # --------------------------------------------------
    # depth sampling
    # --------------------------------------------------

    depth_flat = depth.view(B, -1)
    linear_idx = v * W + u

    z = torch.gather(depth_flat, 1, linear_idx)

    # --------------------------------------------------
    # backprojection
    # --------------------------------------------------

    K = camera.data.intrinsic_matrices

    fx = K[:, 0, 0][:, None]
    fy = K[:, 1, 1][:, None]
    cx = K[:, 0, 2][:, None]
    cy = K[:, 1, 2][:, None]

    #x = (u - cx) * z / fx
    #y = (v - cy) * z / fy

    X_cam = z  # depth along optical axis
    Y_cam = (u - cx) * z / fx
    Z_cam = -(v - cy) * z / fy
    local_points = torch.stack((X_cam, Y_cam, Z_cam), dim=-1)

    # local_points = torch.stack((x, y, z), dim=2)

    # --------------------------------------------------
    # camera -> world
    # --------------------------------------------------

    camera_pos = camera.data.pos_w
    camera_quat = camera.data.quat_w_world

    points_world = transform_points(local_points, camera_pos, camera_quat)  # (B, 64, 3)

    # --------------------------------------------------
    # EE distance
    # --------------------------------------------------

    robot = env.scene[ee_cfg.name]
    ee_cfg.resolve(env.scene)

    ee_pos = robot.data.body_pos_w[:, ee_cfg.body_ids[0]]

    distances = torch.linalg.vector_norm(
        points_world - ee_pos.unsqueeze(1),
        dim=-1
    )
    min_dist = distances.min(dim=1).values

    reward = torch.exp(-2 * min_dist)

    return reward


def mask_change(env):
    result = 0
    if hasattr(env, "_last_masks"):
        current_masks = env._cached_masks.flatten(start_dim=1)
        last_masks = env._last_masks.flatten(start_dim=1)

        return torch.mean(torch.abs(current_masks - last_masks), dim=1)

    env._last_masks = env._cached_masks
    return result


def step_penalty(env):
    return torch.ones(env.num_envs, device=env.device)


def ee_target_distance(env, ee_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg):
    """Penalize distance between end effector and target object."""

    # get positions
    robot = env.scene[ee_cfg.name]
    target = env.scene[target_cfg.name]
    ee_cfg.resolve(env.scene)

    ee_pos = robot.data.body_pos_w[:, ee_cfg.body_ids[0]]
    target_pos, _ = target.get_world_poses()

    # if target is shared across envs expand it
    if target_pos.shape[0] == 1:
        target_pos = target_pos.expand(env.num_envs, -1)

    # compute euclidean distance
    dist = torch.norm(ee_pos - target_pos, dim=-1)
    clamped_dist = torch.clamp(dist - 0.1, min=0.0)

    exp_dist = torch.exp(-clamped_dist * 2)
    # penalty
    return exp_dist


def ee_orientation_action_penalty(env: ManagerBasedRLEnv):
    """Penalize roll/pitch angular velocity commands."""

    actions = env.action_manager.action

    # assuming action = [vx, vy, vz, wx, wy, wz]
    angular_vel = actions[:, 3:6]

    # penalize roll and pitch only
    penalty = torch.sum(torch.square(angular_vel[:, 0:2]), dim=-1)

    return penalty
