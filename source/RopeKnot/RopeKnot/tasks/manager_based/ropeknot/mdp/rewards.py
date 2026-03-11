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
from isaaclab.utils.math import wrap_to_pi
from isaaclab.sensors import TiledCamera

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
        x4 = self.enc4(x3)     # 512, 7x7

        return x0, x1, x2, x3, x4


feature_encoder = ResNet18_Features()
feature_encoder.eval()
reward_model = torch.jit.load("segmentation+reward.pt")
reward_model.eval()


def model_reward(env: ManagerBasedRLEnv, camera_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    camera: TiledCamera = env.scene[camera_cfg.name]
    data = camera.data.output["rgb"]  # (envs, H, W, C) in format (0,255)
    model_device = env.device

    with torch.no_grad():
        feature_encoder.to(model_device)
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

        reward_model.to(model_device)
        rewards, masks = reward_model(*image_features)
        rewards = rewards.squeeze(-1)
        env._cached_masks = masks

        result = rewards
        return result


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
