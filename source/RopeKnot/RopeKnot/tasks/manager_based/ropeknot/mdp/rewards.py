# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.sensors import TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import numpy as np
import cv2

reward_model = torch.jit.load("dummy_reward_model.pt")
reward_model.eval()


def model_reward(env: ManagerBasedRLEnv, camera_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    # camera: TiledCamera = env.scene[camera_cfg.name]
    # data = camera.data.output["rgb"]  # (envs, H, W, C) in format (0,255)
    
    if "image_feat" in env.obs_buf["policy"]:
        image_features = env.obs_buf["policy"]["image_feat"]
        reward_model.to(image_features.device)
        with torch.no_grad():
            rewards = reward_model(image_features)
            
            return rewards.squeeze(-1)
    else:
        print("Could not find image feature vector in observations.")
    return 0
