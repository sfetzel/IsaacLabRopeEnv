
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import torch.nn as nn
from torchvision import models

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.envs.mdp import image

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


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
        x = self.enc0(x)      # 64, 112x112
        x = self.pool(x)     # 64, 56x56
        x = self.enc1(x)     # 64, 56x56
        x = self.enc2(x)     # 128, 28x28
        x = self.enc3(x)     # 256, 14x14
        x = self.enc4(x)     # 512, 7x7

        return x


class image_features_resnet18(ManagerTermBase):
    """Extracted image features from a pre-trained frozen encoder.

    This term uses ResNet18 from the model zoo in PyTorch and extracts features from the images.

    It calls the :func:`image` function to get the images and then processes them using the model zoo.

    Args:
        sensor_cfg: The sensor configuration to poll. Defaults to SceneEntityCfg("tiled_camera").

    Returns:
        The extracted features tensor. Shape is (num_envs, feature_dim).
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)
        self.model_device: str = cfg.params.get("model_device", env.device)  # type: ignore
        self.model = ResNet18_Features().to(self.model_device)

    def reset(self, env_ids: torch.Tensor | None = None):
        pass

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
        data_type: str = "rgb",
        model_device: str | None = None,
    ) -> torch.Tensor:
        # obtain the images from the sensor
        image_data = image(
            env=env,
            sensor_cfg=sensor_cfg,
            data_type=data_type,
            normalize=False,  # we pre-process based on model
        )
        # store the device of the image
        image_device = image_data.device

        # move the image to the model device
        image_proc = image_data.to(self.model_device)
        # permute the image to (num_envs, channel, height, width)
        image_proc = image_proc.permute(0, 3, 1, 2).float() / 255.0
        # normalize the image
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.model_device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.model_device).view(1, 3, 1, 1)
        image_proc = (image_proc - mean) / std

        # forward the images through the model
        features = self.model(image_proc)

        # move the features back to the image device
        return features.detach().flatten(start_dim=1).to(image_device)
