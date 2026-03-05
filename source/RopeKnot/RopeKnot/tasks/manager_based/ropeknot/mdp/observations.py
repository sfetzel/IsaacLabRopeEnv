
from __future__ import annotations

import torch
from typing import TYPE_CHECKING


from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.envs.mdp import image

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def cached_image_features_resnet18(env: ManagerBasedEnv) -> torch.Tensor:
    """Cached image features."""
    result = torch.zeros(env.num_envs, 512*7*7).to(env.device)
    if hasattr(env, "_cached_image_features"):
        result = env._cached_image_features[-1].flatten(start_dim=1)
    else:
        print("Warning, no image features found.")

    return result


