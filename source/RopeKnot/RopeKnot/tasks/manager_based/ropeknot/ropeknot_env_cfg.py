# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from typing import Any, Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg, CameraCfg, Camera
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaacsim.core.prims import XFormPrim
from isaaclab.assets import RigidObjectCfg, AssetBase
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR
from isaaclab.sensors.contact_sensor import ContactSensorCfg
from . import mdp
from math import pi
import numpy as np
import os


##
# Pre-defined configs
##
from isaaclab.actuators import ImplicitActuatorCfg

##
# Scene definition
##

UR5e_ROBOTIQ_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{os.path.dirname(os.path.abspath(__file__))}/assets/ur5e_robotiq.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=True,
        semantic_tags=[("class", "robot")]
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0,
            "shoulder_lift_joint": -75.0 / 180.0 * pi,
            "elbow_joint": 75.0 / 180 * pi,
            "wrist_1_joint": -90.0 / 180 * pi,
            "wrist_2_joint": -90.0 / 180 * pi,
            "wrist_3_joint": 0.0,
        },
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        # 'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*"],
            stiffness=1320.0,
            damping=72.6636085,
            friction=0.0,
            armature=0.0,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"],
            stiffness=600.0,
            damping=34.64101615,
            friction=0.0,
            armature=0.0,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_.*"],
            stiffness=216.0,
            damping=29.39387691,
            friction=0.0,
            armature=0.0,
        ),
        "finger": ImplicitActuatorCfg(
            joint_names_expr=["Slider_.*"],
            stiffness=10.0,
            damping=0.1,
            friction=0.0,
            armature=0.0,
        ),
    },
)


@configclass
class RopeknotSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg(
            usd_path=f"{os.path.dirname(os.path.abspath(__file__))}/assets/ground_plane.usd",
        )
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # articulation
    robot: ArticulationCfg = UR5e_ROBOTIQ_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    rope: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Rope",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{os.path.dirname(os.path.abspath(__file__))}/assets/rope.usd",
            semantic_tags=[("class", "rope")]
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.6, 0.0, 0.01), rot=(0.7071067, 0, 0, 0.7071067)),
    )
    
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(1.2, 0.0, 1.0), rot=(-3.6920e-08, -3.8268e-01, -3.2020e-08,  9.2388e-01), convention="world"),
        data_types=["rgb"], #, "semantic_segmentation"
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=224,
        height=224,
        colorize_semantic_segmentation=False,
    )

    # unfortunately semantic filtering does not work per camera.
    """rope_semantic_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/RopeCamera",
        offset=CameraCfg.OffsetCfg(pos=(1.2, 0.0, 1.0), rot=(-3.6920e-08, -3.8268e-01, -3.2020e-08,  9.2388e-01), convention="world"),
        data_types=["semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=224,
        height=224,
        colorize_semantic_segmentation=False,
        semantic_filter="class : rope"
    )"""

    contact_sensor_left = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ee_link/left_gripper",
        update_period=0.0,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}" + f"/Rope/Rope/capsule_{i}" for i in range(60)],
    )
    contact_sensor_right = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ee_link/right_gripper",
        update_period=0.0,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}" + f"/Rope/Rope/capsule_{i}" for i in range(60)],
    )

##
# MDP settings
##


from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab_tasks.manager_based.manipulation.stack.mdp.franka_stack_events import (
    randomize_joint_by_gaussian_offset,
)

import torch
from isaaclab.envs.mdp.actions import task_space_actions
from isaaclab.envs import ManagerBasedEnv


class PositionWithFixedOrientationIKAction(task_space_actions.DifferentialInverseKinematicsAction):
    """Differential IK action that allows translation + yaw only."""

    def __init__(self, cfg: DifferentialInverseKinematicsActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, 7, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # save the scale as tensors
        self._scale = torch.zeros((self.num_envs, 7), device=self.device)
        self._scale[:] = torch.tensor(self.cfg.scale, device=self.device)
        self._desired_orientation = torch.zeros((self.num_envs, 4), device=self.device)
        self._desired_orientation[:] = torch.tensor([0.4921, -0.4994, 0.4992, -0.5093], device=self.device)
        # change in rad/s
        max_change = 120.0 / 180.0 * torch.pi
        time_step = env.physics_dt * env.cfg.sim.render_interval
        self.max_delta = max_change * time_step

    @property
    def action_dim(self):
        # expose only 3 actions to the policy
        return 3

    def process_actions(self, actions: torch.Tensor):
        """
        Convert [vx, vy, vz] -> [vx, vy, vz, w, x, y, z]
        """
        full_actions = torch.cat((actions, self._desired_orientation), dim=1)
        return super().process_actions(full_actions)

    def apply_actions(self):
        # obtain quantities from simulation
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        # compute the delta in joint-space
        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()
        
        delta_q = joint_pos_des - joint_pos
        delta_q = torch.clamp(delta_q, -self.max_delta, self.max_delta)

        new_joint_pos_des = joint_pos + delta_q

        # set the joint position command
        self._asset.set_joint_position_target(new_joint_pos_des, self._joint_ids)


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        class_type=PositionWithFixedOrientationIKAction,
        joint_names=[".*_joint"],
        body_name="base_link_0",  # base link from hand-e
        controller=DifferentialIKControllerCfg(
            # use (pose and relative mode for teleoperation)
            # use (pose, class and absolute mode for training)
            command_type="pose", use_relative_mode=False, ik_method="dls"
        ),
        #scale=[[1.0, 1.0, 1.0, 0.1, 0.1, 1.0]],
        scale=1.0,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=[0.0, 0.0, 0.0]
        ),
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "Slider_1",
            "Slider_2",
        ],
        open_command_expr={
            "Slider_1": 0.0,
            "Slider_2": 0.0,
        },
        close_command_expr={
            "Slider_1": -0.025,
            "Slider_2": -0.025,
        },
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        #image_feat = ObsTerm(func=mdp.cached_image_features_resnet18)
        #joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        #joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_pos = ObsTerm(func=mdp.body_pose_w, params={
            "asset_cfg": SceneEntityCfg("robot", body_ids=-1)
        })
        current_time = ObsTerm(func=mdp.current_time_s)
        mask = ObsTerm(func=mdp.cached_masks_flattened)

        # eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        # eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        # gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class DebugCfg(ObsGroup):
        image = ObsTerm(func=mdp.image)  # for debugging only
        mask = ObsTerm(func=mdp.cached_masks)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    debug: DebugCfg = DebugCfg()


def hide_robot(env, env_ids):
    robots = XFormPrim(prim_paths_expr="/World/envs/env_.*/Robot")
    robots.set_visibilities(visibilities=[False] * env.num_envs)

@configclass
class EventCfg:
    """Configuration for events."""

    # add randomization - this also sets the joint targets for the controllers.
    randomize_joint_state = EventTerm(
        func=randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    """hide_robot = EventTerm(
        func=hide_robot,
        mode="reset",
        params={}
    )"""

    randomize_rope_joint_state = EventTerm(
        func=mdp.randomize_rope_joints,
        mode="reset",
        params={
            "angle_min": 1.2,
            "angle_max": 1.71,
            "capsule_subpath": "/capsule_.*",
            "rope_path": "Rope/Rope"
        },
    )

    clean_data = EventTerm(
        func=mdp.clean_cache,
        mode="reset",
        params={}
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Reward for terminating early.
    terminating = RewTerm(func=mdp.is_terminated, weight=1.0)

    model = RewTerm(func=mdp.model_reward, weight=1.0, params={
        "camera_cfg": SceneEntityCfg("tiled_camera"),
    })

    """rope_occlusion = RewTerm(func=mdp.occlusion, weight=0.5, params={
        "camera_cfg": SceneEntityCfg("tiled_camera"),
        "object_camera_cfg": SceneEntityCfg("rope_semantic_camera"),
    })"""

    # discourage the robot from hiding the rope.
    #mask_size = RewTerm(func=mdp.mask_size, weight=0.)

    """close_to_mask = RewTerm(func=mdp.close_to_mask, weight=1.0, params={
        "camera_cfg": SceneEntityCfg("tiled_camera"),
        "ee_cfg": SceneEntityCfg("robot", body_names=["left_gripper"])
    })"""

    left_gripper_contact = RewTerm(
        func=mdp.desired_contacts_filtered,  # returns 1.0 when no contact and 0.0 when contact
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor_left"),
            "threshold": 0
        }
    )

    right_gripper_contact = RewTerm(
        func=mdp.desired_contacts_filtered,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor_right"),
            "threshold": 0
        }
    )

    """mask_change = RewTerm(
        func=mdp.mask_change, weight=0.01
    )"""

    # The Action Penalty
    """action_rate = RewTerm(
        func=mdp.action_l2,
        weight=-5e-2, # Negative weight to penalize
        params={}
    )"""
    
    # Penalty for change in actions (smoothness)
    action_control_glitch = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.2,
        params={}
    )

    ee_orientation_penalty = RewTerm(
        func=mdp.ee_orientation_penalty,
        weight=-0.1,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    finished = DoneTerm(func=mdp.done)

##
# Environment configuration
##


from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
import torch

@configclass
class RopeknotEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: RopeknotSceneCfg = RopeknotSceneCfg(
        num_envs=350, env_spacing=4.0,
    )
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    
    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 3
        self.episode_length_s = 5
        self.max_episode_length = 8
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1.0 / 120.0
        np.random.seed(self.seed)

        
        #self.markers = VisualizationMarkers(markers_cfg)

        """marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/myMarkers",
                markers={
                    "frame": sim_utils.UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                        scale=(0.5, 0.5, 0.5),
                    ),
                },
            )
        marker_orientations = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        self.markers = VisualizationMarkers(marker_cfg)
        self.markers.visualize(torch.tensor([[1.0, 1.0,]]), marker_orientations)"""

        self.sim.render_interval = self.decimation
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.02,
                    rot_sensitivity=0.5,
                    sim_device=self.sim.device,
                ),
            }
        )
