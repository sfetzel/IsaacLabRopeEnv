# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from .rope_spawner import RopeSpawnerCfg
from isaaclab.assets import RigidObjectCfg
from . import mdp
from math import pi

##
# Pre-defined configs
##

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip
from isaaclab_assets import UR10e_ROBOTIQ_GRIPPER_CFG  # isort:skip
from isaaclab.sim.spawners.from_files import UsdFileCfg

##
# Scene definition
##

UR10e_ROBOTIQ_CFG = UR10e_ROBOTIQ_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
UR10e_ROBOTIQ_CFG.init_state.joint_pos["shoulder_pan_joint"] = 0.0
UR10e_ROBOTIQ_CFG.init_state.joint_pos["shoulder_lift_joint"] = -50.0 / 180.0 * pi
UR10e_ROBOTIQ_CFG.init_state.joint_pos["elbow_joint"] = 50.0 / 180.0 * pi
UR10e_ROBOTIQ_CFG.init_state.joint_pos["wrist_1_joint"] = -90.0 / 180.0 * pi
UR10e_ROBOTIQ_CFG.init_state.joint_pos["wrist_2_joint"] = -90.0 / 180.0 * pi


@configclass
class RopeknotSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # rigid object
    object: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Rope",
        spawn=RopeSpawnerCfg(),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.5, 0.0)),
    )

    # articulation
    robot: ArticulationCfg = UR10e_ROBOTIQ_CFG.copy()


##
# MDP settings
##


from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab_tasks.manager_based.manipulation.stack.mdp.franka_stack_events import (
    set_default_joint_pose,
    randomize_joint_by_gaussian_offset,
)


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=[".*_joint"],
        body_name="wrist_3_link",
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=True, ik_method="dls"
        ),
        scale=1.0,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=[0.0, 0.0, 0.0]
        ),
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "finger_joint",
        ],
        open_command_expr={
            "finger_joint": -pi / 4,
        },
        close_command_expr={
            "finger_joint": pi / 4,
        },
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        # eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        # gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


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


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: keep pole upright
    # pole_pos = RewTerm(
    #    func=mdp.joint_pos_target_l2,
    #    weight=-1.0,
    #    params={
    #        "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
    #        "target": 0.0,
    #    },
    # )
    # (4) Shaping tasks: lower cart velocity
    # cart_vel = RewTerm(
    #    func=mdp.joint_vel_l1,
    #    weight=-0.01,
    #    params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    # )
    # (5) Shaping tasks: lower pole angular velocity
    # pole_vel = RewTerm(
    #    func=mdp.joint_vel_l1,
    #    weight=-0.005,
    #    params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class RopeknotEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: RopeknotSceneCfg = RopeknotSceneCfg(
        num_envs=32, env_spacing=2.0, replicate_physics=False
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
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        print(self.events)
        self.sim.render_interval = self.decimation
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.02,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )
