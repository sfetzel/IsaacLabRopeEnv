from .rope import RopeFactory
from pxr import Usd
import re
import carb
import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.utils.stage import get_current_stage
import isaaclab.sim as sim_utils
from isaaclab.sim import SpawnerCfg, RigidObjectSpawnerCfg
from isaaclab.utils import configclass
from isaaclab.sim import schemas


def spawn_multi_asset(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    clone_in_fabric: bool = False,
    replicate_physics: bool = False,
) -> Usd.Prim:
    # get stage handle
    stage = get_current_stage()

    # resolve: {SPAWN_NS}/AssetName
    # note: this assumes that the spawn namespace already exists in the stage
    root_path, asset_path = prim_path.rsplit("/", 1)
    # check if input is a regex expression
    # note: a valid prim path can only contain alphanumeric characters, underscores, and forward slashes
    is_regex_expression = re.match(r"^[a-zA-Z0-9/_]+$", root_path) is None

    # resolve matching prims for source prim path expression
    if is_regex_expression and root_path != "":
        source_prim_paths = sim_utils.find_matching_prim_paths(root_path)
        # if no matching prims are found, raise an error
        if len(source_prim_paths) == 0:
            raise RuntimeError(
                f"Unable to find source prim path: '{root_path}'. Please create the prim before spawning."
            )
    else:
        source_prim_paths = [root_path]

    # resolve prim paths for spawning
    prim_paths = [
        f"{source_prim_path}/{asset_path}" for source_prim_path in source_prim_paths
    ]

    for index, prim_path in enumerate(prim_paths):
        # spawn single instance
        factory = RopeFactory(1.0, translation)
        factory.coneAngleLimit = cfg.coneAngleLimit
        factory.rope_stiffness = cfg.rope_stiffness
        factory.rope_damping = cfg.rope_damping
        factory.linkHalfLength = cfg.linkHalfLength
        factory.linkRadius = cfg.linkRadius

        factory.create(prim_path, stage)
        """# apply collision properties
        if cfg.collision_props is not None:
            schemas.define_collision_properties(prim_path, cfg.collision_props)

        # note: we apply rigid properties in the end to later make the instanceable prim
        # apply mass properties
        if cfg.mass_props is not None:
            schemas.define_mass_properties(prim_path, cfg.mass_props)
        # apply rigid body properties
        if cfg.rigid_props is not None:
            schemas.define_rigid_body_properties(prim_path, cfg.rigid_props)"""
    # set carb setting to indicate Isaac Lab's environments that different prims have been spawned
    # at varying prim paths. In this case, PhysX parser shouldn't optimize the stage parsing.
    # the flag is mainly used to inform the user that they should disable `InteractiveScene.replicate_physics`
    carb_settings_iface = carb.settings.get_settings()
    carb_settings_iface.set_bool("/isaaclab/spawn/multi_assets", True)

    # return the prim
    return prim_utils.get_prim_at_path(prim_paths[0])


@configclass
class RopeSpawnerCfg(RigidObjectSpawnerCfg):
    func = spawn_multi_asset
    rope_damping = 100
    rope_stiffness = 50
    coneAngleLimit = 50
    linkHalfLength = 0.05
    linkRadius = 0.5 * 0.05
