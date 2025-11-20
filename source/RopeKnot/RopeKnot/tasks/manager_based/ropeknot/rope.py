from pxr import UsdGeom, Sdf, Gf, UsdPhysics, UsdShade, PhysxSchema
from omni.physx.scripts import physicsUtils
import numpy as np
import isaacsim.core.utils.prims as prim_utils


class RopeFactory:
    """
    Adapted from Isaac Sim Rope demo.
    """

    def __init__(self, rope_length, position=(0.0, 0.0, 0.0), density: float = 800):
        """
        density: float
            Density of the rope in kg/m³.
        """
        self.linkHalfLength = 0.003  # smaller value makes it smoother
        self.linkRadius = 0.5 * self.linkHalfLength  # 0.005
        self.ropeLength = rope_length
        # approximate volume as cylinder.
        self.volume = np.pi * self.linkRadius**2 * rope_length
        self.mass = self.volume * density
        self.numLinks = int(self.ropeLength / (2 * self.linkHalfLength))
        self.mass_per_element = self.mass / self.numLinks
        self.ropeSpacing = 1.50
        self.coneAngleLimit = 5
        self.rope_damping = 0.04 #10.0
        self.rope_stiffness = 0.05 # 1.0
        self.position = position
        self.contactOffset = 0.02 # 2.0
        self.capsuleZ = 0.0
        self.filter_collisions = False

    def create(self, prim_path: str, stage):
        self._defaultPrimPath = Sdf.Path(prim_path)

        physicsMaterialPath = self._defaultPrimPath.AppendChild("PhysicsMaterial")
        UsdShade.Material.Define(stage, physicsMaterialPath)
        material = UsdPhysics.MaterialAPI.Apply(
            stage.GetPrimAtPath(physicsMaterialPath)
        )
        material.CreateStaticFrictionAttr().Set(0.2)
        material.CreateDynamicFrictionAttr().Set(0.05)
        material.CreateRestitutionAttr().Set(0)

        self.createRope(self._defaultPrimPath, stage, physicsMaterialPath)

    def createCapsule(self, path: Sdf.Path, stage, physicsMaterialPath: Sdf.Path):
        capsuleGeom = UsdGeom.Capsule.Define(stage, path)
        capsuleGeom.CreateHeightAttr(self.linkHalfLength)
        capsuleGeom.CreateRadiusAttr(self.linkRadius)
        capsuleGeom.CreateAxisAttr("X")

        UsdPhysics.CollisionAPI.Apply(capsuleGeom.GetPrim())
        rigidBodyApi = UsdPhysics.RigidBodyAPI.Apply(capsuleGeom.GetPrim())
        #rigidBodyApi.CreateLinearDampingAttr(0.55)
        #rigidBodyApi.CreateAngularDampingAttr(0.55)
        physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(capsuleGeom.GetPrim())

        # force all actors in the scene to a fixed position and velocity iteration count
        posIters = 150
        velIters = 30
        physxSceneAPI.CreateMaxPositionIterationCountAttr().Set(posIters)
        physxSceneAPI.CreateMinPositionIterationCountAttr().Set(posIters)
        physxSceneAPI.CreateMaxVelocityIterationCountAttr().Set(velIters)
        physxSceneAPI.CreateMinVelocityIterationCountAttr().Set(velIters)
        physxSceneAPI.CreateTimeStepsPerSecondAttr(300)

        massAPI = UsdPhysics.MassAPI.Apply(capsuleGeom.GetPrim())
        # massAPI.CreateDensityAttr().Set(1e-15)
        massAPI.CreateMassAttr().Set(self.mass_per_element)
        physxCollisionAPI = PhysxSchema.PhysxCollisionAPI.Apply(capsuleGeom.GetPrim())
        physxCollisionAPI.CreateRestOffsetAttr().Set(0.0)
        physxCollisionAPI.CreateContactOffsetAttr().Set(self.contactOffset)
        physicsUtils.add_physics_material_to_prim(
            stage, capsuleGeom.GetPrim(), physicsMaterialPath
        )
        return capsuleGeom

    def createJoint(self, jointPath: Sdf.Path, stage):
        joint = UsdPhysics.Joint.Define(stage, jointPath)
        lower_limit = -0.05
        upper_limit = 0.05
        # locked DOF (lock - low is greater than high)
        d6Prim = joint.GetPrim()
        limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transX")
        limitAPI.CreateLowAttr(upper_limit)
        limitAPI.CreateHighAttr(lower_limit)
        limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transY")
        limitAPI.CreateLowAttr(upper_limit)
        limitAPI.CreateHighAttr(lower_limit)
        limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transZ")
        limitAPI.CreateLowAttr(upper_limit)
        limitAPI.CreateHighAttr(lower_limit)
        limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "rotX")
        limitAPI.CreateLowAttr(upper_limit)
        limitAPI.CreateHighAttr(upper_limit)

        # Moving DOF:
        dofs = ["rotY", "rotZ"]
        for d in dofs:
            limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, d)
            limitAPI.CreateLowAttr(-self.coneAngleLimit)
            limitAPI.CreateHighAttr(self.coneAngleLimit)

            # joint drives for rope dynamics:
            driveAPI = UsdPhysics.DriveAPI.Apply(d6Prim, d)
            driveAPI.CreateTypeAttr("force")
            driveAPI.CreateDampingAttr(self.rope_damping)
            driveAPI.CreateStiffnessAttr(self.rope_stiffness)
        return joint

    def createRope(self, prim_path, stage, physicsMaterialPath: Sdf.Path):
        linkLength = 2.0 * self.linkHalfLength - self.linkRadius

        scopePath = prim_path
        base = UsdGeom.Xform.Define(stage, scopePath)
        base.AddTranslateOp().Set(value=self.position)

        z = self.capsuleZ + self.linkRadius
        angle = 0.0
        final_angle = 2 * (np.random.uniform() - 0.5) * 3.14 / 2
        angles = np.linspace(0, final_angle, self.numLinks)
        for _ in range(2):
            mean = np.random.uniform()
            std = np.random.uniform(0.5, 0.8)
            angles *= 1 + np.exp(-((np.linspace(0, 1, self.numLinks) - mean) ** 2) / std**2)

        # angles = angles * 0.0
        x_values = np.cumsum(linkLength * np.cos(angles))
        y_values = np.cumsum(linkLength * np.sin(angles))

        center_x, center_y = np.mean(x_values), np.mean(y_values)
        center_x += np.random.uniform() * 0.05
        center_y += np.random.uniform() * 0.05
        x_values -= center_x
        y_values -= center_y

        angle = 0.0
        i = 0
        for x, y, angle in zip(x_values, y_values, angles):
            capsulePath = scopePath.AppendChild(f"capsule_{i}")
            capsule = self.createCapsule(capsulePath, stage, physicsMaterialPath)
            capsule.AddTranslateOp().Set(value=(x, y, z))
            rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), angle / 3.14 * 180)
            capsule.AddOrientOp().Set(value=Gf.Quatf(rotation.GetQuat()))
            i += 1

        if self.filter_collisions:
            for i in range(self.numLinks - 1):
                rope_prim = stage.GetPrimAtPath(f"{prim_path}/capsule_{i}")
                fp = UsdPhysics.FilteredPairsAPI.Apply(rope_prim)
                rel = fp.CreateFilteredPairsRel()
                a = Sdf.Path(f"{prim_path}/capsule_{i}")
                b = Sdf.Path(f"{prim_path}/capsule_{i+1}")
                targets = [a, b]
                rel.SetTargets(targets)

        jointX = self.linkHalfLength - 0.5 * self.linkRadius
        for linkInd in range(self.numLinks - 1):
            joint = self.createJoint(scopePath.AppendChild(f"joint_{linkInd}"), stage)
            joint.CreateLocalPos0Attr(Gf.Vec3f(jointX, 0, 0))
            joint.CreateLocalPos1Attr(Gf.Vec3f(-jointX, 0, 0))
            joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
            joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
            joint.CreateBody0Rel().SetTargets(
                [Sdf.Path(f"{prim_path}/capsule_{linkInd}")]
            )
            joint.CreateBody1Rel().SetTargets(
                [Sdf.Path(f"{prim_path}/capsule_{linkInd+1}")]
            )

        return prim_utils.get_prim_at_path(scopePath)


"""stage = omni.usd.get_context().get_stage()
dem = RopeFactory(0.4)
dem.capsuleZ = 0.2
#dem.rope_damping = 5
#dem.rope_stiffness = 50
#dem.coneAngleLimit = 50
print(dem.create("/World/Rope", stage))"""