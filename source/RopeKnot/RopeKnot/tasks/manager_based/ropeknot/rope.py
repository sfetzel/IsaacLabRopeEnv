from pxr import UsdGeom, Sdf, Gf, UsdPhysics, UsdShade, PhysxSchema
from omni.physx.scripts import physicsUtils
import numpy as np
import isaacsim.core.utils.prims as prim_utils


class RopeFactory:
    """
    Adapted from Isaac Sim Rope demo.
    """

    def __init__(self, rope_length, position=(0.0, 0.0, 0.0)):
        self.linkHalfLength = 0.05  # smaller value makes it smoother
        self.linkRadius = 0.48 * self.linkHalfLength
        self.ropeLength = rope_length
        self.ropeSpacing = 1.50
        self.coneAngleLimit = 110
        self.rope_damping = 10.0
        self.rope_stiffness = 1.0
        self.position = position
        self.contactOffset = 2.0
        self.capsuleZ = 0.0

    def create(self, prim_path: str, stage):
        self._defaultPrimPath = Sdf.Path(prim_path)

        physicsMaterialPath = self._defaultPrimPath.AppendChild("PhysicsMaterial")
        UsdShade.Material.Define(stage, physicsMaterialPath)
        material = UsdPhysics.MaterialAPI.Apply(
            stage.GetPrimAtPath(physicsMaterialPath)
        )
        material.CreateStaticFrictionAttr().Set(0.5)
        material.CreateDynamicFrictionAttr().Set(0.5)
        material.CreateRestitutionAttr().Set(0)

        self.createRope(self._defaultPrimPath, stage, physicsMaterialPath)

    def createCapsule(self, path: Sdf.Path, stage, physicsMaterialPath: Sdf.Path):
        capsuleGeom = UsdGeom.Capsule.Define(stage, path)
        capsuleGeom.CreateHeightAttr(self.linkHalfLength)
        capsuleGeom.CreateRadiusAttr(self.linkRadius)
        capsuleGeom.CreateAxisAttr("X")

        UsdPhysics.CollisionAPI.Apply(capsuleGeom.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(capsuleGeom.GetPrim())
        massAPI = UsdPhysics.MassAPI.Apply(capsuleGeom.GetPrim())
        massAPI.CreateDensityAttr().Set(0.00005)
        physxCollisionAPI = PhysxSchema.PhysxCollisionAPI.Apply(capsuleGeom.GetPrim())
        physxCollisionAPI.CreateRestOffsetAttr().Set(0.0)
        physxCollisionAPI.CreateContactOffsetAttr().Set(self.contactOffset)
        physicsUtils.add_physics_material_to_prim(
            stage, capsuleGeom.GetPrim(), physicsMaterialPath
        )

    def createJoint(self, jointPath: Sdf.Path, stage):
        joint = UsdPhysics.Joint.Define(stage, jointPath)

        # locked DOF (lock - low is greater than high)
        d6Prim = joint.GetPrim()
        limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transX")
        limitAPI.CreateLowAttr(1.0)
        limitAPI.CreateHighAttr(-1.0)
        limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transY")
        limitAPI.CreateLowAttr(1.0)
        limitAPI.CreateHighAttr(-1.0)
        limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transZ")
        limitAPI.CreateLowAttr(1.0)
        limitAPI.CreateHighAttr(-1.0)
        limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "rotX")
        limitAPI.CreateLowAttr(1.0)
        limitAPI.CreateHighAttr(-1.0)

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

    def createRope(self, prim_path, stage, physicsMaterialPath: Sdf.Path):
        linkLength = 2.0 * self.linkHalfLength - self.linkRadius
        numLinks = int(self.ropeLength / linkLength)

        scopePath = prim_path.AppendChild(f"Rope")
        base = UsdGeom.Xform.Define(stage, scopePath)
        base.AddTranslateOp().Set(value=self.position)

        # capsule instancer
        instancerPath = scopePath.AppendChild("rigidBodyInstancer")
        rboInstancer = UsdGeom.PointInstancer.Define(stage, instancerPath)

        capsulePath = instancerPath.AppendChild("capsule")
        self.createCapsule(capsulePath, stage, physicsMaterialPath)

        meshIndices = []
        positions = []
        orientations = []

        z = self.capsuleZ + self.linkRadius
        angle = 0.0
        final_angle = 2 * (np.random.uniform() - 0.5) * 3.14 / 2
        angles = np.linspace(0, final_angle, numLinks)
        for _ in range(2):
            mean = np.random.uniform()
            std = np.random.uniform(0.5, 0.8)
            angles *= 1 + np.exp(-((np.linspace(0, 1, numLinks) - mean) ** 2) / std**2)

        # angles = angles * 0.0
        x_values = np.cumsum(linkLength * np.cos(angles))
        y_values = np.cumsum(linkLength * np.sin(angles))

        center_x, center_y = np.mean(x_values), np.mean(y_values)
        center_x += np.random.uniform() * 0.05
        center_y += np.random.uniform() * 0.05
        x_values -= center_x
        y_values -= center_y

        angle = 0.0
        for x, y, angle in zip(x_values, y_values, angles):
            meshIndices.append(0)

            positions.append(Gf.Vec3f(x, y, z))
            rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), angle / 3.14 * 180)
            orientations.append(Gf.Quath(rotation.GetQuat()))
        meshList = rboInstancer.GetPrototypesRel()
        # add mesh reference to point instancer
        meshList.AddTarget(capsulePath)

        rboInstancer.GetProtoIndicesAttr().Set(meshIndices)
        rboInstancer.GetPositionsAttr().Set(positions)
        rboInstancer.GetOrientationsAttr().Set(orientations)

        # joint instancer
        jointInstancerPath = scopePath.AppendChild("jointInstancer")
        jointInstancer = PhysxSchema.PhysxPhysicsJointInstancer.Define(
            stage, jointInstancerPath
        )

        jointPath = jointInstancerPath.AppendChild("joint")
        self.createJoint(jointPath, stage)

        meshIndices = []
        body0s = []
        body0indices = []
        localPos0 = []
        localRot0 = []
        body1s = []
        body1indices = []
        localPos1 = []
        localRot1 = []
        body0s.append(instancerPath)
        body1s.append(instancerPath)

        jointX = self.linkHalfLength - 0.5 * self.linkRadius
        for linkInd in range(numLinks - 1):
            meshIndices.append(0)

            body0indices.append(linkInd)
            body1indices.append(linkInd + 1)

            localPos0.append(Gf.Vec3f(jointX, 0, 0))
            localPos1.append(Gf.Vec3f(-jointX, 0, 0))
            localRot0.append(Gf.Quath(1.0))
            localRot1.append(Gf.Quath(1.0))

        meshList = jointInstancer.GetPhysicsPrototypesRel()
        meshList.AddTarget(jointPath)

        jointInstancer.GetPhysicsProtoIndicesAttr().Set(meshIndices)

        jointInstancer.GetPhysicsBody0sRel().SetTargets(body0s)
        jointInstancer.GetPhysicsBody0IndicesAttr().Set(body0indices)
        jointInstancer.GetPhysicsLocalPos0sAttr().Set(localPos0)
        jointInstancer.GetPhysicsLocalRot0sAttr().Set(localRot0)

        jointInstancer.GetPhysicsBody1sRel().SetTargets(body1s)
        jointInstancer.GetPhysicsBody1IndicesAttr().Set(body1indices)
        jointInstancer.GetPhysicsLocalPos1sAttr().Set(localPos1)
        jointInstancer.GetPhysicsLocalRot1sAttr().Set(localRot1)

        return prim_utils.get_prim_at_path(scopePath)


# stage = omni.usd.get_context().get_stage()
# dem = RopeFactory(1.0)
# print(dem.create("/World/Rope", stage))
