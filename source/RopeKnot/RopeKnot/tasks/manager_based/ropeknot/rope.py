from pxr import UsdGeom, Sdf, Gf, UsdPhysics, UsdShade, PhysxSchema, Usd
from omni.physx.scripts import physicsUtils
import numpy as np
import isaacsim.core.utils.prims as prim_utils


class RopeFactory:
    """
    Adapted from Isaac Sim Rope demo.
    Creates a rope using a series of rigid body capsules, connected by joints.
    """

    def __init__(self, rope_length, position=(0.0, 0.0, 0.0)):
        """
        Initializes a rope factory.
        :param position: the initial center position of the rope.
        """
        # density in kg/m³
        self.density = 0.1
        # half length of the cylinder of the capsules.
        self.linkHalfLength = 0.015
        # radius of the halfspheres at the ends of the capsule and the cylinder.
        self.linkRadius = self.linkHalfLength
        # length of the rope.
        self.ropeLength = rope_length
        # approximate volume as cylinder.
        self.volume = np.pi * self.linkRadius**2 * rope_length
        self.mass = self.volume * self.density
        self.numLinks = int(self.ropeLength / (2 * self.linkHalfLength))
        self.mass_per_element = self.mass / self.numLinks
        # angle limit for y/z rotation of joints.
        self.coneAngleLimit = 160
        # damping and stiffness for joint DriveAPI.
        self.rope_damping = 10.0
        self.rope_stiffness = 50.0
        self.position = position #(position[0] - self.ropeLength / 2, position[1], position[2])
        self.contactOffset = 0.02
        # the z coodinate of the bottom of the rope.
        self.capsuleZ = 0.0
        self.filter_collisions = False
        self.double_joints = False

    def create(self, prim_path: str, stage):
        """
        Create a rope in the specified stage with the specified path
        :param prim_path: the path for the rope.
        :param stage: the target stage.
        """
        # create the rope in a memory stage to speed up the creation. Additionally, the simulation
        # doesn't need to be paused (the rope could start moving while it's still in the process
        # of being created).
        temp_stage = Usd.Stage.CreateInMemory()
        temp_prim_path = "/Rope"

        with Usd.EditContext(temp_stage):
            ropePrimPath = Sdf.Path(temp_prim_path)

            # physics material for the capsules.
            physicsMaterialPath = ropePrimPath.AppendChild("PhysicsMaterial")
            UsdShade.Material.Define(temp_stage, physicsMaterialPath)
            material = UsdPhysics.MaterialAPI.Apply(
                temp_stage.GetPrimAtPath(physicsMaterialPath)
            )
            material.CreateStaticFrictionAttr().Set(0.2)
            material.CreateDynamicFrictionAttr().Set(0.1)
            material.CreateRestitutionAttr().Set(0)

            self.createRope(ropePrimPath, temp_stage, physicsMaterialPath)

        # copy the rope from the temporary stage to the target stage.
        Sdf.CopySpec(
            temp_stage.GetRootLayer(), temp_prim_path, stage.GetRootLayer(), prim_path
        )

    def createCapsule(
        self, path: Sdf.Path, stage, physicsMaterialPath: Sdf.Path
    ) -> UsdGeom.Capsule:
        """
        Creates a capsule on the specified stage.
        :param path: the path for the created capsule on the stage.
        :param stage: the stage where the capsule should be created.
        :param pyhsicalMaterialPath: the physics material to use.
        """
        capsuleGeom = UsdGeom.Capsule.Define(stage, path)
        capsuleGeom.CreateHeightAttr(2 * self.linkHalfLength)
        capsuleGeom.CreateRadiusAttr(self.linkRadius)
        capsuleGeom.CreateAxisAttr("X")

        UsdPhysics.CollisionAPI.Apply(capsuleGeom.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(capsuleGeom.GetPrim())

        massAPI = UsdPhysics.MassAPI.Apply(capsuleGeom.GetPrim())
        massAPI.CreateMassAttr().Set(self.mass_per_element)
        physxCollisionAPI = PhysxSchema.PhysxCollisionAPI.Apply(capsuleGeom.GetPrim())
        physxCollisionAPI.CreateRestOffsetAttr().Set(0.0)
        physxCollisionAPI.CreateContactOffsetAttr().Set(self.contactOffset)
        physicsUtils.add_physics_material_to_prim(
            stage, capsuleGeom.GetPrim(), physicsMaterialPath
        )
        return capsuleGeom

    def createJoint(self, jointPath: Sdf.Path, stage) -> UsdPhysics.Joint:
        """
        Creates a joint on the specified stage.
        :param jointPath: the path for the joint.
        :param stage: the target stage.
        :return: the created joint.
        """
        joint = UsdPhysics.Joint.Define(stage, jointPath)
        lower_limit = 0.2
        upper_limit = -0.2
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
        limitAPI.CreateHighAttr(lower_limit)

        # Moving DOF:
        dofs = ["rotY", "rotZ"]
        for d in dofs:
            if self.coneAngleLimit is not None:
                limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, d)
                limitAPI.CreateLowAttr(-self.coneAngleLimit)
                limitAPI.CreateHighAttr(self.coneAngleLimit)

            # joint drives for rope dynamics:
            driveAPI = UsdPhysics.DriveAPI.Apply(d6Prim, d)
            driveAPI.CreateTypeAttr("acceleration")
            driveAPI.CreateDampingAttr(self.rope_damping)
            driveAPI.CreateStiffnessAttr(self.rope_stiffness)
        return joint

    def setJointProperties(
        self,
        joint: UsdPhysics.Joint,
        pos0: Gf.Vec3f,
        pos1: Gf.Vec3f,
        prim0: str,
        prim1: str,
    ):
        """
        Set joint position properties and targets
        :param joint: the joint to modify.
        :param pos0: the joint position on object 0.
        :param pos1: the joint position on object 1.
        :param prim0: the stage path to the first prim of the joint.
        :param prim1: the stage path to the second prim of the joint.
        """
        joint.CreateLocalPos0Attr(pos0)
        joint.CreateLocalPos1Attr(pos1)
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
        joint.CreateBody0Rel().SetTargets([Sdf.Path(prim0)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(prim1)])

    def createRope(self, prim_path: Sdf.Path, stage, physicsMaterialPath: Sdf.Path):
        """
        Creates a rope on the specified stage.
        :param prim_path: the path where the rope should be created.
        :param stage: the target stage.
        :param physicsMaterialPath: the physics material for the capsules.
        """
        linkLength = 2.0 * self.linkHalfLength

        # the base Xform containing all capsules and joints.
        base = UsdGeom.Xform.Define(stage, prim_path)
        base.AddTranslateOp().Set(value=self.position)

        z = self.capsuleZ + self.linkRadius
        angles = np.zeros(self.numLinks)


        x_values = np.cumsum(linkLength * np.ones(self.numLinks)) - self.ropeLength / 2.0
        y_values = np.cumsum(linkLength * np.zeros(self.numLinks))


        angle = 0
        i = 0
        for x, y, angle in zip(x_values, y_values, angles):
            capsulePath = prim_path.AppendChild(f"capsule_{i}")
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

        jointX = self.linkHalfLength
        jointY = self.linkRadius * 0.05 if self.double_joints else 0
        for linkInd in range(self.numLinks - 1):
            joint = self.createJoint(prim_path.AppendChild(f"joint_{linkInd}"), stage)
            self.setJointProperties(
                joint,
                Gf.Vec3f(jointX, jointY, 0),
                Gf.Vec3f(-jointX, -jointY, 0),
                f"{prim_path}/capsule_{linkInd}",
                f"{prim_path}/capsule_{linkInd+1}",
            )

            if self.double_joints:
                joint = self.createJoint(
                    prim_path.AppendChild(f"joint_2_{linkInd}"), stage
                )
                self.setJointProperties(
                    joint,
                    Gf.Vec3f(jointX, -jointY, 0),
                    Gf.Vec3f(-jointX, jointY, 0),
                    f"{prim_path}/capsule_{linkInd}",
                    f"{prim_path}/capsule_{linkInd+1}",
                )

        return prim_utils.get_prim_at_path(prim_path)


if __name__ == "__main__":
    import omni

    stage = omni.usd.get_context().get_stage()
    dem = RopeFactory(1.2)
    dem.capsuleZ = 0.0
    print(dem.create("/World/Rope", stage))
