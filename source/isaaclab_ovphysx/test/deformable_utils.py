# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test utilities for OVPhysX volume deformables."""

from __future__ import annotations

from isaaclab_physx.sim.spawners.materials import PhysxDeformableBodyMaterialCfg

from pxr import Gf, Sdf, UsdGeom, UsdShade

import isaaclab.sim as sim_utils

_DEFORMABLE_MATERIAL_CFG = PhysxDeformableBodyMaterialCfg(
    dynamic_friction=0.5,
    youngs_modulus=1000.0,
    poissons_ratio=0.3,
)


def _add_api_schemas(prim, schemas: list[str]) -> None:
    """Author applied schemas without requiring the Omni PhysX Kit modules."""
    api_schemas = Sdf.TokenListOp()
    api_schemas.explicitItems = schemas
    prim.SetMetadata("apiSchemas", api_schemas)


@sim_utils.clone
def spawn_pre_tetrahedralized_deformable(
    prim_path: str,
    cfg: sim_utils.SpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
):
    """Spawn a tiny volume TetMesh using USD APIs only."""
    stage = sim_utils.get_current_stage()
    tet_mesh = UsdGeom.TetMesh.Define(stage, prim_path)
    body_prim = tet_mesh.GetPrim()
    _add_api_schemas(
        body_prim,
        [
            "OmniPhysicsDeformableBodyAPI",
            "OmniPhysicsVolumeDeformableSimAPI",
            "OmniPhysicsDeformablePoseAPI:default",
            "PhysicsCollisionAPI",
            "MaterialBindingAPI",
        ],
    )

    points = [
        Gf.Vec3f(0.0, 0.0, 0.0),
        Gf.Vec3f(0.2, 0.0, 0.0),
        Gf.Vec3f(0.0, 0.2, 0.0),
        Gf.Vec3f(0.0, 0.0, 0.2),
        Gf.Vec3f(0.2, 0.2, 0.2),
    ]
    tet_indices = [Gf.Vec4i(0, 1, 2, 3), Gf.Vec4i(1, 2, 3, 4)]
    surface_indices = [
        Gf.Vec3i(0, 2, 1),
        Gf.Vec3i(0, 1, 3),
        Gf.Vec3i(0, 3, 2),
        Gf.Vec3i(1, 2, 4),
        Gf.Vec3i(2, 3, 4),
        Gf.Vec3i(3, 1, 4),
    ]
    body_prim.CreateAttribute("deformablePose:default:omniphysics:points", Sdf.ValueTypeNames.Point3fArray).Set(points)
    body_prim.CreateAttribute("deformablePose:default:omniphysics:purposes", Sdf.ValueTypeNames.TokenArray).Set(
        ["bindPose"]
    )
    body_prim.CreateAttribute("omniphysics:restShapePoints", Sdf.ValueTypeNames.Point3fArray).Set(points)
    body_prim.CreateAttribute("omniphysics:restTetVtxIndices", Sdf.ValueTypeNames.Int4Array).Set(tet_indices)
    body_prim.CreateAttribute("points", Sdf.ValueTypeNames.Point3fArray).Set(points)
    body_prim.CreateAttribute("surfaceFaceVertexIndices", Sdf.ValueTypeNames.Int3Array).Set(surface_indices)
    body_prim.CreateAttribute("tetVertexIndices", Sdf.ValueTypeNames.Int4Array).Set(tet_indices)
    body_prim.CreateAttribute("velocities", Sdf.ValueTypeNames.Vector3fArray).Set([Gf.Vec3f()] * len(points))

    xform = UsdGeom.Xformable(body_prim)
    if translation is not None:
        xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
    if orientation is not None:
        xform.AddOrientOp().Set(Gf.Quatf(orientation[0], orientation[1], orientation[2], orientation[3]))
    xform.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))

    material_path = f"{prim_path}/material"
    material_prim = _DEFORMABLE_MATERIAL_CFG.func(material_path, _DEFORMABLE_MATERIAL_CFG)
    material = UsdShade.Material(material_prim)
    UsdShade.MaterialBindingAPI.Apply(body_prim)
    UsdShade.MaterialBindingAPI(body_prim).Bind(
        material,
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics",
    )
    return body_prim


def pre_tetrahedralized_deformable_spawn_cfg() -> sim_utils.SpawnerCfg:
    """Create the pre-tetrahedralized volume-deformable spawner used by the tests."""
    return sim_utils.SpawnerCfg(func=spawn_pre_tetrahedralized_deformable)
