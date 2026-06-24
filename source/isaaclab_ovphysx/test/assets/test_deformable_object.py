# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Real-backend tests for the OVPhysX DeformableObject."""

from __future__ import annotations

import sys

import pytest
import torch
import warp as wp

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ovphysx.physics import OvPhysxCfg  # noqa: E402
from isaaclab_physx.sim.schemas import PhysxCollisionPropertiesCfg, PhysxRigidBodyPropertiesCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import DeformableObject, DeformableObjectCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, build_simulation_context  # noqa: E402
from isaaclab.utils.configclass import configclass  # noqa: E402

from ..deformable_utils import pre_tetrahedralized_deformable_spawn_cfg  # noqa: E402

wp.init()


def _deformable_spawn_cfg() -> sim_utils.SpawnerCfg:
    """Create the pre-tetrahedralized volume-deformable spawner used by the tests."""
    return pre_tetrahedralized_deformable_spawn_cfg()


@configclass
class DeformableSceneCfg(InteractiveSceneCfg):
    """InteractiveScene configuration for one cloned deformable asset."""

    deformable: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=_deformable_spawn_cfg(),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )


@configclass
class MixedDeformableRigidSceneCfg(InteractiveSceneCfg):
    """InteractiveScene configuration for cloned deformable and rigid assets."""

    deformable: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=_deformable_spawn_cfg(),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=PhysxRigidBodyPropertiesCfg(disable_gravity=True),
            collision_props=PhysxCollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, 0.0, 1.0)),
    )


def _ovphysx_sim_context(device: str, **kwargs):
    """Build a kitless OVPhysX simulation context."""
    dt = kwargs.pop("dt", 1.0 / 60.0)
    gravity_enabled = kwargs.pop("gravity_enabled", True)
    gravity = (0.0, 0.0, -9.81) if gravity_enabled else (0.0, 0.0, 0.0)
    sim_cfg = SimulationCfg(physics=OvPhysxCfg(), device=device, dt=dt, gravity=gravity)
    return build_simulation_context(device=device, sim_cfg=sim_cfg, **kwargs)


def generate_deformables_scene(num_objects: int = 2, height: float = 1.0, device: str = "cuda:0") -> DeformableObject:
    """Generate a small procedural volume-deformable scene."""
    origins = torch.tensor([(i * 0.5, 0.0, height) for i in range(num_objects)], device=device)
    for i, origin in enumerate(origins):
        sim_utils.create_prim(f"/World/Table_{i}", "Xform", translation=origin)

    cfg = DeformableObjectCfg(
        prim_path="/World/Table_.*/Object",
        spawn=_deformable_spawn_cfg(),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    return DeformableObject(cfg=cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OVPhysX deformables require CUDA")
@pytest.mark.isaacsim_ci
def test_volume_deformable_read_write_and_kinematic_targets():
    """Create volume deformables, read/write nodal state, and apply kinematic targets."""
    with _ovphysx_sim_context(device="cuda:0", auto_add_lighting=True, gravity_enabled=False) as sim:
        deformable = generate_deformables_scene(num_objects=2)

        assert sys.getrefcount(deformable) < 10

        sim.reset()

        assert deformable.is_initialized
        assert deformable.num_instances == 2
        assert deformable.num_bodies == 1
        assert deformable.root_view.count == 2
        assert deformable.max_sim_vertices_per_body > 0
        assert deformable.max_sim_elements_per_body > 0
        assert deformable.material_physx_view is not None
        assert deformable.material_physx_view.count == 2

        assert deformable.data.nodal_state_w.torch.shape == (2, deformable.max_sim_vertices_per_body, 6)
        assert deformable.data.nodal_kinematic_target.torch.shape == (2, deformable.max_sim_vertices_per_body, 4)
        assert deformable.data.root_pos_w.torch.shape == (2, 3)
        assert deformable.data.root_vel_w.torch.shape == (2, 3)

        rest_pos = wp.to_torch(deformable.root_view.get_rest_nodal_positions()).reshape(
            2, deformable.max_sim_vertices_per_body, 3
        )
        torch.testing.assert_close(rest_pos, deformable.data.default_nodal_state_w.torch[..., :3], rtol=1e-5, atol=1e-5)
        element_indices = wp.to_torch(deformable.root_view.get_simulation_element_indices())
        assert element_indices.shape == (2, deformable.max_sim_elements_per_body, 4)
        assert element_indices.dtype == torch.int32
        assert torch.all(element_indices >= 0)
        assert torch.all(element_indices < deformable.max_sim_vertices_per_body)

        reset_state = deformable.data.default_nodal_state_w.torch.clone()
        reset_translation = torch.tensor([[0.0, 0.0, 0.02], [0.0, 0.0, 0.04]], device=sim.device)
        reset_state[..., :3] = deformable.transform_nodal_pos(reset_state[..., :3], pos=reset_translation)
        deformable.write_nodal_state_to_sim_index(reset_state)
        torch.testing.assert_close(
            wp.to_torch(deformable.root_view.get_simulation_nodal_positions()),
            reset_state[..., :3],
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            wp.to_torch(deformable.root_view.get_simulation_nodal_velocities()),
            reset_state[..., 3:],
            rtol=1e-5,
            atol=1e-5,
        )

        initial_pos = deformable.data.nodal_pos_w.torch.clone()
        updated_pos = initial_pos.clone()
        updated_pos[1, :, 2] += 0.05
        deformable.write_nodal_pos_to_sim_index(updated_pos[1:2], env_ids=torch.tensor([1], device=sim.device))
        readback_pos = wp.to_torch(deformable.root_view.get_simulation_nodal_positions())
        torch.testing.assert_close(readback_pos[0], initial_pos[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(readback_pos[1], updated_pos[1], rtol=1e-5, atol=1e-5)

        cpu_id_pos = readback_pos.clone()
        cpu_id_pos[0, :, 0] += 0.025
        deformable.write_nodal_pos_to_sim_index(cpu_id_pos[0:1], env_ids=torch.tensor([0]))
        readback_pos = wp.to_torch(deformable.root_view.get_simulation_nodal_positions())
        torch.testing.assert_close(readback_pos[0], cpu_id_pos[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(readback_pos[1], updated_pos[1], rtol=1e-5, atol=1e-5)

        mask_pos = readback_pos.clone()
        mask_pos[1, :, 1] += 0.035
        deformable.write_nodal_pos_to_sim_mask(
            mask_pos,
            env_mask=wp.array([False, True], dtype=wp.bool, device=sim.device),
        )
        readback_pos = wp.to_torch(deformable.root_view.get_simulation_nodal_positions())
        torch.testing.assert_close(readback_pos[0], cpu_id_pos[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(readback_pos[1], mask_pos[1], rtol=1e-5, atol=1e-5)

        direct_vec_pos = deformable.data.nodal_pos_w.warp
        deformable.root_view.set_simulation_nodal_positions(direct_vec_pos)
        torch.testing.assert_close(
            wp.to_torch(deformable.root_view.get_simulation_nodal_positions()),
            deformable.data.nodal_pos_w.torch,
            rtol=1e-5,
            atol=1e-5,
        )
        with pytest.raises(ValueError, match="shape mismatch"):
            deformable.root_view.set_simulation_nodal_positions(
                torch.zeros((deformable.max_sim_vertices_per_body, deformable.num_instances, 3), device=sim.device)
            )

        updated_vel = torch.zeros_like(deformable.data.nodal_vel_w.torch)
        updated_vel[0, :, 0] = 0.1
        deformable.write_nodal_velocity_to_sim_index(updated_vel)
        readback_vel = wp.to_torch(deformable.root_view.get_simulation_nodal_velocities())
        torch.testing.assert_close(readback_vel, updated_vel, rtol=1e-5, atol=1e-5)

        targets = deformable.data.nodal_kinematic_target.torch.clone()
        targets[:, :, 3] = 1.0
        targets[0, :, :3] = readback_pos[0] + torch.tensor([0.0, 0.0, 0.03], device=sim.device)
        targets[0, :, 3] = 0.0
        deformable.write_nodal_kinematic_target_to_sim_index(targets[0:1], env_ids=torch.tensor([0], device=sim.device))
        readback_targets = wp.to_torch(deformable.root_view.get_simulation_nodal_kinematic_targets())
        torch.testing.assert_close(readback_targets[0], targets[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(readback_targets[1, :, 3], torch.ones_like(readback_targets[1, :, 3]))

        mask_targets = readback_targets.clone()
        mask_targets[1, :, :3] = readback_pos[1] + torch.tensor([0.0, 0.02, 0.0], device=sim.device)
        mask_targets[1, :, 3] = 0.0
        deformable.write_nodal_kinematic_target_to_sim_mask(
            mask_targets,
            env_mask=wp.array([False, True], dtype=wp.bool, device=sim.device),
        )
        readback_targets = wp.to_torch(deformable.root_view.get_simulation_nodal_kinematic_targets())
        torch.testing.assert_close(readback_targets[0], targets[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(readback_targets[1], mask_targets[1], rtol=1e-5, atol=1e-5)

        friction = deformable.material_physx_view.get_dynamic_frictions()
        youngs = deformable.material_physx_view.get_youngs_moduli()
        poisson = deformable.material_physx_view.get_poissons_ratios()
        torch.testing.assert_close(wp.to_torch(friction), torch.full((2,), 0.5), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(wp.to_torch(youngs), torch.full((2,), 1000.0), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(wp.to_torch(poisson), torch.full((2,), 0.3), rtol=1e-5, atol=1e-5)

        deformable.material_physx_view.set_youngs_moduli(torch.tensor([1200.0, 2400.0], device=sim.device))
        torch.testing.assert_close(
            wp.to_torch(deformable.material_physx_view.get_youngs_moduli()),
            torch.tensor([1200.0, 2400.0]),
            rtol=1e-5,
            atol=1e-5,
        )
        deformable.material_physx_view.set_dynamic_frictions(
            torch.tensor([0.75, 0.25]),
            mask=torch.tensor([True, False]),
        )
        torch.testing.assert_close(
            wp.to_torch(deformable.material_physx_view.get_dynamic_frictions()),
            torch.tensor([0.75, 0.5]),
            rtol=1e-5,
            atol=1e-5,
        )

        pre_step_pos = deformable.data.nodal_pos_w.torch.clone()
        for _ in range(5):
            sim.step()
            deformable.update(sim.cfg.dt)
        post_step_pos = deformable.data.nodal_pos_w.torch
        assert torch.mean(post_step_pos[0, :, 2] - pre_step_pos[0, :, 2]) > 0.02
        deformable.update(sim.cfg.dt)
        assert torch.isfinite(deformable.data.nodal_pos_w.torch).all()

        root_view = deformable.root_view
        material_view = deformable.material_physx_view
        deformable._invalidate_initialize_callback(None)
        assert deformable._root_physx_view is None
        assert deformable._material_physx_view is None
        assert root_view._bindings == {}
        assert material_view._bindings == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OVPhysX deformables require CUDA")
@pytest.mark.isaacsim_ci
def test_volume_deformable_interactive_scene_replication():
    """Create cloned volume deformables through the normal InteractiveScene path."""
    with _ovphysx_sim_context(device="cuda:0", auto_add_lighting=True, gravity_enabled=False) as sim:
        scene = InteractiveScene(DeformableSceneCfg(num_envs=3, env_spacing=0.75, lazy_sensor_update=False))

        sim.reset()

        deformable = scene["deformable"]
        assert deformable.is_initialized
        assert deformable.num_instances == 3
        assert deformable.root_view.count == 3
        assert deformable.material_physx_view is not None
        assert deformable.material_physx_view.count == 3

        nodal_pos = deformable.data.nodal_pos_w.torch
        assert nodal_pos.shape == (3, deformable.max_sim_vertices_per_body, 3)
        assert torch.isfinite(nodal_pos).all()

        element_indices = wp.to_torch(deformable.root_view.get_simulation_element_indices())
        assert element_indices.shape == (3, deformable.max_sim_elements_per_body, 4)
        assert element_indices.dtype == torch.int32

        base_pos = deformable.data.nodal_pos_w.torch.clone()
        env_ids = torch.tensor([2, 0], device=sim.device)
        permuted_pos = base_pos[env_ids].clone()
        permuted_pos[0, :, 0] += 0.07
        permuted_pos[1, :, 1] -= 0.04
        deformable.write_nodal_pos_to_sim_index(permuted_pos, env_ids=env_ids)

        readback_pos = wp.to_torch(deformable.root_view.get_simulation_nodal_positions())
        torch.testing.assert_close(readback_pos[0], permuted_pos[1], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(readback_pos[1], base_pos[1], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(readback_pos[2], permuted_pos[0], rtol=1e-5, atol=1e-5)

        sim.step()
        scene.update(sim.cfg.dt)
        assert torch.isfinite(deformable.data.nodal_pos_w.torch).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OVPhysX deformables require CUDA")
@pytest.mark.isaacsim_ci
def test_volume_deformable_mixed_scene_full_usd_fallback():
    """Create cloned deformables with a rigid asset while full USD export is active."""
    with _ovphysx_sim_context(device="cuda:0", auto_add_lighting=True, gravity_enabled=False) as sim:
        scene = InteractiveScene(MixedDeformableRigidSceneCfg(num_envs=3, env_spacing=0.75, lazy_sensor_update=False))

        sim.reset()

        deformable = scene["deformable"]
        cube = scene["cube"]
        assert deformable.num_instances == 3
        assert cube.num_instances == 3

        sim.step()
        scene.update(sim.cfg.dt)
        assert torch.isfinite(deformable.data.nodal_pos_w.torch).all()
        assert torch.isfinite(cube.data.root_pos_w.torch).all()
