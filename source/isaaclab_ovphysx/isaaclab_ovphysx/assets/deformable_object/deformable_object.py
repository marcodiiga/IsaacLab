# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import re
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from pxr import UsdShade

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets.deformable_object.base_deformable_object import BaseDeformableObject
from isaaclab.cloner import queue_usd_replication
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.warp import ProxyArray

from isaaclab_ovphysx import tensor_types as TT
from isaaclab_ovphysx.physics import OvPhysxManager

from .deformable_object_data import DeformableObjectData
from .kernels import (
    compute_nodal_state_w,
    set_kinematic_flags_to_one,
    vec6f,
    write_nodal_vec3f_to_buffer,
    write_nodal_vec4f_to_buffer,
)
from .views import OvPhysxDeformableBodyView, OvPhysxDeformableMaterialView

if TYPE_CHECKING:
    from isaaclab.assets.deformable_object.deformable_object_cfg import DeformableObjectCfg

logger = logging.getLogger(__name__)

_REQUIRED_DEFORMABLE_TENSOR_TYPES = (
    "DEFORMABLE_SIM_NODAL_POSITION",
    "DEFORMABLE_SIM_NODAL_VELOCITY",
    "DEFORMABLE_SIM_KINEMATIC_TARGET",
    "DEFORMABLE_REST_NODAL_POSITION",
    "DEFORMABLE_SIM_ELEMENT_INDICES",
    "DEFORMABLE_MATERIAL_DYNAMIC_FRICTION",
    "DEFORMABLE_MATERIAL_YOUNGS_MODULUS",
    "DEFORMABLE_MATERIAL_POISSONS_RATIO",
)


def _require_deformable_tensor_types() -> None:
    missing = [name for name in _REQUIRED_DEFORMABLE_TENSOR_TYPES if not hasattr(TT, name)]
    if missing:
        raise RuntimeError(
            "The installed ovphysx wheel does not expose deformable tensor bindings. "
            "Install an ovphysx wheel with deformable TensorType support before using "
            f"isaaclab_ovphysx.assets.DeformableObject. Missing: {', '.join(missing)}."
        )


def _get_api_schemas(prim) -> set[str]:
    """Return authored API schema names, including schemas unknown to USD Python."""
    schemas = set(prim.GetAppliedSchemas())
    api_schemas = prim.GetMetadata("apiSchemas")
    if api_schemas is None:
        return schemas
    if hasattr(api_schemas, "ApplyOperations"):
        schemas.update(api_schemas.ApplyOperations([]))
    for item_name in ("explicitItems", "prependedItems", "appendedItems", "addedItems"):
        schemas.update(getattr(api_schemas, item_name, []) or [])
    return schemas


class DeformableObject(BaseDeformableObject):
    """OVPhysX-backed volume deformable object asset.

    Deformable objects are simulated through nodal positions and velocities
    rather than through a single rigid root pose. OVPhysX v1 exposes volume
    deformables only; surface deformables, collision-mesh tensors, and
    attachment tensors are not part of this adapter.

    Volume deformables can be partially kinematic. Kinematic targets store the
    desired nodal position and a flag where ``0.0`` marks a kinematically driven
    node and ``1.0`` marks a free simulated node.
    """

    cfg: DeformableObjectCfg
    __backend_name__: str = "ovphysx"

    def __init__(self, cfg: DeformableObjectCfg):
        super().__init__(cfg)
        queue_usd_replication(cfg)
        OvPhysxManager.require_full_usd_export()
        self._DTYPE_TO_TORCH_TRAILING_DIMS = {**self._DTYPE_TO_TORCH_TRAILING_DIMS, vec6f: (6,)}
        self._deformable_type: str | None = None

    @property
    def data(self) -> DeformableObjectData:
        """Data container for the deformable object."""
        return self._data

    @property
    def num_instances(self) -> int:
        """Number of deformable object instances matched by the asset."""
        return self.root_view.count

    @property
    def num_bodies(self) -> int:
        """Number of bodies in the asset.

        This is always 1 since each object is a single deformable body.
        """
        return 1

    @property
    def root_view(self) -> OvPhysxDeformableBodyView:
        """Deformable body view for direct OVPhysX tensor access."""
        return self._root_physx_view

    @property
    def root_physx_view(self) -> OvPhysxDeformableBodyView:
        """Deprecated property. Please use :attr:`root_view` instead."""
        logger.warning(
            "The `root_physx_view` property will be deprecated in a future release. Please use `root_view` instead."
        )
        return self.root_view

    @property
    def material_physx_view(self) -> OvPhysxDeformableMaterialView | None:
        """Optional deformable material view for runtime material properties."""
        return self._material_physx_view

    @property
    def max_sim_elements_per_body(self) -> int:
        """Maximum number of simulation mesh elements per deformable body."""
        return self.root_view.max_simulation_elements_per_body

    @property
    def max_collision_elements_per_body(self) -> int:
        """Maximum number of collision mesh elements per deformable body.

        OVPhysX v1 does not expose collision mesh tensors, so this is 0.
        """
        return self.root_view.max_collision_elements_per_body

    @property
    def max_sim_vertices_per_body(self) -> int:
        """Maximum number of simulation mesh vertices per deformable body."""
        return self.root_view.max_simulation_nodes_per_body

    @property
    def max_collision_vertices_per_body(self) -> int:
        """Maximum number of collision mesh vertices per deformable body.

        OVPhysX v1 does not expose collision mesh tensors, so this is 0.
        """
        return self.root_view.max_collision_nodes_per_body

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None) -> None:
        """Reset the deformable object.

        Args:
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. If None, then all instances are used.
        """
        pass

    def write_data_to_sim(self) -> None:
        """Write pending data to the simulator."""
        pass

    def update(self, dt: float) -> None:
        """Update cached deformable data timestamps."""
        self._data.update(dt)

    def write_nodal_state_to_sim_index(
        self,
        nodal_state: torch.Tensor | wp.array | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set nodal positions and velocities over selected environment indices.

        Args:
            nodal_state: Nodal state in simulation frame [m, m/s].
                Shape is ``(len(env_ids), max_sim_vertices_per_body, 6)`` or
                ``(num_instances, max_sim_vertices_per_body, 6)`` when
                ``full_data`` is true.
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether ``nodal_state`` contains all instances.
        """
        if isinstance(nodal_state, ProxyArray):
            nodal_state = nodal_state.torch
        elif isinstance(nodal_state, wp.array):
            nodal_state = wp.to_torch(nodal_state)
        self.write_nodal_pos_to_sim_index(nodal_state[..., :3], env_ids=env_ids, full_data=full_data)
        self.write_nodal_velocity_to_sim_index(nodal_state[..., 3:], env_ids=env_ids, full_data=full_data)

    def write_nodal_state_to_sim_mask(
        self,
        nodal_state: torch.Tensor | wp.array | ProxyArray,
        env_mask: torch.Tensor | wp.array | ProxyArray | None = None,
    ) -> None:
        """Set nodal positions and velocities over selected environment mask."""
        env_ids = self._resolve_env_mask_ids(env_mask)
        self.write_nodal_state_to_sim_index(nodal_state, env_ids=env_ids, full_data=True)

    def write_nodal_pos_to_sim_index(
        self,
        nodal_pos: torch.Tensor | wp.array | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set nodal positions over selected environment indices."""
        env_ids = self._resolve_env_ids(env_ids)
        if isinstance(nodal_pos, ProxyArray):
            nodal_pos = nodal_pos.warp
        if full_data:
            self.assert_shape_and_dtype(
                nodal_pos, (self.num_instances, self.max_sim_vertices_per_body), wp.vec3f, "nodal_pos"
            )
        else:
            self.assert_shape_and_dtype(
                nodal_pos, (env_ids.shape[0], self.max_sim_vertices_per_body), wp.vec3f, "nodal_pos"
            )
        if isinstance(nodal_pos, torch.Tensor):
            nodal_pos = wp.from_torch(nodal_pos.contiguous(), dtype=wp.vec3f)
        wp.launch(
            write_nodal_vec3f_to_buffer,
            dim=(env_ids.shape[0], self.max_sim_vertices_per_body),
            inputs=[nodal_pos, env_ids, full_data],
            outputs=[self._data._nodal_pos_w.data],
            device=self.device,
        )
        self._data._nodal_pos_w.timestamp = self._data._sim_timestamp
        self._data._nodal_state_w.timestamp = -1.0
        self._data._root_pos_w.timestamp = -1.0
        self.root_view.set_simulation_nodal_positions(self._get_nodal_pos_w_f32(), indices=env_ids)

    def write_nodal_pos_to_sim_mask(
        self,
        nodal_pos: torch.Tensor | wp.array | ProxyArray,
        env_mask: torch.Tensor | wp.array | ProxyArray | None = None,
    ) -> None:
        """Set nodal positions over selected environment mask."""
        env_ids = self._resolve_env_mask_ids(env_mask)
        self.write_nodal_pos_to_sim_index(nodal_pos, env_ids=env_ids, full_data=True)

    def write_nodal_velocity_to_sim_index(
        self,
        nodal_vel: torch.Tensor | wp.array | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set nodal velocities over selected environment indices."""
        env_ids = self._resolve_env_ids(env_ids)
        if isinstance(nodal_vel, ProxyArray):
            nodal_vel = nodal_vel.warp
        if full_data:
            self.assert_shape_and_dtype(
                nodal_vel, (self.num_instances, self.max_sim_vertices_per_body), wp.vec3f, "nodal_vel"
            )
        else:
            self.assert_shape_and_dtype(
                nodal_vel, (env_ids.shape[0], self.max_sim_vertices_per_body), wp.vec3f, "nodal_vel"
            )
        if isinstance(nodal_vel, torch.Tensor):
            nodal_vel = wp.from_torch(nodal_vel.contiguous(), dtype=wp.vec3f)
        wp.launch(
            write_nodal_vec3f_to_buffer,
            dim=(env_ids.shape[0], self.max_sim_vertices_per_body),
            inputs=[nodal_vel, env_ids, full_data],
            outputs=[self._data._nodal_vel_w.data],
            device=self.device,
        )
        self._data._nodal_vel_w.timestamp = self._data._sim_timestamp
        self._data._nodal_state_w.timestamp = -1.0
        self._data._root_vel_w.timestamp = -1.0
        self.root_view.set_simulation_nodal_velocities(self._get_nodal_vel_w_f32(), indices=env_ids)

    def write_nodal_velocity_to_sim_mask(
        self,
        nodal_vel: torch.Tensor | wp.array | ProxyArray,
        env_mask: torch.Tensor | wp.array | ProxyArray | None = None,
    ) -> None:
        """Set nodal velocities over selected environment mask."""
        env_ids = self._resolve_env_mask_ids(env_mask)
        self.write_nodal_velocity_to_sim_index(nodal_vel, env_ids=env_ids, full_data=True)

    def write_nodal_kinematic_target_to_sim_index(
        self,
        targets: torch.Tensor | wp.array | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set kinematic targets over selected environment indices."""
        if self._deformable_type != "volume":
            raise ValueError("Kinematic targets can only be set for volume deformable bodies.")

        env_ids = self._resolve_env_ids(env_ids)
        if isinstance(targets, ProxyArray):
            targets = targets.warp
        if full_data:
            self.assert_shape_and_dtype(
                targets, (self.num_instances, self.max_sim_vertices_per_body), wp.vec4f, "targets"
            )
        else:
            self.assert_shape_and_dtype(
                targets, (env_ids.shape[0], self.max_sim_vertices_per_body), wp.vec4f, "targets"
            )
        if isinstance(targets, torch.Tensor):
            if targets.dim() == 2:
                targets = targets.unsqueeze(0)
            targets = wp.from_torch(targets.contiguous(), dtype=wp.vec4f)
        wp.launch(
            write_nodal_vec4f_to_buffer,
            dim=(env_ids.shape[0], self.max_sim_vertices_per_body),
            inputs=[targets, env_ids, full_data],
            outputs=[self._data.nodal_kinematic_target.warp],
            device=self.device,
        )
        self.root_view.set_simulation_nodal_kinematic_targets(
            self._data.nodal_kinematic_target.warp.view(wp.float32), indices=env_ids
        )

    def write_nodal_kinematic_target_to_sim_mask(
        self,
        targets: torch.Tensor | wp.array | ProxyArray,
        env_mask: torch.Tensor | wp.array | ProxyArray | None = None,
    ) -> None:
        """Set kinematic targets over selected environment mask."""
        env_ids = self._resolve_env_mask_ids(env_mask)
        self.write_nodal_kinematic_target_to_sim_index(targets, env_ids=env_ids, full_data=True)

    def write_nodal_state_to_sim(
        self,
        nodal_state: torch.Tensor | wp.array | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated. Please use :meth:`write_nodal_state_to_sim_index` instead."""
        warnings.warn(
            "The method 'write_nodal_state_to_sim' is deprecated. Please use 'write_nodal_state_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_nodal_state_to_sim_index(nodal_state, env_ids=env_ids)

    def write_nodal_kinematic_target_to_sim(
        self,
        targets: torch.Tensor | wp.array | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated. Please use :meth:`write_nodal_kinematic_target_to_sim_index` instead."""
        warnings.warn(
            "The method 'write_nodal_kinematic_target_to_sim' is deprecated."
            " Please use 'write_nodal_kinematic_target_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_nodal_kinematic_target_to_sim_index(targets, env_ids=env_ids)

    def write_nodal_pos_to_sim(
        self,
        nodal_pos: torch.Tensor | wp.array | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated. Please use :meth:`write_nodal_pos_to_sim_index` instead."""
        warnings.warn(
            "The method 'write_nodal_pos_to_sim' is deprecated. Please use 'write_nodal_pos_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_nodal_pos_to_sim_index(nodal_pos, env_ids=env_ids)

    def write_nodal_velocity_to_sim(
        self,
        nodal_vel: torch.Tensor | wp.array | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated. Please use :meth:`write_nodal_velocity_to_sim_index` instead."""
        warnings.warn(
            "The method 'write_nodal_velocity_to_sim' is deprecated."
            " Please use 'write_nodal_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_nodal_velocity_to_sim_index(nodal_vel, env_ids=env_ids)

    def transform_nodal_pos(
        self, nodal_pos: torch.Tensor, pos: torch.Tensor | None = None, quat: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Transform nodal positions around each body's current nodal center.

        Args:
            nodal_pos: Nodal positions in simulation frame [m].
            pos: Translation to apply [m].
            quat: Rotation to apply as ``(x, y, z, w)``.

        Returns:
            Transformed nodal positions [m].
        """
        mean_nodal_pos = nodal_pos.mean(dim=1, keepdim=True)
        nodal_pos = nodal_pos - mean_nodal_pos
        return math_utils.transform_points(nodal_pos, pos, quat) + mean_nodal_pos

    def _initialize_impl(self) -> None:
        _require_deformable_tensor_types()

        physx_instance = OvPhysxManager.get_physx_instance()
        if physx_instance is None:
            raise RuntimeError("OvPhysxManager has not been initialized yet.")
        self._ovphysx = physx_instance
        self._device = OvPhysxManager.get_device()

        template_prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if template_prim is None:
            raise RuntimeError(f"Failed to find prim for expression: '{self.cfg.prim_path}'.")
        template_prim_path = template_prim.GetPath().pathString

        root_prims = sim_utils.get_all_matching_child_prims(
            template_prim_path,
            predicate=lambda prim: "OmniPhysicsDeformableBodyAPI" in _get_api_schemas(prim),
            traverse_instance_prims=False,
        )
        if len(root_prims) == 0:
            raise RuntimeError(
                f"Failed to find a deformable body when resolving '{self.cfg.prim_path}'."
                " Please ensure that the prim has 'OmniPhysicsDeformableBodyAPI' applied."
            )
        if len(root_prims) > 1:
            raise RuntimeError(
                f"Failed to find a single deformable body when resolving '{self.cfg.prim_path}'."
                f" Found multiple '{root_prims}' under '{template_prim_path}'."
                " Please ensure that there is only one deformable body in the prim path tree."
            )
        root_prim = root_prims[0]

        material_prim = self._find_deformable_material(root_prim)
        if material_prim is None:
            logger.warning(
                "Failed to find a deformable material binding for '%s'. Material properties will use defaults.",
                root_prim.GetPath().pathString,
            )
        self._deformable_type = self._detect_deformable_type(root_prim, material_prim)
        if self._deformable_type == "surface":
            raise NotImplementedError("OVPhysX deformable object supports volume deformables only.")
        if self._deformable_type != "volume":
            raise RuntimeError(
                f"Failed to determine deformable material type for '{root_prim.GetPath().pathString}'."
                " Please ensure that the material has 'PhysxDeformableMaterialAPI' applied, or that a valid "
                "TetMesh is found under the root prim."
            )

        root_prim_path = root_prim.GetPath().pathString
        root_prim_path_expr = self.cfg.prim_path + root_prim_path[len(template_prim_path) :]
        root_pattern = self._to_ovphysx_pattern(root_prim_path_expr)
        self._root_physx_view = OvPhysxDeformableBodyView(physx_instance, root_pattern, self._device)
        if not self._root_physx_view.check():
            raise RuntimeError(f"Failed to create deformable body at: {self.cfg.prim_path}. Please check PhysX logs.")

        if material_prim is not None:
            material_prim_path = material_prim.GetPath().pathString
            if material_prim_path.startswith(template_prim_path):
                material_prim_path_expr = self.cfg.prim_path + material_prim_path[len(template_prim_path) :]
            else:
                material_prim_path_expr = material_prim_path
            material_pattern = self._to_ovphysx_pattern(material_prim_path_expr)
            self._material_physx_view = OvPhysxDeformableMaterialView(physx_instance, material_pattern)
        else:
            self._material_physx_view = None

        logger.info("Deformable body initialized at: %s", root_pattern)
        logger.info("Number of instances: %s", self.num_instances)
        if self._material_physx_view is not None:
            logger.info("Deformable material initialized with %s instance(s)", self._material_physx_view.count)
        else:
            logger.info("No deformable material found. Material properties will be set to default values.")

        self._data = DeformableObjectData(self.root_view, self.device)
        self._create_buffers()
        self.update(0.0)

        if self._debug_vis_handle is None:
            self.set_debug_vis(self.cfg.debug_vis)

    def _create_buffers(self) -> None:
        """Create buffers for storing data."""
        self._ALL_INDICES = wp.array(np.arange(self.num_instances, dtype=np.int32), device=self.device)
        self._nodal_pos_w_f32: wp.array | None = None
        self._nodal_vel_w_f32: wp.array | None = None

        nodal_positions_raw = self.root_view.get_simulation_nodal_positions()
        nodal_positions = nodal_positions_raw.view(wp.vec3f).reshape(
            (self.num_instances, self.max_sim_vertices_per_body)
        )
        nodal_velocities = wp.zeros(
            (self.num_instances, self.max_sim_vertices_per_body), dtype=wp.vec3f, device=self.device
        )
        default_nodal_state = wp.zeros(
            (self.num_instances, self.max_sim_vertices_per_body), dtype=vec6f, device=self.device
        )
        wp.launch(
            compute_nodal_state_w,
            dim=(self.num_instances, self.max_sim_vertices_per_body),
            inputs=[nodal_positions, nodal_velocities],
            outputs=[default_nodal_state],
            device=self.device,
        )
        self._data.default_nodal_state_w = ProxyArray(default_nodal_state)

        kinematic_raw = self.root_view.get_simulation_nodal_kinematic_targets()
        kinematic_view = kinematic_raw.view(wp.vec4f).reshape((self.num_instances, self.max_sim_vertices_per_body))
        kinematic_target = wp.zeros(
            (self.num_instances, self.max_sim_vertices_per_body), dtype=wp.vec4f, device=self.device
        )
        wp.copy(kinematic_target, kinematic_view)
        wp.launch(
            set_kinematic_flags_to_one,
            dim=(self.num_instances * self.max_sim_vertices_per_body,),
            inputs=[kinematic_target.reshape((self.num_instances * self.max_sim_vertices_per_body,))],
            device=self.device,
        )
        self._data.nodal_kinematic_target = ProxyArray(kinematic_target)
        self.root_view.set_simulation_nodal_kinematic_targets(kinematic_target.view(wp.float32))

    def _find_deformable_material(self, root_prim):
        if not root_prim.HasAPI(UsdShade.MaterialBindingAPI):
            return None
        material_paths = UsdShade.MaterialBindingAPI(root_prim).GetDirectBindingRel("physics").GetTargets()
        for mat_path in material_paths:
            mat_prim = root_prim.GetStage().GetPrimAtPath(mat_path)
            if "OmniPhysicsDeformableMaterialAPI" in _get_api_schemas(mat_prim):
                return mat_prim
        return None

    def _detect_deformable_type(self, root_prim, material_prim) -> str | None:
        if material_prim is not None:
            applied = _get_api_schemas(material_prim)
            if "PhysxSurfaceDeformableMaterialAPI" in applied:
                return "surface"
            if "PhysxDeformableMaterialAPI" in applied:
                return "volume"
        if len(sim_utils.get_all_matching_child_prims(root_prim.GetPath(), lambda p: p.GetTypeName() == "TetMesh")) > 0:
            return "volume"
        if len(sim_utils.get_all_matching_child_prims(root_prim.GetPath(), lambda p: p.GetTypeName() == "Mesh")) > 0:
            return "surface"
        return None

    def _resolve_env_ids(self, env_ids):
        """Resolve environment indices to a warp array on the simulation device."""
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            return self._ALL_INDICES
        if isinstance(env_ids, torch.Tensor):
            return wp.from_torch(env_ids.to(device=self.device, dtype=torch.int32), dtype=wp.int32)
        if isinstance(env_ids, list):
            return wp.array(env_ids, dtype=wp.int32, device=self.device)
        if isinstance(env_ids, wp.array) and str(env_ids.device) != self.device:
            return wp.clone(env_ids, device=self.device)
        return env_ids

    def _resolve_env_mask_ids(self, env_mask):
        """Resolve an environment mask to selected environment indices on the simulation device."""
        if env_mask is None:
            return self._ALL_INDICES
        if isinstance(env_mask, ProxyArray):
            env_mask = env_mask.torch
        elif isinstance(env_mask, wp.array):
            env_mask = wp.to_torch(env_mask)
        if isinstance(env_mask, torch.Tensor):
            env_mask = env_mask.to(device=self.device, dtype=torch.bool)
            env_ids = torch.nonzero(env_mask, as_tuple=False).flatten().to(dtype=torch.int32).contiguous()
            return wp.from_torch(env_ids, dtype=wp.int32)
        raise TypeError(f"env_mask must be a torch tensor, warp array, ProxyArray, or None, got {type(env_mask)}")

    def _get_nodal_pos_w_f32(self) -> wp.array:
        if self._nodal_pos_w_f32 is None:
            self._nodal_pos_w_f32 = self._data._nodal_pos_w.data.view(wp.float32)
        return self._nodal_pos_w_f32

    def _get_nodal_vel_w_f32(self) -> wp.array:
        if self._nodal_vel_w_f32 is None:
            self._nodal_vel_w_f32 = self._data._nodal_vel_w.data.view(wp.float32)
        return self._nodal_vel_w_f32

    def _to_ovphysx_pattern(self, path_expr: str) -> str:
        pattern = re.sub(r"\{ENV_REGEX_NS\}", "*", path_expr)
        return re.sub(r"\.\*", "*", pattern)

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            if not hasattr(self, "target_visualizer"):
                self.target_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            self.target_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_visualizer"):
                self.target_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        kinematic_target_torch = self.data.nodal_kinematic_target.torch
        targets_enabled = kinematic_target_torch[:, :, 3] == 0.0
        num_enabled = int(torch.sum(targets_enabled).item())
        if num_enabled == 0:
            positions = torch.tensor([[0.0, 0.0, -10.0]], device=self.device)
        else:
            positions = kinematic_target_torch[targets_enabled][..., :3]
        self.target_visualizer.visualize(positions)

    def _invalidate_initialize_callback(self, event) -> None:
        super()._invalidate_initialize_callback(event)
        if getattr(self, "_root_physx_view", None) is not None:
            try:
                self._root_physx_view.destroy()
            except Exception:
                logger.warning("Failed to destroy OVPhysX deformable body view.", exc_info=True)
        if getattr(self, "_material_physx_view", None) is not None:
            try:
                self._material_physx_view.destroy()
            except Exception:
                logger.warning("Failed to destroy OVPhysX deformable material view.", exc_info=True)
        self._root_physx_view = None
        self._material_physx_view = None
