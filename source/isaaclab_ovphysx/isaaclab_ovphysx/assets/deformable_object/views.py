# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import warp as wp

from isaaclab_ovphysx import tensor_types as TT


def _float32_lane_count(dtype: type) -> int | None:
    if dtype == wp.float32:
        return 1
    if dtype == wp.vec3f:
        return 3
    if dtype == wp.vec4f:
        return 4
    return None


def _as_float32_binding_shape(values: wp.array, shape: Sequence[int], name: str) -> wp.array:
    expected_shape = tuple(shape)
    if values.dtype == wp.float32:
        if tuple(values.shape) != expected_shape:
            raise ValueError(f"{name} shape mismatch: {values.shape} != {expected_shape}")
        return values

    lane_count = _float32_lane_count(values.dtype)
    if lane_count is None:
        raise TypeError(f"{name} dtype mismatch: expected float32-compatible data, got {values.dtype}")
    if tuple(values.shape) + (lane_count,) != expected_shape:
        raise ValueError(f"{name} shape mismatch: {values.shape} with {lane_count} lanes != {expected_shape}")

    alias = wp.array(
        ptr=values.ptr,
        shape=shape,
        dtype=wp.float32,
        device=str(values.device),
        copy=False,
    )
    alias._ref = values
    return alias


class OvPhysxDeformableBodyView:
    """Soft-body view adapter backed by ovphysx per-tensor bindings."""

    def __init__(self, physx_instance: Any, pattern: str, device: str):
        self._ovphysx = physx_instance
        self._pattern = pattern
        self._device = device
        self._bindings: dict[int, Any] = {}

        # Position is the canonical binding for count and node dimensions.
        pos = self._get_binding(TT.DEFORMABLE_SIM_NODAL_POSITION)
        elem = self._get_binding(TT.DEFORMABLE_SIM_ELEMENT_INDICES)
        self._count = pos.count
        self._max_simulation_nodes_per_body = int(pos.shape[1])
        self._max_simulation_elements_per_body = int(elem.shape[1])

    @property
    def count(self) -> int:
        return self._count

    @property
    def prim_paths(self) -> list[str]:
        return list(self._get_binding(TT.DEFORMABLE_SIM_NODAL_POSITION).prim_paths)

    @property
    def max_simulation_nodes_per_body(self) -> int:
        return self._max_simulation_nodes_per_body

    @property
    def max_simulation_elements_per_body(self) -> int:
        return self._max_simulation_elements_per_body

    @property
    def max_collision_nodes_per_body(self) -> int:
        # The ovphysx v1 binding exposes simulation mesh tensors only.
        return 0

    @property
    def max_collision_elements_per_body(self) -> int:
        # The ovphysx v1 binding exposes simulation mesh tensors only.
        return 0

    @property
    def max_sim_vertices_per_body(self) -> int:
        return self._max_simulation_nodes_per_body

    def check(self) -> bool:
        return self._count > 0

    def get_simulation_nodal_positions(self) -> wp.array:
        return self._read_float_binding(TT.DEFORMABLE_SIM_NODAL_POSITION)

    def get_sim_nodal_positions(self) -> wp.array:
        return self.get_simulation_nodal_positions()

    def set_simulation_nodal_positions(self, values: torch.Tensor | wp.array, indices=None) -> None:
        self._write_float_binding(TT.DEFORMABLE_SIM_NODAL_POSITION, values, indices=indices)

    def set_sim_nodal_positions(self, values: torch.Tensor | wp.array, indices=None) -> None:
        self.set_simulation_nodal_positions(values, indices=indices)

    def get_simulation_nodal_velocities(self) -> wp.array:
        return self._read_float_binding(TT.DEFORMABLE_SIM_NODAL_VELOCITY)

    def get_sim_nodal_velocities(self) -> wp.array:
        return self.get_simulation_nodal_velocities()

    def set_simulation_nodal_velocities(self, values: torch.Tensor | wp.array, indices=None) -> None:
        self._write_float_binding(TT.DEFORMABLE_SIM_NODAL_VELOCITY, values, indices=indices)

    def set_sim_nodal_velocities(self, values: torch.Tensor | wp.array, indices=None) -> None:
        self.set_simulation_nodal_velocities(values, indices=indices)

    def get_simulation_nodal_kinematic_targets(self) -> wp.array:
        return self._read_float_binding(TT.DEFORMABLE_SIM_KINEMATIC_TARGET)

    def get_sim_nodal_kinematic_targets(self) -> wp.array:
        return self.get_simulation_nodal_kinematic_targets()

    def set_simulation_nodal_kinematic_targets(self, values: torch.Tensor | wp.array, indices=None) -> None:
        self._write_float_binding(TT.DEFORMABLE_SIM_KINEMATIC_TARGET, values, indices=indices)

    def set_sim_nodal_kinematic_targets(self, values: torch.Tensor | wp.array, indices=None) -> None:
        self.set_simulation_nodal_kinematic_targets(values, indices=indices)

    def get_rest_nodal_positions(self) -> wp.array:
        return self._read_float_binding(TT.DEFORMABLE_REST_NODAL_POSITION)

    def get_simulation_element_indices(self) -> wp.array:
        binding = self._get_binding(TT.DEFORMABLE_SIM_ELEMENT_INDICES)
        dst = wp.zeros(binding.shape, dtype=wp.int32, device=self._device)
        binding.read(dst)
        return dst

    def get_sim_element_indices(self) -> wp.array:
        return self.get_simulation_element_indices()

    def destroy(self) -> None:
        for binding in self._bindings.values():
            binding.destroy()
        self._bindings.clear()

    def _get_binding(self, tensor_type: int):
        binding = self._bindings.get(tensor_type)
        if binding is not None:
            return binding
        binding = self._ovphysx.create_tensor_binding(
            pattern=self._pattern,
            tensor_type=tensor_type,
            raise_if_empty=True,
        )
        self._bindings[tensor_type] = binding
        return binding

    def _read_float_binding(self, tensor_type: int) -> wp.array:
        binding = self._get_binding(tensor_type)
        dst = wp.zeros(binding.shape, dtype=wp.float32, device=self._device)
        binding.read(dst)
        return dst

    def _write_float_binding(self, tensor_type: int, values: torch.Tensor | wp.array, indices=None) -> None:
        binding = self._get_binding(tensor_type)
        src = self._as_float_binding_view(values, binding.shape)
        binding.write(src, indices=self._resolve_indices(indices))

    def _as_float_binding_view(self, values: torch.Tensor | wp.array, shape: Sequence[int]) -> wp.array:
        if isinstance(values, torch.Tensor):
            values = wp.from_torch(values.contiguous(), dtype=wp.float32)
        if str(values.device) != self._device:
            values = wp.clone(values, device=self._device)
        return _as_float32_binding_shape(values, shape, "values")

    def _resolve_indices(self, indices):
        if indices is None:
            return None
        if isinstance(indices, list):
            return wp.array(indices, dtype=wp.int32, device=self._device)
        if isinstance(indices, torch.Tensor):
            return wp.from_torch(indices.to(device=self._device, dtype=torch.int32), dtype=wp.int32)
        if isinstance(indices, wp.array) and str(indices.device) != self._device:
            return wp.clone(indices, device=self._device)
        return indices


class OvPhysxDeformableMaterialView:
    """Deformable material view adapter backed by CPU ovphysx tensor bindings."""

    def __init__(self, physx_instance: Any, pattern: str):
        self._ovphysx = physx_instance
        self._pattern = pattern
        self._bindings: dict[int, Any] = {}
        self._count = self._get_binding(TT.DEFORMABLE_MATERIAL_DYNAMIC_FRICTION).count

    @property
    def count(self) -> int:
        return self._count

    def get_dynamic_frictions(self) -> wp.array:
        return self._read_float_binding(TT.DEFORMABLE_MATERIAL_DYNAMIC_FRICTION)

    def set_dynamic_frictions(self, values: torch.Tensor | wp.array, indices=None, mask=None) -> None:
        self._write_float_binding(TT.DEFORMABLE_MATERIAL_DYNAMIC_FRICTION, values, indices=indices, mask=mask)

    def get_youngs_moduli(self) -> wp.array:
        return self._read_float_binding(TT.DEFORMABLE_MATERIAL_YOUNGS_MODULUS)

    def set_youngs_moduli(self, values: torch.Tensor | wp.array, indices=None, mask=None) -> None:
        self._write_float_binding(TT.DEFORMABLE_MATERIAL_YOUNGS_MODULUS, values, indices=indices, mask=mask)

    def get_poissons_ratios(self) -> wp.array:
        return self._read_float_binding(TT.DEFORMABLE_MATERIAL_POISSONS_RATIO)

    def set_poissons_ratios(self, values: torch.Tensor | wp.array, indices=None, mask=None) -> None:
        self._write_float_binding(TT.DEFORMABLE_MATERIAL_POISSONS_RATIO, values, indices=indices, mask=mask)

    def destroy(self) -> None:
        for binding in self._bindings.values():
            binding.destroy()
        self._bindings.clear()

    def _get_binding(self, tensor_type: int):
        binding = self._bindings.get(tensor_type)
        if binding is not None:
            return binding
        binding = self._ovphysx.create_tensor_binding(
            pattern=self._pattern,
            tensor_type=tensor_type,
            raise_if_empty=True,
        )
        self._bindings[tensor_type] = binding
        return binding

    def _read_float_binding(self, tensor_type: int) -> wp.array:
        binding = self._get_binding(tensor_type)
        dst = wp.zeros(binding.shape, dtype=wp.float32, device="cpu", pinned=True)
        binding.read(dst)
        return dst

    def _write_float_binding(self, tensor_type: int, values: torch.Tensor | wp.array, indices=None, mask=None) -> None:
        binding = self._get_binding(tensor_type)
        src = self._as_cpu_float_array(values, binding.shape)
        binding.write(src, indices=self._resolve_cpu_indices(indices), mask=self._resolve_cpu_mask(mask))

    def _as_cpu_float_array(self, values: torch.Tensor | wp.array, shape: Sequence[int]) -> wp.array:
        if isinstance(values, torch.Tensor):
            values = wp.from_torch(values.contiguous().to(torch.float32), dtype=wp.float32)
        if str(values.device) != "cpu":
            values = wp.clone(values, device="cpu")
        return _as_float32_binding_shape(values, shape, "values")

    def _resolve_cpu_indices(self, indices):
        if indices is None:
            return None
        if isinstance(indices, list):
            return wp.array(indices, dtype=wp.int32, device="cpu")
        if isinstance(indices, torch.Tensor):
            return wp.from_torch(indices.to(torch.int32).cpu(), dtype=wp.int32)
        if isinstance(indices, wp.array) and str(indices.device) != "cpu":
            return wp.clone(indices, device="cpu")
        return indices

    def _resolve_cpu_mask(self, mask):
        if mask is None:
            return None
        if isinstance(mask, torch.Tensor):
            return wp.from_torch(mask.to(torch.bool).cpu(), dtype=wp.bool)
        if isinstance(mask, wp.array) and str(mask.device) != "cpu":
            return wp.clone(mask, device="cpu")
        return mask
