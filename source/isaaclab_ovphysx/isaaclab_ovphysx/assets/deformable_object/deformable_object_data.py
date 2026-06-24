# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import weakref

import warp as wp

from isaaclab.assets.deformable_object.base_deformable_object_data import BaseDeformableObjectData
from isaaclab.utils.buffers import TimestampedBufferWarp as TimestampedBuffer
from isaaclab.utils.warp import ProxyArray

from .kernels import compute_mean_vec3f_over_vertices, compute_nodal_state_w, vec6f
from .views import OvPhysxDeformableBodyView


class DeformableObjectData(BaseDeformableObjectData):
    """Data container for an ovphysx-backed deformable object."""

    def __init__(self, root_view: OvPhysxDeformableBodyView, device: str):
        """Initialize the deformable object data."""
        super().__init__(device)
        self._root_view: OvPhysxDeformableBodyView = weakref.proxy(root_view)

        self._num_instances = root_view.count
        self._max_sim_vertices = root_view.max_simulation_nodes_per_body
        self._max_sim_elements = root_view.max_simulation_elements_per_body
        self._max_collision_elements = root_view.max_collision_elements_per_body

        self._nodal_pos_w = TimestampedBuffer((self._num_instances, self._max_sim_vertices), device, wp.vec3f)
        self._nodal_vel_w = TimestampedBuffer((self._num_instances, self._max_sim_vertices), device, wp.vec3f)
        self._nodal_state_w = TimestampedBuffer((self._num_instances, self._max_sim_vertices), device, vec6f)
        self._root_pos_w = TimestampedBuffer((self._num_instances,), device, wp.vec3f)
        self._root_vel_w = TimestampedBuffer((self._num_instances,), device, wp.vec3f)

        self._nodal_pos_w_ta: ProxyArray | None = None
        self._nodal_vel_w_ta: ProxyArray | None = None
        self._nodal_state_w_ta: ProxyArray | None = None
        self._root_pos_w_ta: ProxyArray | None = None
        self._root_vel_w_ta: ProxyArray | None = None

    default_nodal_state_w: ProxyArray | None = None
    """Default nodal state ``[nodal_pos, nodal_vel]`` in simulation world frame."""

    nodal_kinematic_target: ProxyArray | None = None
    """Simulation mesh kinematic targets for the deformable bodies."""

    @property
    def nodal_pos_w(self) -> ProxyArray:
        """Nodal positions in simulation world frame."""
        if self._nodal_pos_w.timestamp < self._sim_timestamp:
            src = (
                self._root_view.get_simulation_nodal_positions()
                .view(wp.vec3f)
                .reshape((self._num_instances, self._max_sim_vertices))
            )
            wp.copy(self._nodal_pos_w.data, src)
            self._nodal_pos_w.timestamp = self._sim_timestamp
            if self._nodal_pos_w_ta is not None:
                self._nodal_pos_w_ta = ProxyArray(self._nodal_pos_w.data)
        if self._nodal_pos_w_ta is None:
            self._nodal_pos_w_ta = ProxyArray(self._nodal_pos_w.data)
        return self._nodal_pos_w_ta

    @property
    def nodal_vel_w(self) -> ProxyArray:
        """Nodal velocities in simulation world frame."""
        if self._nodal_vel_w.timestamp < self._sim_timestamp:
            src = (
                self._root_view.get_simulation_nodal_velocities()
                .view(wp.vec3f)
                .reshape((self._num_instances, self._max_sim_vertices))
            )
            wp.copy(self._nodal_vel_w.data, src)
            self._nodal_vel_w.timestamp = self._sim_timestamp
            if self._nodal_vel_w_ta is not None:
                self._nodal_vel_w_ta = ProxyArray(self._nodal_vel_w.data)
        if self._nodal_vel_w_ta is None:
            self._nodal_vel_w_ta = ProxyArray(self._nodal_vel_w.data)
        return self._nodal_vel_w_ta

    @property
    def nodal_state_w(self) -> ProxyArray:
        """Nodal state ``[nodal_pos, nodal_vel]`` in simulation world frame."""
        if self._nodal_state_w.timestamp < self._sim_timestamp:
            wp.launch(
                compute_nodal_state_w,
                dim=(self._num_instances, self._max_sim_vertices),
                inputs=[self.nodal_pos_w.warp, self.nodal_vel_w.warp],
                outputs=[self._nodal_state_w.data],
                device=self.device,
            )
            self._nodal_state_w.timestamp = self._sim_timestamp
        if self._nodal_state_w_ta is None:
            self._nodal_state_w_ta = ProxyArray(self._nodal_state_w.data)
        return self._nodal_state_w_ta

    @property
    def root_pos_w(self) -> ProxyArray:
        """Mean nodal position in simulation world frame."""
        if self._root_pos_w.timestamp < self._sim_timestamp:
            wp.launch(
                compute_mean_vec3f_over_vertices,
                dim=(self._num_instances,),
                inputs=[self.nodal_pos_w.warp, self._max_sim_vertices],
                outputs=[self._root_pos_w.data],
                device=self.device,
            )
            self._root_pos_w.timestamp = self._sim_timestamp
        if self._root_pos_w_ta is None:
            self._root_pos_w_ta = ProxyArray(self._root_pos_w.data)
        return self._root_pos_w_ta

    @property
    def root_vel_w(self) -> ProxyArray:
        """Mean nodal velocity in simulation world frame."""
        if self._root_vel_w.timestamp < self._sim_timestamp:
            wp.launch(
                compute_mean_vec3f_over_vertices,
                dim=(self._num_instances,),
                inputs=[self.nodal_vel_w.warp, self._max_sim_vertices],
                outputs=[self._root_vel_w.data],
                device=self.device,
            )
            self._root_vel_w.timestamp = self._sim_timestamp
        if self._root_vel_w_ta is None:
            self._root_vel_w_ta = ProxyArray(self._root_vel_w.data)
        return self._root_vel_w_ta
