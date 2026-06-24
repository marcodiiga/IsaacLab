# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp

vec6f = wp.types.vector(length=6, dtype=wp.float32)


@wp.kernel
def write_nodal_vec3f_to_buffer(
    data: wp.array2d(dtype=wp.vec3f),
    env_ids: wp.array(dtype=wp.int32),
    full_data: bool,
    out_data: wp.array2d(dtype=wp.vec3f),
):
    """Write selected nodal vec3f rows into a full-instance buffer."""
    i, j = wp.tid()
    if full_data:
        out_data[env_ids[i], j] = data[env_ids[i], j]
    else:
        out_data[env_ids[i], j] = data[i, j]


@wp.kernel
def write_nodal_vec4f_to_buffer(
    data: wp.array2d(dtype=wp.vec4f),
    env_ids: wp.array(dtype=wp.int32),
    full_data: bool,
    out_data: wp.array2d(dtype=wp.vec4f),
):
    """Write selected nodal vec4f rows into a full-instance buffer."""
    i, j = wp.tid()
    if full_data:
        out_data[env_ids[i], j] = data[env_ids[i], j]
    else:
        out_data[env_ids[i], j] = data[i, j]


@wp.kernel
def compute_nodal_state_w(
    nodal_pos: wp.array2d(dtype=wp.vec3f),
    nodal_vel: wp.array2d(dtype=wp.vec3f),
    nodal_state: wp.array2d(dtype=vec6f),
):
    """Concatenate nodal position and velocity into a vec6f state."""
    i, j = wp.tid()
    p = nodal_pos[i, j]
    v = nodal_vel[i, j]
    nodal_state[i, j] = vec6f(p[0], p[1], p[2], v[0], v[1], v[2])


@wp.kernel
def compute_mean_vec3f_over_vertices(
    data: wp.array2d(dtype=wp.vec3f),
    num_vertices: int,
    result: wp.array(dtype=wp.vec3f),
):
    """Compute the mean vec3f over the vertex dimension."""
    i = wp.tid()
    acc = wp.vec3f(0.0, 0.0, 0.0)
    for j in range(num_vertices):
        acc = acc + data[i, j]
    result[i] = acc / float(num_vertices)


@wp.kernel
def set_kinematic_flags_to_one(data: wp.array(dtype=wp.vec4f)):
    """Initialize all kinematic target flags to non-kinematic."""
    i = wp.tid()
    v = data[i]
    data[i] = wp.vec4f(v[0], v[1], v[2], 1.0)
