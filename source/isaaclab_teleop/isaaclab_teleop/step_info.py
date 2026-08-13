# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stable step metadata for Isaac Teleop integrations."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class TeleopStepInfo:
    """Identity, age, and timing metadata for the latest teleop step.

    In pipelined execution, a call can submit one request while returning an
    older completed result. The two frame identifiers make that relationship
    explicit without implying that the call waited for the submitted frame.

    Attributes:
        returned_frame_id: Identifier of the completed result returned by the
            latest step, or ``None`` before a result is available.
        submitted_frame_id: Identifier of the request submitted by the latest
            step, or ``None`` before a request is submitted.
        returned_age_frames: Age of the returned result relative to the
            submitted request [frames], or ``None`` when unavailable.
        returned_age_s: Age of the returned result at return time [s], or
            ``None`` when unavailable.
        compute_duration_s: Retargeting compute duration for the returned
            result [s], or ``None`` when unavailable.
        dropped_submissions: Requests dropped by this step because the
            pipelined worker was still busy.
        ran_synchronously: Whether this step performed retargeting on the
            calling thread.
        frame_deadline_miss: Whether the returned result missed its frame
            deadline.
        worker_failed: Whether the retargeting worker reported a failure.
        worker_error_type: Exception class name reported by the worker, or
            ``None`` when no worker failure was reported.
    """

    returned_frame_id: int | None = None
    submitted_frame_id: int | None = None
    returned_age_frames: int | None = None
    returned_age_s: float | None = None
    compute_duration_s: float | None = None
    dropped_submissions: int = 0
    ran_synchronously: bool = False
    frame_deadline_miss: bool = False
    worker_failed: bool = False
    worker_error_type: str | None = None
