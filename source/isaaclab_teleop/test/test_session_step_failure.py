# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for step-failure handling in :class:`TeleopSessionLifecycle`.

The async retarget worker dies permanently on any pipeline exception, so
session re-entry is the only recovery.  These tests cover the failure
diagnosis (pipeline error vs. external XR teardown) and the restart
cooldown that prevents a persistent pipeline error from churning
teardown/restart cycles every frame.

All heavy dependencies (isaacteleop, carb, omni.kit, isaacsim) are stubbed
out via ``sys.modules`` so these tests run in a plain Python environment.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub out isaacteleop and Kit modules before any isaaclab_teleop imports.
# ---------------------------------------------------------------------------

_MODULES_TO_STUB = [
    "isaacteleop",
    "isaacteleop.cloudxr",
    "isaacteleop.oxr",
    "isaacteleop.retargeting_engine",
    "isaacteleop.retargeting_engine.interface",
    "isaacteleop.retargeting_engine.interface.execution_events",
    "isaacteleop.retargeting_engine.interface.retargeter_core_types",
    "isaacteleop.retargeting_engine.interface.tensor_group_type",
    "isaacteleop.retargeting_engine.deviceio_source_nodes",
    "isaacteleop.retargeting_engine.deviceio_source_nodes.deviceio_tensor_types",
    "isaacteleop.retargeting_engine.tensor_types",
    "isaacteleop.retargeting_engine_ui",
    "isaacteleop.teleop_session_manager",
    "isaacteleop.teleop_session_manager.teleop_state_manager_retargeter",
    "isaacteleop.teleop_session_manager.teleop_state_manager_types",
    "isaacsim",
    "isaacsim.kit",
    "isaacsim.kit.xr",
    "isaacsim.kit.xr.teleop",
    "isaacsim.kit.xr.teleop.bridge",
    "carb",
    "carb.settings",
    "carb.eventdispatcher",
    "omni",
    "omni.kit",
    "omni.kit.app",
    "omni.kit.xr",
    "omni.kit.xr.system",
    "omni.kit.xr.system.openxr",
]

_stubs_installed: dict[str, ModuleType | MagicMock] = {}


class _FakeBaseRetargeter:
    """Stand-in base class so modules subclassing ``BaseRetargeter`` import cleanly."""

    def __init__(self, name: str) -> None:
        self.name = name


def _install_stubs():
    """Insert MagicMock modules for all heavy dependencies."""
    for name in _MODULES_TO_STUB:
        if name not in sys.modules:
            sys.modules[name] = _stubs_installed.setdefault(name, MagicMock())
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            setattr(sys.modules[parent_name], child_name, sys.modules[name])

    iface_mod = sys.modules["isaacteleop.retargeting_engine.interface"]
    iface_mod.BaseRetargeter = _FakeBaseRetargeter  # type: ignore[attr-defined]
    iface_mod.RetargeterIOType = dict  # type: ignore[attr-defined]


def _restore_stubs():
    """Remove stubs installed for this test module from ``sys.modules``."""
    for name in reversed(_MODULES_TO_STUB):
        stub = _stubs_installed.get(name)
        if stub is None:
            continue
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = sys.modules.get(parent_name)
            if parent is not None and getattr(parent, child_name, None) is stub:
                delattr(parent, child_name)
        if sys.modules.get(name) is stub:
            del sys.modules[name]


_install_stubs()

from isaaclab_teleop import TeleopStepInfo  # noqa: E402
from isaaclab_teleop.isaac_teleop_cfg import IsaacTeleopCfg  # noqa: E402
from isaaclab_teleop.isaac_teleop_device import IsaacTeleopDevice  # noqa: E402
from isaaclab_teleop.session_lifecycle import TeleopSessionLifecycle  # noqa: E402

_restore_stubs()


@pytest.fixture(autouse=True)
def _stub_heavy_dependencies():
    """Keep these tests isolated from modules collected later in the suite."""
    _install_stubs()
    yield
    _restore_stubs()


def _make_lifecycle(**kwargs) -> TeleopSessionLifecycle:
    cfg = IsaacTeleopCfg(pipeline_builder=lambda: None, control_channel_uuid=None)
    return TeleopSessionLifecycle(cfg, **kwargs)


def _make_failing_lifecycle(error: Exception) -> TeleopSessionLifecycle:
    """Lifecycle with a live session mock whose step raises *error*."""
    lifecycle = _make_lifecycle()
    lifecycle._pipeline = object()
    session = MagicMock()
    session.step.side_effect = error
    session.last_step_info = _make_step_info()
    lifecycle._session = session
    return lifecycle


def _make_step_info(**overrides) -> SimpleNamespace:
    """Return an upstream-shaped metadata object without importing the optional package."""
    values = {
        "returned_frame_id": None,
        "submitted_frame_id": None,
        "returned_age_frames": None,
        "returned_age_s": None,
        "compute_duration_s": None,
        "dropped_submissions": 0,
        "ran_synchronously": False,
        "frame_deadline_miss": False,
        "worker_exception": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestStepFailureDiagnosis:
    def test_pipeline_error_with_active_xr_sets_holdoff(self):
        lifecycle = _make_failing_lifecycle(ValueError("Found zero norm quaternions in `quat`."))
        session = lifecycle._session
        session.last_step_info = _make_step_info(worker_exception=RuntimeError("retarget worker failed"))
        with (
            patch.object(TeleopSessionLifecycle, "_kit_xr_session_is_active", return_value=True),
            patch.object(lifecycle, "_build_external_inputs", return_value=None),
        ):
            assert lifecycle.step() is None
        session.__exit__.assert_called_once()
        assert lifecycle._session is None
        assert lifecycle._restart_holdoff_until > time.monotonic()
        assert lifecycle.last_step_info == TeleopStepInfo(worker_failed=True, worker_error_type="RuntimeError")

    def test_external_xr_teardown_has_no_holdoff(self):
        lifecycle = _make_failing_lifecycle(RuntimeError("XR_ERROR_INSTANCE_LOST"))
        session = lifecycle._session
        with (
            patch.object(TeleopSessionLifecycle, "_kit_xr_session_is_active", return_value=False),
            patch.object(lifecycle, "_build_external_inputs", return_value=None),
        ):
            assert lifecycle.step() is None
        session.__exit__.assert_called_once()
        assert lifecycle._session is None
        # Stop-AR -> Start-AR recovery latency must stay unchanged.
        assert lifecycle._restart_holdoff_until == 0.0


class TestRestartHoldoff:
    def test_holdoff_blocks_session_restart(self):
        lifecycle = _make_lifecycle()
        lifecycle._pipeline = object()
        lifecycle._restart_holdoff_until = time.monotonic() + 60.0
        with patch.object(lifecycle, "_try_start_session") as try_start:
            assert lifecycle.step() is None
        try_start.assert_not_called()

    def test_expired_holdoff_allows_restart(self):
        lifecycle = _make_lifecycle()
        lifecycle._pipeline = object()
        lifecycle._restart_holdoff_until = time.monotonic() - 1.0
        with patch.object(lifecycle, "_try_start_session", return_value=False) as try_start:
            assert lifecycle.step() is None
        try_start.assert_called_once()


class TestLastStepInfo:
    def test_lifecycle_returns_none_without_active_session(self):
        lifecycle = _make_lifecycle()

        assert lifecycle.last_step_info is None

    def test_lifecycle_forwards_active_session_metadata(self):
        lifecycle = _make_lifecycle()
        step_info = _make_step_info(
            returned_frame_id=6,
            submitted_frame_id=7,
            returned_age_frames=1,
            returned_age_s=0.02,
            compute_duration_s=0.004,
            dropped_submissions=2,
            frame_deadline_miss=True,
        )
        lifecycle._session = MagicMock(last_step_info=step_info)

        assert lifecycle.last_step_info == TeleopStepInfo(
            returned_frame_id=6,
            submitted_frame_id=7,
            returned_age_frames=1,
            returned_age_s=0.02,
            compute_duration_s=0.004,
            dropped_submissions=2,
            frame_deadline_miss=True,
        )

    def test_lifecycle_exposes_default_metadata_before_first_step(self):
        lifecycle = _make_lifecycle()
        lifecycle._session = MagicMock(last_step_info=_make_step_info())

        assert lifecycle.last_step_info == TeleopStepInfo()

    def test_lifecycle_tolerates_session_without_metadata_api(self):
        lifecycle = _make_lifecycle()
        lifecycle._session = SimpleNamespace()

        assert lifecycle.last_step_info is None

    def test_dead_session_teardown_tolerates_missing_metadata_api(self):
        lifecycle = _make_lifecycle()
        exit_session = MagicMock()
        lifecycle._session = SimpleNamespace(__exit__=exit_session)

        lifecycle._teardown_dead_session()

        exit_session.assert_called_once_with(None, None, None)
        assert lifecycle._session is None
        assert lifecycle.last_step_info is None

    def test_device_forwards_lifecycle_metadata(self):
        device = object.__new__(IsaacTeleopDevice)
        step_info = TeleopStepInfo(returned_frame_id=4, submitted_frame_id=5)
        device._session_lifecycle = MagicMock(last_step_info=step_info)

        assert device.last_step_info is step_info


def test_device_imports_and_constructs_without_carb():
    """Kitless consumers can construct the device when the Kit carb module is absent."""
    script = """
import importlib.abc
import sys

class BlockCarb(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "carb" or fullname.startswith("carb."):
            raise ModuleNotFoundError("carb deliberately unavailable")
        return None

for name in tuple(sys.modules):
    if name == "carb" or name.startswith("carb."):
        del sys.modules[name]
sys.meta_path.insert(0, BlockCarb())
from isaaclab_teleop import IsaacTeleopCfg, IsaacTeleopDevice, TeleopStepInfo
device = IsaacTeleopDevice(
    IsaacTeleopCfg(pipeline_builder=lambda: None),
    use_kit_xr_bridge=False,
    mcap_replay_path="/tmp/nonexistent-replay",
)
assert device.last_step_info is None
assert TeleopStepInfo is not None
del device
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)


@pytest.mark.parametrize(
    ("use_kit_xr_bridge", "mcap_replay_path"),
    [(True, "/tmp/replay"), (False, None)],
)
def test_device_preserves_stage_aware_anchor_manager(use_kit_xr_bridge, mcap_replay_path):
    """Replay and standalone modes keep dynamic anchor transforms when Kit is available."""
    cfg = IsaacTeleopCfg(pipeline_builder=lambda: None)

    with patch("isaaclab_teleop.isaac_teleop_device.XrAnchorManager") as anchor_manager:
        IsaacTeleopDevice(
            cfg,
            use_kit_xr_bridge=use_kit_xr_bridge,
            mcap_replay_path=mcap_replay_path,
        )

    anchor_manager.assert_called_once_with(cfg.xr_cfg)


def test_replay_agent_disables_kit_xr_bridge():
    """Replay bypasses live Kit handles while preserving stage-aware anchor transforms."""
    repo_root = Path(__file__).resolve().parents[3]
    replay_agent = (repo_root / "scripts/environments/teleoperation/teleop_replay_agent.py").read_text(encoding="utf-8")

    assert re.search(
        r"create_isaac_teleop_device\([\s\S]*?use_kit_xr_bridge=False,[\s\S]*?mcap_replay_path=",
        replay_agent,
    )
