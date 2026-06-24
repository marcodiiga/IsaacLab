# OVPhysX DeformableObject Adapter Implementation SDD

## Documentation Control

| Item | Description |
|------|-------------|
| Title | OVPhysX DeformableObject Adapter |
| Author(s) | IsaacLab and Physics teams |
| Revision | 2026-06-08 |
| State | Draft |

## 1 Introduction

### 1.1 Purpose and Scope

This document records the IsaacLab-local implementation design for enabling
`DeformableObject` on the `ovphysx` backend. The adapter lives under
`source/isaaclab_ovphysx` and uses ovphysx TensorBindings directly. It does not
import Kit, `omni.physics.tensors`, or the old PhysX TensorAPI frontend.

The scope is limited to volume deformables, nodal position and velocity state,
kinematic targets, root-state summaries derived from nodal data, and deformable
material scalar properties exposed by ovphysx.

The supported authoring path for this implementation is any USD stage that
authors volume-deformable body and material schemas before the ovphysx backend
loads it. This includes already-authored volume TetMesh assets. Kit-backed
procedural mesh-to-volume-deformable tetrahedralization remains outside the
kitless ovphysx runtime adapter.

### 1.2 Requirements

- Provide an `isaaclab_ovphysx.assets.deformable_object.DeformableObject`
  implementation registered through IsaacLab's existing backend factory model.
- Mirror the old `isaaclab_physx` deformable object behavior where it maps onto
  ovphysx: asset discovery, nodal state buffers, kinematic target buffers,
  index and mask entrypoints, and debug visualization.
- Keep implementation changes backend-local except for exported ovphysx tensor
  type aliases.
- Use the new ovphysx deformable tensor bindings for body and material data.
- Support indexed writes for nodal positions, nodal velocities, and kinematic
  targets.
- Initialize simulator-side kinematic target flags to the IsaacLab default free
  node state before partial target writes.

### 1.3 Non-Goals

- Kit or `omni.physics.tensors` imports.
- Surface deformables.
- Procedural mesh-to-volume-deformable tetrahedralization.
- Collision mesh tensors.
- Element rotations, deformation gradients, stresses, attachments, or other
  fields not exposed by ovphysx.

## 2 Architecture

The adapter follows the existing `isaaclab_ovphysx` asset pattern:

- `tensor_types.py` provides ovphysx tensor type aliases.
- `assets/deformable_object/views.py` wraps ovphysx `TensorBinding` objects in a
  SoftBodyView-compatible shape.
- `assets/deformable_object/deformable_object.py` implements the backend asset
  by subclassing the core `BaseDeformableObject`.
- `assets/deformable_object/deformable_object_data.py` implements lazy Warp
  buffers for nodal and derived root state.
- `assets/deformable_object/kernels.py` holds small Warp kernels shared by the
  asset and data container.

The design deliberately does not add a generic soft-body sim-view shim. The
current ovphysx backend already uses per-tensor bindings for assets, and the
deformable implementation stays consistent with that pattern.

Deformables also opt the ovphysx backend out of its runtime `physx.clone()`
fast path for the current scene. `physx.clone()` does not support the authored
deformable body and material schema prims, so the backend exports and parses the
full USD-authored environment clone set when a `DeformableObject` is present.
Rigid and articulation scenes keep the env_0-only export plus runtime clone
path.

## 3 Design Details

### 3.1 Asset Discovery

The adapter starts from the user-provided deformable `prim_path`, finds the
template prim, then searches the template subtree for a prim authored with
`OmniPhysicsDeformableBodyAPI`. In a kitless ovphysx USD environment, USD
Python may not report Omni deformable APIs through `GetAppliedSchemas()`, so the
adapter also reads authored `apiSchemas` metadata. This preserves compatibility
with old PhysX-authored assets without introducing a Kit dependency.

Material discovery uses the authored physics material binding relationship and
the same schema metadata helper. Volume deformables are accepted; surface
deformables fail clearly.

### 3.2 Tensor Views

`OvPhysxDeformableBodyView` owns cached ovphysx tensor bindings for:

- simulation nodal positions,
- simulation nodal velocities,
- simulation nodal kinematic targets,
- rest nodal positions,
- simulation element indices.

The body view exposes shape and count properties expected by IsaacLab's
deformable object data path. Body tensors use the simulation device.

`OvPhysxDeformableMaterialView` owns cached CPU tensor bindings for dynamic
friction, Young's modulus, and Poisson's ratio. It supports indexed and masked
material writes through ovphysx's CPU binding path.

### 3.3 Data Flow

On initialization, the asset creates body and material views, creates Warp
buffers, reads initial nodal positions, computes default nodal state, and reads
the kinematic target tensor. It sets all kinematic target flags to the
IsaacLab free-node convention and writes that default target buffer back to
ovphysx so simulator state matches the asset data contract before indexed
writes.

Nodal position, velocity, and kinematic target writes update the asset's full
internal buffer first, then forward the selected rows to ovphysx through the
indexed TensorBinding path. Lazy data properties refresh from ovphysx using the
simulation timestamp inherited from the base data class.

For `InteractiveScene`, the asset queues USD replication and marks
`OvPhysxManager` as requiring full USD export. If other assets have queued
ovphysx runtime clones in the same scene, the manager skips those pending clone
operations because their USD-authored copies are already present in the full
exported stage.

### 3.4 Error Handling

Initialization fails if no matching deformable body is found, if multiple body
roots are found below the template prim, if the asset is not a volume
deformable, or if ovphysx cannot create the required body view. Unsupported
surface-deformable behavior raises `NotImplementedError`.

### 3.5 Security and Observability

The adapter introduces no new network access, subprocess execution, or
persistent storage. Diagnostics use existing IsaacLab and ovphysx logging paths.

## 4 Test Automation

Required validation:

- Install an ovphysx wheel containing deformable tensor bindings into the
  environment that runs IsaacLab tests.
- Run Ruff format/check on touched Python and stub files.
- Compile touched Python files with `python -m py_compile`.
- Run `source/isaaclab_ovphysx/test/assets/test_deformable_object.py` with Kit
  Python and the ovphysx backend.
- Run a sample-style smoke that authors a pre-tetrahedralized volume deformable
  in memory using IsaacLab's stage/spawner facilities, creates
  `isaaclab.assets.DeformableObject`, applies kinematic targets, steps
  simulation, and reads finite nodal positions.

The test covers volume deformable creation, two-instance discovery, nodal
position indexed write/readback, velocity write/readback, kinematic target
indexed write/readback, material property readback, stepping/updating finite
nodal positions, and three-environment `InteractiveScene` replication through
the full-USD deformable fallback.

## 5 References

- `source/isaaclab_ovphysx/isaaclab_ovphysx/assets/deformable_object/`
- `source/isaaclab_ovphysx/isaaclab_ovphysx/tensor_types.py`
- `source/isaaclab_ovphysx/test/assets/test_deformable_object.py`
- `source/isaaclab_physx/isaaclab_physx/assets/deformable_object/`
