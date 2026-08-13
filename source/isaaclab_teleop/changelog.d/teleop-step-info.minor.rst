Added
^^^^^

* Added :class:`~isaaclab_teleop.TeleopStepInfo` and
  :attr:`~isaaclab_teleop.IsaacTeleopDevice.last_step_info` to expose stable request
  identity, result age, compute time, dropped submissions, deadline misses, and worker
  failure status. Failure metadata remains available after a failed session is torn down.
  Kitless device flows can construct without requiring Kit XR stage services while Kit-hosted
  replay continues to honor dynamic anchor transforms.
