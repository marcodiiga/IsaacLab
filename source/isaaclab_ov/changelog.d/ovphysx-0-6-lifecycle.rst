Changed
^^^^^^^

* Sealed the populated OVStage ordinal before attaching it to OVPhysX, matching
  the current OVPhysX reader contract.

Added
^^^^^

* Added support for the OVPhysX 0.6 ``warmup()`` API while retaining the
  released 0.5 ``warmup_gpu()`` path.
