Artifact Policy
===============

Artifact policies make checkpoint/run-bundle lifecycle behavior declarative.

.. code-block:: ruby

   trainer.artifact_policy(
     checkpoint: { path: "checkpoints/ep-%{epoch}.bin", strategy: :latest },
     retention: { keep_last_n: 2 }
   )

API
---

- ``artifact_policy(checkpoint:, retention:, resume:, run_bundle:)``
- ``checkpoint_history``

.. code-block:: ruby

   trainer.artifact_policy(
     checkpoint: { path: "checkpoints/ep-%{epoch}.bin" },
     retention: { keep_last_n: 2 },
     resume: :latest
   )
   pp trainer.checkpoint_history

Checkpoint policy
-----------------

``checkpoint`` supports:

- ``path``
- ``strategy``: ``:latest``, ``:best``, ``:every``
- ``every`` for periodic saves

.. code-block:: ruby

   trainer.artifact_policy(
     checkpoint: {
       path: "checkpoints/%{monitor_name}-%{epoch}.bin",
       strategy: :every,
       every: 2
     }
   )

Retention policy
----------------

``retention`` supports:

- ``keep_last_n``

.. code-block:: ruby

   trainer.artifact_policy(
     checkpoint: { path: "checkpoints/ep-%{epoch}.bin", strategy: :every, every: 1 },
     retention: { keep_last_n: 3 }
   )

Resume policy
-------------

``resume`` supports policy-driven selection (for example ``:latest`` and
``:best``), as well as explicit values accepted by ``resume_from``.

.. code-block:: ruby

   trainer.artifact_policy(
     checkpoint: { path: "checkpoints/ep-%{epoch}.bin", strategy: :best },
     resume: :best
   )
   trainer.fit(train_data, epochs: 10)

Run-bundle policy
-----------------

``run_bundle`` supports:

- ``enabled``
- ``path``
- ``config``

When enabled, trainer reports include ``run_bundle_path`` and policy metadata.

.. code-block:: ruby

   trainer.artifact_policy(
     checkpoint: { path: "checkpoints/ep-%{epoch}.bin", strategy: :latest },
     retention: { keep_last_n: 3 },
     resume: :latest,
     run_bundle: { enabled: true, path: "artifacts/auto_bundle.json" }
   )

See implementation:

- ``lib/mlx/dsl/trainer.rb``
