Checkpoints and Resume
======================

The DSL provides model-level checkpoint helpers and trainer-level resume
orchestration.

.. code-block:: ruby

   model.save_checkpoint("checkpoints/latest.bin", optimizer: optimizer)
   model.load_checkpoint("checkpoints/latest.bin", optimizer: optimizer)

Model checkpoint helpers
------------------------

- ``save_checkpoint(path, optimizer:, metadata:, format:)``
- ``load_checkpoint(path, optimizer:, strict:, format:)``

Supported formats:

- marshal-compatible payload
- native weights + sidecar metadata (``.npz``/``.safetensors``)

.. code-block:: ruby

   model.save_checkpoint(
     "checkpoints/model_epoch_5.safetensors",
     optimizer: optimizer,
     metadata: { epoch: 5, val_loss: 0.42 }
   )

   model.load_checkpoint("checkpoints/model_epoch_5.safetensors", optimizer: optimizer)

Trainer checkpoint flows
------------------------

- ``checkpoint_path`` accepts template strings and callables
- template placeholders include ``%{epoch}``, ``%{next_epoch}``,
  ``%{monitor}``, ``%{monitor_name}``, ``%{epoch_loss}``, ``%{improved}``

.. code-block:: ruby

   trainer.fit(
     train_data,
     epochs: 10,
     validation_data: validation_data,
     monitor: :loss,
     checkpoint_path: "checkpoints/ep-%{epoch}-loss-%{monitor}.bin"
   )

Resume sources
--------------

``resume_from`` accepts:

- checkpoint path
- run-bundle path
- run-bundle hash
- inline metadata payload hash
- callable loader

.. code-block:: ruby

   trainer
     .resume_from("checkpoints/ep-10.bin")
     .fit(train_data, epochs: 20)

Run bundles
-----------

- ``run_bundle(report:, config:, schema_version:)``
- ``save_run_bundle(path, report:, config:, schema_version:)``
- ``resume_payload_from_bundle(bundle_or_path)``

.. code-block:: ruby

   report = trainer.fit_report(train_data, epochs: 3, validation_data: validation_data)
   trainer.save_run_bundle("artifacts/run_bundle.json", report: report, config: { experiment: "mnist" })

   payload = trainer.resume_payload_from_bundle("artifacts/run_bundle.json")
   trainer.resume_from(payload).fit(train_data, epochs: 5)

See implementation:

- ``lib/mlx/dsl/model_mixin.rb``
- ``lib/mlx/dsl/trainer.rb``
