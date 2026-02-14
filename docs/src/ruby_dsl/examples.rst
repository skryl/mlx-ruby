Ruby DSL Examples
=================

Runnable examples in this repository:

- ``examples/dsl/streaming_factory.rb``
- ``examples/dsl/validation_monitor.rb``
- ``examples/dsl/memory_friendly_reporting.rb``
- ``examples/dsl/collate_schemas.rb``
- ``examples/dsl/compile_sync_and_native_checkpoint.rb``

Suggested reading path:

1. Start with ``streaming_factory.rb`` and ``validation_monitor.rb``.
2. Review ``collate_schemas.rb`` for data wiring patterns.
3. Review ``compile_sync_and_native_checkpoint.rb`` for compile/sync and
   persistence.
4. Use ``memory_friendly_reporting.rb`` for long-running report ergonomics.

.. code-block:: ruby

   require "mlx"

   # Mirrors the structure used by examples/dsl/validation_monitor.rb
   trainer = model.trainer(optimizer: optimizer) do |x:, y:|
     MLX::NN.cross_entropy(model.call(x), y, reduction: "mean")
   end

   report = trainer.fit_report(
     train_data,
     epochs: 3,
     validation_data: validation_data,
     monitor: :loss
   )

   puts report[:best_monitor]
