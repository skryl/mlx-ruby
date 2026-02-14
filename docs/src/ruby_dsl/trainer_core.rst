Trainer Core
============

``MLX::DSL::Trainer`` provides high-level training and reporting workflows.

.. code-block:: ruby

   trainer = model.trainer(optimizer: optimizer) do |x:, y:|
     MLX::NN.cross_entropy(model.call(x), y, reduction: "mean")
   end

Core methods
------------

- ``fit(dataset, **kwargs)``
- ``fit_report(dataset, **kwargs)``

.. code-block:: ruby

   trainer.fit(train_data, epochs: 5)
   report = trainer.fit_report(train_data, epochs: 5, validation_data: validation_data)
   puts report[:epochs_ran]

Core fit options
----------------

- ``epochs``, ``limit``, ``reduce``
- ``validation_data``, ``validation_limit``, ``validation_reduce``
- ``monitor``, ``metric``, ``monitor_mode``
- ``patience``, ``min_delta``
- ``keep_losses``, ``strict_data_reuse``
- ``compile`` and ``sync`` (via trainer construction)

.. code-block:: ruby

   trainer.fit_report(
     train_data,
     epochs: 20,
     limit: 200,
     reduce: :mean,
     validation_data: validation_data,
     validation_limit: 50,
     monitor: :loss,
     monitor_mode: :min,
     patience: 3,
     min_delta: 0.001
   )

Lifecycle hooks
---------------

- ``before_fit``
- ``before_epoch``
- ``after_batch``
- ``before_validation``
- ``after_validation_batch``
- ``after_validation``
- ``after_epoch``
- ``checkpoint``
- ``after_fit``

All hook registration routes through ``on(event, ...)`` and supports
``priority``, ``every``, ``once``, and conditional ``if``.

.. code-block:: ruby

   trainer.before_epoch { |ctx| puts "epoch=#{ctx[:epoch]}" }
   trainer.after_batch(every: 50) { |ctx| puts "step=#{ctx[:step]}" }
   trainer.after_fit { |ctx| puts "best=#{ctx[:best_monitor]}" }

Error diagnostics
-----------------

Train/validation batch failures include ``kind``, ``epoch``, and
``batch_index`` context.

.. code-block:: ruby

   begin
     trainer.fit(train_data, epochs: 1)
   rescue => e
     warn("#{e.class}: #{e.message}")
     warn("kind=#{e.respond_to?(:kind) ? e.kind : :unknown}")
   end

See implementation:

- ``lib/mlx/dsl/trainer.rb``
