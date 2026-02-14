Trainer Presets and Tasks
=========================

Trainer presets reduce repetitive fit keyword boilerplate.

.. code-block:: ruby

   trainer = trainer.with_fit_defaults(reduce: :mean, monitor_mode: :min)
   trainer.fit(train_data, epochs: 1)

Fit defaults and named presets
------------------------------

- ``with_fit_defaults(**defaults)``
- ``register_fit_preset(name, **defaults)``
- ``fit_with(name, dataset, **overrides)``
- ``fit_report_with(name, dataset, **overrides)``

.. code-block:: ruby

   trainer = trainer.with_fit_defaults(reduce: :mean, monitor_mode: :min)
   trainer.register_fit_preset(:fast, epochs: 3, limit: 128)
   trainer.fit_with(:fast, train_data)

Precedence:

- explicit call overrides
- preset defaults
- trainer defaults
- trainer intrinsic defaults

Task presets
------------

- ``register_task(name, **defaults)``
- ``fit_task(task, dataset, **overrides)``
- ``fit_task_report(task, dataset, **overrides)``

.. code-block:: ruby

   trainer.register_task(:image_cls, monitor: :accuracy, monitor_mode: :max)
   report = trainer.fit_task_report(
     :image_cls,
     train_data,
     validation_data: validation_data,
     epochs: 10
   )
   puts report[:best_monitor]

Built-ins include:

- ``:classification``
- ``:regression``
- ``:language_modeling`` (includes a perplexity-style monitor metric)

.. code-block:: ruby

   trainer = trainer.with_fit_defaults(reduce: :mean, monitor_mode: :min)
   trainer.register_fit_preset(:fast, epochs: 3, limit: 128)

   report = trainer.fit_report_with(:fast, train_data, validation_data: val_data)

See implementation:

- ``lib/mlx/dsl/trainer.rb``
