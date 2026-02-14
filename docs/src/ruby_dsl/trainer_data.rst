Trainer Data Ergonomics
=======================

Trainer data wiring supports reusable collation, transforms, binding, and
profile composition.

.. code-block:: ruby

   trainer.fit(
     train_data,
     collate: :xy,
     bind: :auto
   )

Collation
---------

- ``collate`` / ``validation_collate`` accept callables, symbols, and mappings
- built-in schemas: ``:x``, ``:xy``
- registry: ``register_collate(name, spec, extends:)``
- multi-base schema composition via ``extends: [:base_a, :base_b]``

.. code-block:: ruby

   trainer.register_collate(:xy_map, { x: :features, y: :label })
   trainer.fit(train_data, collate: :xy_map, epochs: 3)

Batch schema and auto-collate
-----------------------------

- ``batch_schema(spec)``
- split-specific forms: ``batch_schema(train:, validation:)``
- auto mode: ``collate: :auto`` / ``validation_collate: :auto``

.. code-block:: ruby

   trainer.batch_schema(
     train: { x: :features, y: :label },
     validation: { x: :input, y: :target }
   )

   trainer.fit(
     train_data,
     validation_data: validation_data,
     collate: :auto,
     validation_collate: :auto
   )

Binding
-------

- ``bind`` and ``validation_bind``
- supports ``:auto`` argument-name inference (including common aliases)
- supports explicit key path mappings (for example ``{x: [:payload, :x]}``)
- explicit ``collate`` takes precedence over bind

.. code-block:: ruby

   trainer.fit(
     train_data,
     bind: :auto,
     validation_data: validation_data,
     validation_bind: { x: [:payload, :x], y: [:payload, :y] }
   )

Transforms and dynamic limits
-----------------------------

- ``train_transform`` / ``validation_transform`` with context-aware signatures
- ``limit`` / ``validation_limit`` accept integers or callables per epoch

.. code-block:: ruby

   trainer.fit(
     train_data,
     train_transform: ->(batch, epoch:) { augment(batch, epoch: epoch) },
     limit: ->(epoch:) { epoch < 3 ? 100 : 500 },
     validation_data: validation_data,
     validation_limit: 100
   )

Dataflow profiles
-----------------

- ``register_dataflow(name, train:, validation:, extends:)``
- ``use_dataflow(name, **overrides)``

.. code-block:: ruby

   trainer.register_dataflow(
     :xy_flow,
     train: { collate: { x: 0, y: 1 }, limit: 256 },
     validation: { collate: { x: :input, y: :target }, reduce: :mean }
   )

   report = trainer.fit_report(data, **trainer.use_dataflow(:xy_flow))

See implementation:

- ``lib/mlx/dsl/trainer.rb``
