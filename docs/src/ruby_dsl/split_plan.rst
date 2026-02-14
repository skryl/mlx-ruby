Split Plan DSL
==============

``MLX::DSL.splits`` creates reusable train/validation/test plans that can be
passed directly to ``Trainer#fit`` and ``Trainer#fit_report``.

.. code-block:: ruby

   plan = MLX::DSL.splits do
     train train_data
     validation validation_data
   end

Entry point
-----------

- ``MLX::DSL.splits { ... }``

.. code-block:: ruby

   plan = MLX::DSL.splits do
     train train_data
     validation validation_data
   end

Plan declarations
-----------------

- ``shared(collate:, transform:, limit:, reduce:)``
- ``train(dataset, **options)``
- ``validation(dataset, **options)``
- ``test(dataset, **options)``

Split options map onto trainer fit kwargs:

- ``collate``
- ``transform``
- ``limit``
- ``reduce``

.. code-block:: ruby

   plan = MLX::DSL.splits do
     shared collate: :xy
     train train_data
     validation validation_data, reduce: :mean
   end

   report = trainer.fit_report(plan, epochs: 5)

See implementation:

- ``lib/mlx/dsl/split_plan.rb``
- ``lib/mlx/dsl/trainer.rb``
