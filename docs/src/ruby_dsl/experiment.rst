Experiment DSL
==============

Use ``MLX::DSL.experiment`` for declarative run wiring across model,
optimizer, trainer, data, and artifacts.

.. code-block:: ruby

   exp = MLX::DSL.experiment("demo") do
     model { model }
     optimizer { optimizer }
   end

Entry point
-----------

- ``MLX::DSL.experiment(name = nil) { ... }``

.. code-block:: ruby

   exp = MLX::DSL.experiment("classifier") do
   end

Declaration sections
--------------------

- ``model { ... }``
- ``optimizer { ... }``
- ``trainer(**kwargs) { loss }`` or ``trainer(existing_trainer)``
- ``data(train:, validation:, **fit_kwargs)``
- ``artifacts(**fit_kwargs)``

.. code-block:: ruby

   exp = MLX::DSL.experiment("classifier") do
     model { Classifier.new(input_dim: 128, num_classes: 10) }
     optimizer { MLX::Optimizers::Adam.new(learning_rate: 1e-3) }
     trainer { |x:, y:| MLX::NN.cross_entropy(model.call(x), y, reduction: "mean") }
     data train: train_data, validation: validation_data
     artifacts checkpoint_path: "checkpoints/exp-%{epoch}.bin"
   end

Execution helpers
-----------------

- ``run(report: false, **overrides)``
- ``report(**overrides)``
- ``save_run_bundle(path, report:, config:, **overrides)``

.. code-block:: ruby

   exp = MLX::DSL.experiment("mnist") do
     model { model }
     optimizer { optimizer }
     trainer { |x:, y:| MLX::NN.cross_entropy(model.call(x), y, reduction: "mean") }
     data train: train_data, validation: validation_data
     artifacts checkpoint_path: "checkpoints/ep-%{epoch}.bin"
   end

   report = exp.report(epochs: 5)

See implementation:

- ``lib/mlx/dsl/experiment.rb``
