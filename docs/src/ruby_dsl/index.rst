Ruby DSL
========

The Ruby DSL provides high-level ergonomics on top of ``MLX::NN::Module`` and
``MLX::Optimizers`` while preserving compatibility with native MLX execution.

Core entry points:

- ``MLX::DSL::Model``
- ``MLX::DSL::ModelMixin``
- ``MLX::DSL::Trainer``
- ``MLX::DSL::Data``
- ``MLX::DSL.experiment``
- ``MLX::DSL.splits``

.. code-block:: ruby

   require "mlx"

   model = MyModel.new
   trainer = model.trainer(optimizer: optimizer) { |x:, y:| loss_fn.call(model.call(x), y) }
   data = MLX::DSL::Data.from(train_rows).batch(32)
   plan = MLX::DSL.splits { train data }

Quick start:

.. code-block:: ruby

   require "mlx"

   class Mlp < MLX::DSL::Model
     option :in_dim
     option :hidden_dim, default: 128
     option :out_dim

     layer :net do
       sequential do
         linear in_dim, hidden_dim
         relu
         linear hidden_dim, out_dim
       end
     end

     def call(x)
       net.call(x)
     end
   end

   model = Mlp.new(in_dim: 32, out_dim: 10)

Reference pages:

- :doc:`model_declaration`
- :doc:`builder_and_graphs`
- :doc:`train_step`
- :doc:`trainer_core`
- :doc:`trainer_data`
- :doc:`trainer_presets`
- :doc:`checkpoints_and_resume`
- :doc:`artifact_policy`
- :doc:`data_pipeline`
- :doc:`experiment`
- :doc:`split_plan`
- :doc:`examples`

.. code-block:: ruby

   # Typical progression:
   # 1) model_declaration + builder_and_graphs
   # 2) train_step + trainer_core/trainer_data
   # 3) checkpoints_and_resume + artifact_policy
   # 4) experiment + split_plan for reusable run configs
