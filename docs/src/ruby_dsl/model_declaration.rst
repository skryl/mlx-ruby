Model Declaration DSL
=====================

Use ``MLX::DSL::Model`` or ``MLX::DSL::ModelMixin`` to declare options,
submodules, parameters, and buffers with Ruby macros.

.. code-block:: ruby

   class Head < MLX::DSL::Model
     option :in_dim
     option :out_dim
     layer :proj, MLX::NN::Linear, -> { in_dim }, -> { out_dim }
   end

Class macros
------------

- ``option :name, default:, required:``
- ``layer :name, factory = nil, *factory_args, **factory_kwargs, &block``
- ``network`` (alias for ``layer``)
- ``param :name, shape:, init:, dtype:``
- ``buffer :name, shape:, init:, dtype:``

.. code-block:: ruby

   class TinyClassifier < MLX::DSL::Model
     option :in_dim
     option :classes
     param :temperature, shape: [1], init: 1.0
     buffer :running_scale, shape: [1], init: 1.0

     layer :head, MLX::NN::Linear, -> { in_dim }, -> { classes }
   end

Factory forms
-------------

``layer`` and ``network`` support:

- block-based module construction
- module class + constructor args/kwargs
- callable factory + dynamic args/kwargs

.. code-block:: ruby

   class Block < MLX::DSL::Model
     option :dims, default: 64

     layer :proj, MLX::NN::Linear, -> { dims }, -> { dims }, bias: false

     def call(x)
       proj.call(x)
     end
   end

Runtime helpers
---------------

- ``optimizer_groups { group(matcher) { optimizer } }``
- ``trainer(optimizer:, clip_grad_norm:, compile:, sync:) { ... }``
- ``save_checkpoint`` / ``load_checkpoint``
- ``train_mode`` / ``eval_mode``
- ``freeze_paths!`` / ``unfreeze_paths!``
- ``parameter_paths`` / ``parameter_count`` / ``trainable_parameter_count``
- ``summary(as: :hash|:text)``

.. code-block:: ruby

   model.freeze_paths!("encoder.*")
   puts model.trainable_parameter_count

   trainer = model.trainer(optimizer: optimizer) do |x:, y:|
     MLX::NN.cross_entropy(model.call(x), y, reduction: "mean")
   end

   puts model.summary(as: :text)
   model.unfreeze_paths!("encoder.*")

Checkpoint format notes
-----------------------

Model helpers support marshal and native checkpoints (``.npz``/
``.safetensors``), create parent directories automatically, and support
extensionless native load autodetection.

.. code-block:: ruby

   model.save_checkpoint("artifacts/model.safetensors", optimizer: optimizer)
   model.load_checkpoint("artifacts/model", optimizer: optimizer)

See implementation:

- ``lib/mlx/dsl/model.rb``
- ``lib/mlx/dsl/model_mixin.rb``
