Train Step DSL
==============

``model.train_step`` returns a reusable, hookable step runner.

.. code-block:: ruby

   step = model.train_step(optimizer: optimizer) do |x:, y:|
     MLX::NN.cross_entropy(model.call(x), y, reduction: "mean")
   end

API
---

- ``model.train_step(optimizer:, clip_grad_norm:, compile:, sync:) { ... }``
- ``step.call(*args, **kwargs)``
- ``step.on(event, priority:, every:, once:, if:)``
- shorthand events: ``before_step``, ``after_backward``, ``after_step``

.. code-block:: ruby

   step = model.train_step(optimizer: optimizer) do |x:, y:|
     MLX::NN.cross_entropy(model.call(x), y, reduction: "mean")
   end

   loss = step.call(x: batch_x, y: batch_y)

Compile and sync
----------------

- ``compile: true|false|{inputs:, outputs:, shapeless:}``
- ``sync: :none|:step``

.. code-block:: ruby

   step = model.train_step(
     optimizer: optimizer,
     compile: { inputs: [:x, :y], shapeless: true },
     sync: :step
   ) { |x:, y:| MLX::NN.cross_entropy(model.call(x), y, reduction: "mean") }

Hook scheduling
---------------

Hooks support:

- deterministic ordering by ``priority``
- periodic execution via ``every``
- one-shot execution via ``once``
- conditional execution via ``if`` predicate

.. code-block:: ruby

   step = model.train_step(optimizer: optimizer, compile: true, sync: :step) do |x:, y:|
     MLX::NN.cross_entropy(model.call(x), y, reduction: "mean")
   end

   step.after_step(priority: -10) { |ctx| puts ctx[:step] }

See implementation:

- ``lib/mlx/dsl/train_step.rb``
