.. _data_parallelism:

Data Parallelism
================

MLX enables efficient data parallel distributed training through its
distributed communication primitives.

.. _training_example:

Training Example
----------------

In this section we will adapt an MLX training loop to support data parallel
distributed training. Namely, we will average the gradients across a set of
hosts before applying them to the model.

Our training loop looks like the following code snippet if we omit the model,
dataset, and optimizer initialization.

.. code:: ruby

    model = Struct.new(:parameters).new({"w" => mx.array([0.0])})
    optimizer = Struct.new(:calls) do
      def update(model, grads)
        model.parameters["w"] = model.parameters["w"] - grads["w"] * 0.1
      end
    end.new(0)
    dataset = [[MLX::Core.array([1.0]), MLX::Core.array([1.0])]]

    loss_grad_fn = ->(model, x, y) do
      pred = model.parameters["w"] * x
      loss = mx.mean((pred - y).square)
      grads = {"w" => mx.mean((pred - y) * x * 2.0)}
      [loss, grads]
    end

    step = ->(model, x, y) do
      loss, grads = loss_grad_fn.call(model, x, y)
      optimizer.update(model, grads)
      loss
    end

    dataset.each do |x, y|
      loss = step.call(model, x, y)
      mx.eval(loss, model.parameters)
    end

All we have to do to average the gradients across machines is perform an
:func:`all_sum` and divide by the size of the :class:`Group`. Namely we
have to :func:`MLX::Utils.tree_map` the gradients with following function.

.. code:: ruby

    def all_avg(x)
      world = mx.init
      mx.all_sum(x, world) / world.size
    end

Putting everything together our training loop step looks as follows with
everything else remaining the same.

.. code:: ruby

    def all_reduce_grads(grads)
      world = mx.init
      world_size = world.size
      return grads if world_size == 1

      MLX::Utils.tree_map(
        ->(x) { mx.all_sum(x, world) / world_size },
        grads
      )
    end

    step = ->(model, x, y) do
      loss, grads = loss_grad_fn.call(model, x, y)
      grads = all_reduce_grads(grads) # <--- This line was added
      optimizer.update(model, grads)
      loss
    end

Using ``nn.average_gradients``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Although the code example above works correctly; it performs one communication
per gradient. It is significantly more efficient to aggregate several gradients
together and perform fewer communication steps.

This is the purpose of :func:`mlx.nn.average_gradients`. The final code looks
almost identical to the example above:

.. code:: ruby

    model = Struct.new(:parameters).new({"w" => mx.array([0.0])})
    optimizer = Struct.new(:calls) do
      def update(model, grads)
        model.parameters["w"] = model.parameters["w"] - grads["w"] * 0.1
      end
    end.new(0)
    dataset = [[MLX::Core.array([1.0]), MLX::Core.array([1.0])]]

    loss_grad_fn = ->(model, x, y) do
      pred = model.parameters["w"] * x
      loss = mx.mean((pred - y).square)
      grads = {"w" => mx.mean((pred - y) * x * 2.0)}
      [loss, grads]
    end

    step = ->(model, x, y) do
      world = mx.init
      loss, grads = loss_grad_fn.call(model, x, y)
      grads = MLX::NN.average_gradients(grads, world) # <---- This line was added
      optimizer.update(model, grads)
      loss
    end

    dataset.each do |x, y|
      loss = step.call(model, x, y)
      mx.eval(loss, model.parameters)
    end
