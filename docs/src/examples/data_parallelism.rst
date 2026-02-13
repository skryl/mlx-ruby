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

    model = ...
    optimizer = ...
    dataset = ...

    def step(model, x, y)
        loss, grads = loss_grad_fn(model, x, y)
        optimizer.update(model, grads)
        loss
    end

    dataset.each do |x, y|
        loss = step(model, x, y)
        mx.eval(loss, model.parameters)
    end

All we have to do to average the gradients across machines is perform an
:func:`all_sum` and divide by the size of the :class:`Group`. Namely we
have to :func:`MLX::Utils.tree_map` the gradients with following function.

.. code:: ruby

    def all_avg(x)
        mx.distributed.all_sum(x) / mx.distributed.init.size
    end

Putting everything together our training loop step looks as follows with
everything else remaining the same.

.. code:: ruby

    # Ruby: implement tree_map on nested structures via MLX utilities

    def all_reduce_grads(grads)
        world_size = mx.distributed.init.size
        return grads if world_size == 1

        MLX::Utils.tree_map(
            ->(x) { mx.distributed.all_sum(x) / world_size },
            grads
        )
    end

    def step(model, x, y)
        loss, grads = loss_grad_fn(model, x, y)
        grads = all_reduce_grads(grads)  # <--- This line was added
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

    model = ...
    optimizer = ...
    dataset = ...

    def step(model, x, y)
        loss, grads = loss_grad_fn(model, x, y)
        grads = MLX::NN.average_gradients(grads)  # <---- This line was added
        optimizer.update(model, grads)
        loss
    end

    dataset.each do |x, y|
        loss = step(model, x, y)
        mx.eval(loss, model.parameters)
    end
