.. _optimizers:

.. currentmodule:: mlx.optimizers

Optimizers
==========

The optimizers in MLX can be used both with :mod:`mlx.nn` but also with pure
:mod:`mlx.core` functions. A typical example involves calling
:meth:`Optimizer.update` to update a model's parameters based on the loss
gradients and subsequently calling :func:`mlx.core.eval` to evaluate both the
model's parameters and the **optimizer state**.

.. code-block:: ruby

    require "mlx"
    mx = MLX::Core
    nn = MLX::NN
    optim = MLX::Optimizers

    # Create a model
    model = nn::Linear.new(8, 2)
    mx.eval(model.parameters)

    x = mx.random_uniform([16, 8], 0.0, 1.0, mx.float32)
    y = mx.randint(0, 2, [16], mx.int32)
    learning_rate = 1e-1

    loss_fn = ->(model, x, y) { mx.mean(nn.cross_entropy(model.call(x), y)) }
    # Create the gradient function and the optimizer
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim::SGD.new(learning_rate: learning_rate)

    3.times do
      loss, grads = loss_and_grad_fn.call(model, x, y)

      # Update the model with the gradients. So far no computation has happened.
      optimizer.update(model, grads)

      # Compute the new parameters but also the optimizer state.
      mx.eval(loss, model.parameters, optimizer.state)
    end

Saving and Loading
------------------

To serialize an optimizer, save its state. To load an optimizer, load and set
the saved state. Here's a simple example:

.. code-block:: ruby

   require "mlx"
   mx = MLX::Core
   optim = MLX::Optimizers
   optimizer = optim::Adam.new(learning_rate: 1e-2)

   # Perform some updates with the optimizer
   model = {"w" => mx.zeros([5, 5])}
   grads = {"w" => mx.ones([5, 5])}
   optimizer.update(model, grads)
   mx.eval(model, optimizer.state)

   # Save the state
   state = optimizer.state
   serializable = {
     step: mx.array(state["step"]),
     learning_rate: mx.array(state["learning_rate"]),
     m: state["w"]["m"],
     v: state["w"]["v"]
   }
   mx.savez("optimizer_state.npz", **serializable)

   # Later on, for example when loading from a checkpoint,
   # recreate the optimizer and load the state
   optimizer = optim::Adam.new(learning_rate: 1e-2)

   loaded = mx.load("optimizer_state.npz")
   optimizer.state = {
     "step" => loaded["step"].item,
     "learning_rate" => loaded["learning_rate"].item,
     "w" => {"m" => loaded["m"], "v" => loaded["v"]}
   }

Note, not every optimizer configuation parameter is saved in the state. For
example, for Adam the learning rate is saved but the ``betas`` and ``eps``
parameters are not. A good rule of thumb is if the parameter can be scheduled
then it will be included in the optimizer state.

.. toctree::

   optimizers/optimizer
   optimizers/common_optimizers
   optimizers/schedulers

.. autosummary::
   :toctree: _autosummary

   clip_grad_norm
