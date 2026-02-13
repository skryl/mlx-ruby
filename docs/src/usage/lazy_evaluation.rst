.. _lazy eval:

Lazy Evaluation
===============

.. currentmodule:: mlx.core

Why Lazy Evaluation
-------------------

When you perform operations in MLX, no computation actually happens. Instead a
compute graph is recorded. The actual computation only happens if an
:func:`eval` is performed.

MLX uses lazy evaluation because it has some nice features, some of which we
describe below.

Transforming Compute Graphs
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Lazy evaluation lets us record a compute graph without actually doing any
computations. This is useful for function transformations like :func:`grad` and
:func:`vmap` and graph optimizations.

Currently, MLX does not compile and rerun compute graphs. They are all
generated dynamically. However, lazy evaluation makes it much easier to
integrate compilation for future performance enhancements.

Only Compute What You Use
^^^^^^^^^^^^^^^^^^^^^^^^^

In MLX you do not need to worry as much about computing outputs that are never
used. For example:

.. code-block:: ruby

  fun1 = ->(x) { mx.exp(x) }
  expensive_fun = ->(a) { mx.exp(mx.exp(a)) }
  fun = ->(x) do
    a = fun1.call(x)
    b = expensive_fun.call(a)
    [a, b]
  end

  x = mx.array(1.0)
  y, _ = fun.call(x)
  mx.eval(y)

Here, we never actually compute the output of ``expensive_fun``. Use this
pattern with care though, as the graph of ``expensive_fun`` is still built, and
that has some cost associated to it.

Similarly, lazy evaluation can be beneficial for saving memory while keeping
code simple. Say you have a very large model ``Model`` derived from
:obj:`mlx.nn.Module`. You can instantiate this model with ``model = Model()``.
Typically, this will initialize all of the weights as ``float32``, but the
initialization does not actually compute anything until you perform an
:func:`eval`. If you update the model with ``float16`` weights, your maximum
consumed memory will be half that required if eager computation was used
instead.

This pattern is simple to do in MLX thanks to lazy computation:

.. code-block:: ruby

  model = MLX::NN::Linear.new(8, 8) # no memory used yet
  mx.eval(model.parameters)

When to Evaluate
----------------

A common question is when to use :func:`eval`. The trade-off is between
letting graphs get too large and not batching enough useful work.

For example:

.. code-block:: ruby

  a = mx.array(1.0)
  b = mx.array(2.0)
  100.times do
    a = a + b
    mx.eval(a)
    b = b * 2
    mx.eval(b)
  end

This is a bad idea because there is some fixed overhead with each graph
evaluation. On the other hand, there is some slight overhead which grows with
the compute graph size, so extremely large graphs (while computationally
correct) can be costly.

Luckily, a wide range of compute graph sizes work pretty well with MLX:
anything from a few tens of operations to many thousands of operations per
evaluation should be okay.

Most numerical computations have an iterative outer loop (e.g. the iteration in
stochastic gradient descent). A natural and usually efficient place to use
:func:`eval` is at each iteration of this outer loop.

Here is a concrete example:

.. code-block:: ruby

   model = MLX::NN::Linear.new(4, 1)
   optimizer = MLX::Optimizers::SGD.new(learning_rate: 0.1)
   value_and_grad_fn = MLX::NN.value_and_grad(
     model,
     ->(model, batch) do
       x, y = batch
       MLX::NN.binary_cross_entropy(model.call(x).squeeze, y)
     end
   )
   dataset = [
     [mx.random_uniform([4, 4], 0.0, 1.0, mx.float32), mx.array([0.0, 1.0, 0.0, 1.0])],
     [mx.random_uniform([4, 4], 0.0, 1.0, mx.float32), mx.array([1.0, 0.0, 1.0, 0.0])]
   ]

   dataset.each do |batch|
     # Nothing has been evaluated yet
     loss, grad = value_and_grad_fn.call(model, batch)

     # Still nothing has been evaluated
     optimizer.update(model, grad)

     # Evaluate the loss and the new parameters which will
     # run the full gradient computation and optimizer update
     mx.eval(loss, model.parameters)
   end


An important behavior to be aware of is when the graph will be implicitly
evaluated. Anytime you ``print`` an array, convert it to an
:obj:`numpy.ndarray`, or otherwise access its memory via :obj:`memoryview`,
the graph will be evaluated. Saving arrays via :func:`save` (or any other MLX
saving functions) will also evaluate the array.


Calling :func:`array.item` on a scalar array will also evaluate it. In the
example above, printing the loss (``print(loss)``) or adding the loss scalar to
a list (``losses.append(loss.item())``) would cause a graph evaluation. If
these lines are before ``mx.eval(loss, model.parameters())`` then this
will be a partial evaluation, computing only the forward pass.

Also, calling :func:`eval` on an array or set of arrays multiple times is
perfectly fine. This is effectively a no-op.

.. warning::

  Using scalar arrays for control-flow will cause an evaluation.

Here is an example:

.. code-block:: ruby

   first_layer = ->(x) { [x * 2.0, x.sum] }
   second_layer_a = ->(h) { mx.exp(h) }
   second_layer_b = ->(h) { h.square }

   fun = ->(x) do
     h, y = first_layer.call(x)
     if y.item > 0 # An evaluation is done here!
       second_layer_a.call(h)
     else
       second_layer_b.call(h)
     end
   end

   mx.eval(fun.call(mx.array([1.0, -0.5, 0.25])))

Using arrays for control flow should be done with care. The above example works
and can even be used with gradient transformations. However, this can be very
inefficient if evaluations are done too frequently.
