.. _unified_memory:

Unified Memory
==============

.. currentmodule:: mlx.core

Apple silicon has a unified memory architecture. The CPU and GPU have direct
access to the same memory pool. MLX is designed to take advantage of that.

Concretely, when you make an array in MLX you don't have to specify its location:


.. code-block:: ruby

  a = mx.random_uniform([100], -1.0, 1.0, mx.float32)
  b = mx.random_uniform([100], -1.0, 1.0, mx.float32)

Both ``a`` and ``b`` live in unified memory.

In MLX, rather than moving arrays to devices, you specify the device when you
run the operation. Any device can perform any operation on ``a`` and ``b``
without needing to move them from one memory location to another. For example:

.. code-block:: ruby

  mx.stream(mx.cpu) { mx.add(a, b) }
  mx.stream(mx.gpu) { mx.add(a, b) }

In the above, both the CPU and the GPU will perform the same add
operation. The operations can (and likely will) be run in parallel since
there are no dependencies between them. See :ref:`using_streams` for more
information the semantics of streams in MLX.

In the above ``add`` example, there are no dependencies between operations, so
there is no possibility for race conditions. If there are dependencies, the
MLX scheduler will automatically manage them. For example:

.. code-block:: ruby

  c = mx.stream(mx.cpu) { mx.add(a, b) }
  d = mx.stream(mx.gpu) { mx.add(a, c) }

In the above case, the second ``add`` runs on the GPU but it depends on the
output of the first ``add`` which is running on the CPU. MLX will
automatically insert a dependency between the two streams so that the second
``add`` only starts executing after the first is complete and ``c`` is
available.

A Simple Example
~~~~~~~~~~~~~~~~

Here is a more interesting (albeit slightly contrived example) of how unified
memory can be helpful. Suppose we have the following computation:

.. code-block:: ruby

  def fun(a, b, d1, d2)
    x = mx.stream(d1) { mx.matmul(a, b) }
    20.times do
      b = mx.stream(d2) { mx.exp(b) }
    end
    [x, b]
  end

which we want to run with the following arguments:

.. code-block:: ruby

  a = mx.random_uniform([1024, 256], 0.0, 1.0, mx.float32)
  b = mx.random_uniform([256, 4], 0.0, 1.0, mx.float32)

The first ``matmul`` operation is a good fit for the GPU since it's more
compute dense. The second sequence of operations are a better fit for the CPU,
since they are very small and would probably be overhead bound on the GPU.

If we time the computation fully on the GPU, we get 2.8 milliseconds. But if we
run the computation with ``d1=mx.gpu`` and ``d2=mx.cpu``, then the time is only
about 1.4 milliseconds, about twice as fast. These times were measured on an M1
Max.
