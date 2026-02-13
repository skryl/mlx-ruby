Quick Start Guide
=================


Basics
------

.. currentmodule:: mlx.core

Import ``mlx.core`` and make an :class:`array`:

.. code-block:: ruby

  require "mlx"
  mx = MLX::Core
  a = mx.array([1, 2, 3, 4])
  puts a.shape.inspect
  puts a.dtype
  b = mx.array([1.0, 2.0, 3.0, 4.0])
  puts b.dtype

Operations in MLX are lazy. The outputs of MLX operations are not computed
until they are needed. To force an array to be evaluated use
:func:`eval`.  Arrays will automatically be evaluated in a few cases. For
example, inspecting a scalar with :meth:`array.item`, printing an array,
or converting an array from :class:`array` to an external array bridge all
automatically evaluate the array.

.. code-block:: ruby

  c = a + b    # c not yet evaluated
  mx.eval(c)  # evaluates c
  c = a + b
  puts c      # Also evaluates c
  c = a + b
  p c.to_a    # => [2.0, 4.0, 6.0, 8.0]


See the page on :ref:`Lazy Evaluation <lazy eval>` for more details.

Function and Graph Transformations
----------------------------------

MLX has standard function transformations like :func:`grad` and :func:`vmap`.
Transformations can be composed arbitrarily. For example
``grad(vmap(grad(fn)))`` (or any other composition) is allowed.

.. code-block:: ruby

  x = mx.array(0.0)
  puts mx.sin(x)
  puts mx.grad(->(t) { mx.sin(t) }).call(x)
  puts mx.grad(mx.grad(->(t) { mx.sin(t) })).call(x)

Other gradient transformations include :func:`vjp` for vector-Jacobian products
and :func:`jvp` for Jacobian-vector products.

Use :func:`value_and_grad` to efficiently compute both a function's output and
gradient with respect to the function's input.
