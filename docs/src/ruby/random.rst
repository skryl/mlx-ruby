.. _random:

Random
======

Random sampling in MLX Ruby is exposed through core functions:
``random_seed``, ``random_split``, and ``random_uniform``.

For example, you can generate random numbers with:

.. code-block:: ruby

  require "mlx"
  mx = MLX::Core

  3.times do
    puts mx.random_uniform([1], 0.0, 1.0, mx.float32).item
  end

which will print a sequence of unique pseudo random numbers. Alternatively you
can explicitly set the key:

.. code-block:: ruby

  require "mlx"
  mx = MLX::Core

  mx.random_seed(0)
  key = mx.array([0, 0], mx.uint32)
  k1, k2 = mx.random_split(key)

  3.times do
    puts mx.random_uniform([1], 0.0, 1.0, mx.float32).item
  end

which will yield the same pseudo random number at each iteration.

Following `JAX's PRNG design <https://jax.readthedocs.io/en/latest/jep/263-prng.html>`_
we use a splittable version of Threefry, which is a counter-based PRNG.

.. currentmodule:: mlx.core

.. autosummary:: 
  :toctree: _autosummary

   random_seed
   random_split
   random_uniform
