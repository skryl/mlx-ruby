.. _numpy:

Conversion to Other Array Frameworks
====================================

MLX arrays can interoperate with frameworks that support buffer sharing/DLPack.
Use a Ruby-backed bridge (for example :doc:`ruby-dlpack` if you use one) to
convert between MLX arrays and arrays in other ecosystems.

From MLX to another framework
-------------------------------

.. code-block:: ruby

  require "mlx"
  mx = MLX::Core

  x = mx.arange(3)

  # Convert MLX -> external framework using its bridging helper
  external = Bridge.from_mlx(x)
  # Convert back
  x_round_trip = mx.array(external)

From another framework back to MLX
----------------------------------

.. code-block:: ruby

  # x_other should be an array-like value from your target framework
  x = mx.array(x_other)

.. note::

  Many interop helpers currently flow through Python tooling in the wider MLX
  ecosystem. In Ruby, keep conversions focused on adapters available for your
  selected bridge library.

PyTorch
-------

When using a Torch-backed bridge, treat tensors as external arrays and avoid
in-place mutation from non-MLX contexts while gradients are being computed.

.. code-block:: ruby

  # Ruby: require your Torch bridge and load a tensor
  # then pass it through mx.array(...)

JAX
---

For JAX workflows, use the same interoperability path through the adapter layer
you use for NumPy-compatible buffers.

TensorFlow
----------

For TensorFlow workflows, follow your tensor bridge's DLPack/array conversion API.

.. code-block:: ruby

  # TensorFlow tensor -> external buffer -> MLX
  # x_tf = ...
  # x = mx.array(Bridge.from_tensor(x_tf))

