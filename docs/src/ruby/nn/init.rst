.. _init:

.. currentmodule:: mlx.nn.init

Initializers
------------

The ``mlx.nn.init`` package contains commonly used initializers for neural
network parameters. Initializers return a function which can be applied to any
input :obj:`mlx.core.array` to produce an initialized output.

For example:

.. code:: ruby

   require "mlx"
   mx = MLX::Core
   nn = MLX::NN

   init_fn = nn::Init.uniform

   # Produces a [2, 2] uniform matrix
   param = init_fn.call(mx.zeros([2, 2]))

To re-initialize all the parameter in an :obj:`mlx.nn.Module` from say a uniform 
distribution, you can do:

.. code:: ruby
  
   nn = MLX::NN
   model = nn::Sequential.new(
     nn::Linear.new(5, 10),
     nn::ReLU.new,
     nn::Linear.new(10, 5)
   )
   init_fn = nn::Init.uniform(low: -0.1, high: 0.1)
   model.apply(init_fn)
   

.. autosummary::
   :toctree: _autosummary

   constant
   normal
   uniform
   identity
   glorot_normal
   glorot_uniform
   he_normal
   he_uniform
