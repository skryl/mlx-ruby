Builder and Graphs
==================

The DSL builder composes modules and callables into graph structures.

.. code-block:: ruby

   class Block < MLX::DSL::Model
     layer :net do
       sequential do
         linear 64, 64
         relu
       end
     end
   end

Composition helpers
-------------------

- ``sequential``
- ``residual``
- ``branch``
- ``concat(axis:)``
- ``sum``
- ``layer`` (module instance, module class, callable, or block)
- ``fn`` / ``lambda_layer``

.. code-block:: ruby

   layer :encoder do
     residual do
       sequential do
         linear 128, 128
         relu
         linear 128, 128
       end
     end
   end

Repetition helpers
------------------

- ``repeat_layers(count) { |i| ... }``
- ``stack(count, layer_class = nil, *args, **kwargs, &block)``

.. code-block:: ruby

   layer :tower do
     sequential do
       repeat_layers(2) { relu }
       stack(3, MLX::NN::Linear, 128, 128)
     end
   end

Variadic forwarding
-------------------

Sequential and callable compositions support variadic positional and keyword
forwarding for multi-input graph wiring.

.. code-block:: ruby

   layer :fusion do
     fn do |x, skip:, scale: 1.0|
       x + (skip * scale)
     end
   end

Layer shorthands
----------------

Builder exposes common layers and blocks including linear/conv/norm/pool,
recurrent layers, and transformer components.

.. code-block:: ruby

   layer :block do
     sequential do
       conv2d 32, 64, kernel_size: 3, padding: 1
       batch_norm 64
       relu
       max_pool2d kernel_size: 2, stride: 2
     end
   end

See implementation:

- ``lib/mlx/dsl/builder.rb``
- ``lib/mlx/dsl/graph_modules.rb``
- ``lib/mlx/nn/layers/containers.rb``
