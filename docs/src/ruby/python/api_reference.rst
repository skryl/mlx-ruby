Ruby API Reference
==================

This page summarizes the Ruby-facing API surface for MLX Ruby.

The runtime entry point is:

.. code-block:: ruby

  require "mlx"
  mx = MLX::Core
  nn = MLX::NN
  optim = MLX::Optimizers

.. _ops:

Core Operations
---------------

Core tensor and math operations live in ``MLX::Core``.

Representative groups include:

- Array creation and shape ops: ``array``, ``arange``, ``reshape``, ``transpose``, ``concatenate``, ``stack``
- Math ops: ``add``, ``subtract``, ``multiply``, ``divide``, ``exp``, ``log``, ``sqrt``
- Reductions/statistics: ``sum``, ``mean``, ``max``, ``min``, ``argmax``, ``argmin``
- Device/memory helpers: ``cpu``, ``gpu``, ``default_device``, ``set_default_device``

See implementation:

- ``lib/mlx/core.rb``
- ``ext/mlx/native.cpp``

.. _transforms:

Function Transforms
-------------------

MLX Ruby exposes higher-order transformations via ``MLX::Core`` and ``MLX::NN``.

- ``MLX::Core.grad``
- ``MLX::Core.value_and_grad``
- ``MLX::Core.vmap``
- ``MLX::Core.jvp``
- ``MLX::Core.vjp``
- ``MLX::Core.compile``
- ``MLX::NN.value_and_grad`` (module-parameter convenience wrapper)

.. _module_class:

Neural Network Module
---------------------

The base class for models and layers is ``MLX::NN::Module``.

Key methods:

- Parameter and state handling: ``parameters``, ``trainable_parameters``, ``state``, ``update``
- Mode control: ``train``, ``eval``
- Persistence: ``save_weights``, ``load_weights``

Important pattern: register trainable members with ``self.<name> = ...`` so
they are tracked by module state and optimizer updates.

See implementation:

- ``lib/mlx/nn/base.rb``
- ``lib/mlx/nn/layers/*.rb``

Optimizers
----------

Optimizers are under ``MLX::Optimizers``.

Common classes:

- ``SGD``
- ``Adam``
- ``AdamW``
- ``RMSprop``
- ``Adagrad``
- ``AdaDelta``
- ``Adamax``
- ``Lion``
- ``Adafactor``
- ``Muon``

Schedulers are under ``MLX::Optimizers::Schedulers``.

See implementation:

- ``lib/mlx/optimizers/optimizers.rb``
- ``lib/mlx/optimizers/schedulers.rb``

.. _export:

Export and Import
-----------------

MLX Ruby can serialize and load compiled function artifacts.

Core entry points in ``MLX::Core``:

- ``export_function``
- ``import_function``
- ``exporter``
- ``save`` / ``load``
- ``savez`` / ``savez_compressed``

.. _distributed:

Distributed
-----------

Distributed helpers are exposed via:

- ``MLX::Core`` distributed collectives (native bindings)
- ``MLX::DistributedUtils`` runtime/config tooling

See implementation:

- ``lib/mlx/distributed_utils/common.rb``
- ``lib/mlx/distributed_utils/config.rb``
- ``lib/mlx/distributed_utils/launch.rb``
