MLX
===

MLX is a NumPy-like array framework designed for efficient and flexible machine
learning on Apple silicon, brought to you by Apple machine learning research.

The Ruby API closely follows NumPy with a few exceptions. MLX also has a
fully featured C++ API which closely follows the Ruby API.

The main differences between MLX and NumPy are:

 - **Composable function transformations**: MLX has composable function
   transformations for automatic differentiation, automatic vectorization,
   and computation graph optimization.
 - **Lazy computation**: Computations in MLX are lazy. Arrays are only
   materialized when needed.
 - **Multi-device**: Operations can run on any of the supported devices (CPU,
   GPU, ...)

The design of MLX is inspired by frameworks like `PyTorch
<https://pytorch.org/>`_, `Jax <https://github.com/google/jax>`_, and
`ArrayFire <https://arrayfire.org/>`_. A notable difference from these
frameworks and MLX is the *unified memory model*. Arrays in MLX live in shared
memory. Operations on MLX arrays can be performed on any of the supported
device types without performing data copies. Currently supported device types
are the CPU and GPU.

.. toctree::
   :caption: Install
   :maxdepth: 1

   install

.. toctree::
   :caption: Usage 
   :maxdepth: 1

   usage/quick_start
   usage/lazy_evaluation
   usage/unified_memory
   usage/indexing
   usage/saving_and_loading
   usage/function_transforms
   usage/compile
   usage/numpy
   usage/distributed
   usage/using_streams
   usage/export

.. toctree::
   :caption: Examples
   :maxdepth: 1

   examples/linear_regression
   examples/mlp
   examples/llama-inference
   examples/data_parallelism
   examples/tensor_parallelism

.. toctree::
   :caption: Ruby API Reference
   :maxdepth: 1

   ruby/api_reference

.. toctree::
   :caption: Ruby DSL
   :maxdepth: 1

   ruby_dsl/index
   ruby_dsl/model_declaration
   ruby_dsl/builder_and_graphs
   ruby_dsl/train_step
   ruby_dsl/trainer_core
   ruby_dsl/trainer_data
   ruby_dsl/trainer_presets
   ruby_dsl/checkpoints_and_resume
   ruby_dsl/artifact_policy
   ruby_dsl/data_pipeline
   ruby_dsl/experiment
   ruby_dsl/split_plan
   ruby_dsl/examples

.. toctree::
   :caption: C++ API Reference
   :maxdepth: 1

   cpp/ops

.. toctree::
   :caption: Further Reading
   :maxdepth: 1

   dev/extensions
   dev/metal_debugger
   dev/metal_logging
   dev/custom_metal_kernels
   dev/mlx_in_cpp
