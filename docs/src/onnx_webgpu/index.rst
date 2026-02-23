Onnx/WebGPU Support
===================

This section is a process guide for the MLX -> Graph IR -> ONNX -> WebGPU
path in MLX Ruby.

If you want API-by-API reference details, see :doc:`../ruby/export`.

Architecture boundary (post-refactor)
-------------------------------------

Use ``MLX::ONNX.*`` methods as the public API. The implementation is split across:

- ``MLX::ONNX``
- ``MLX::ONNX::Native``
- ``MLX::ONNX::WebGPUHarness``

Pipeline stages
---------------

1. Capture MLX execution as Graph IR.
2. Convert Graph IR to ONNX (JSON stub and/or binary model).
3. Run optional compatibility diagnostics (native report).
5. Package browser harness assets.
6. Run smoke validation in Node/browser runtime.

Pages in this section:

- :doc:`mlx_to_onnx`
- :doc:`validation_and_compatibility`
- :doc:`onnx_to_onnx`
- :doc:`webgpu_harness_and_smoke`
- :doc:`end_to_end_examples`
