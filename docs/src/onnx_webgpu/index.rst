Onnx/WebGPU Support
===================

This section is a process guide for the MLX -> Graph IR -> ONNX -> WebGPU
path in MLX Ruby.

If you want API-by-API reference details, see :doc:`../ruby/export`.

Pipeline stages
---------------

1. Capture MLX execution as Graph IR.
2. Validate Graph IR schema/topology.
3. Run WebGPU/ONNX compatibility preflight.
4. Convert Graph IR to ONNX (binary and/or JSON stub).
5. Package browser harness assets.
6. Run smoke validation in Node/browser runtime.

Pages in this section:

- :doc:`mlx_to_graph_ir`
- :doc:`validation_and_compatibility`
- :doc:`graph_ir_to_onnx`
- :doc:`webgpu_harness_and_smoke`
- :doc:`end_to_end_examples`
