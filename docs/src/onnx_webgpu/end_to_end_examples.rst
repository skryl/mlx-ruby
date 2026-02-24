End To End Examples
===================

This page shows practical workflows for MLX -> Graph IR -> ONNX -> WebGPU.

Minimal script example
----------------------

.. code-block:: ruby

   require "json"
   require "stringio"
   require "mlx"

   mx = MLX::Core
   x = mx.array([[1.0, 2.0]], mx.float32)
   y = mx.array([[0.5, 0.25]], mx.float32)

   trace = ->(lhs, rhs) { MLX::Core.add(lhs, rhs) }
   payload = MLX::ONNX.export_graph_ir(trace, x, y)
   onnx_json = MLX::ONNX.graph_ir_to_onnx_json(payload, model_name: "minimal_add")

   report = JSON.parse(
     MLX::ONNX::Native.ir_compatibility_report_json(payload)
   )
   abort("unsupported ops: #{report.fetch('unsupported_ops').inspect}") unless report.fetch("unsupported_nodes").zero?

   MLX::ONNX.graph_ir_to_onnx("artifacts/model.onnx", payload, model_name: "minimal_add")
   MLX::ONNX::WebGPUHarness.export_onnx_webgpu_harness(
     "artifacts/web_harness",
     payload,
     model_name: "minimal_add"
   )

   telemetry = MLX::ONNX::WebGPUHarness.smoke_test_onnx_webgpu_harness(
     "artifacts/web_harness",
     mock_ort: true
   )
   puts telemetry.fetch("format")

Repository task examples
------------------------

The repository already contains end-to-end export flows:

- ``tasks/web_assets_task/export_gpt2_assets.rb``
- ``tasks/web_assets_task/export_nanogpt_assets.rb``
- ``tasks/web_assets_task/export_stable_diffusion_assets.rb``

Run the web asset pipeline:

.. code-block:: bash

   bundle exec rake web:assets

Browser demo integration
------------------------

Generated assets are consumed by:

- ``web/demo/gpt2/``
- ``web/demo/nanogpt/``
- ``web/demo/stable_diffusion/``

Use the local server task to validate demos after export:

.. code-block:: bash

   bundle exec rake web:serve

Parity and harness checks
-------------------------

Useful parity coverage examples:

- ``test/integration/onnx/export_onnx_webgpu_harness_test.rb``
- ``test/integration/onnx/onnx_webgpu_harness_smoke_test.rb``
- ``test/integration/onnx/onnx_webgpu_harness_real_runtime_smoke_test.rb``
