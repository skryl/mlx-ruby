WebGPU Harness And Smoke
========================

After ONNX export, package browser assets and run smoke verification.

Package harness assets
----------------------

Use ``export_onnx_webgpu_harness`` to emit browser-ready files.

.. code-block:: ruby

   manifest = MLX::Core.export_onnx_webgpu_harness(
     "artifacts/web_harness",
     payload,
     model_name: "demo_graph",
     execution_providers: %w[webgpu wasm],
     benchmark_warmup_runs: 1,
     benchmark_measure_runs: 5
   )

   puts manifest.fetch("format") # => "onnx_webgpu_harness_v1"

Generated files include:

- ``model.onnx``
- ``harness.manifest.json``
- ``inputs.example.json``
- ``index.html``
- ``harness.js``

Run smoke test
--------------

Use ``smoke_test_onnx_webgpu_harness`` to validate harness wiring and runtime
selection.

.. code-block:: ruby

   telemetry = MLX::Core.smoke_test_onnx_webgpu_harness(
     "artifacts/web_harness",
     timeout_seconds: 30,
     mock_ort: false,
     local_ort: true
   )

   puts telemetry.fetch("format") # => "onnx_webgpu_telemetry_v1"
   puts telemetry.fetch("provider", "unknown")

Notes
-----

- Smoke tests require Node.js.
- Real runtime checks require Playwright + ``onnxruntime-web`` in the harness
  environment.
