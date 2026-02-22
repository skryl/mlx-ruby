WebGPU Harness And Smoke
========================

After ONNX export, package browser assets and run smoke verification.

Ownership boundary
------------------

- Public facade API:
  ``MLX::GraphIR.export_onnx_webgpu_harness`` and
  ``MLX::GraphIR.smoke_test_onnx_webgpu_harness``.
- Implementation module: ``MLX::GraphIR::WebGPUHarness``.

Package harness assets
----------------------

Use ``export_onnx_webgpu_harness`` to emit browser-ready files.

.. code-block:: ruby

   manifest = MLX::GraphIR.export_onnx_webgpu_harness(
     "artifacts/web_harness",
     payload,
     model_name: "demo_graph",
     execution_providers: %w[webgpu wasm],
     benchmark_warmup_runs: 1,
     benchmark_measure_runs: 5,
     external_data: false
   )

   puts manifest.fetch("format") # => "onnx_webgpu_harness_v1"
   puts manifest.fetch("execution_providers").inspect # => ["webgpu", "wasm"]

Generated files include:

- ``model.onnx``
- ``harness.manifest.json``
- ``inputs.example.json``
- ``index.html``
- ``harness.js``
- optional external data file (for example ``model.data``)

Manifest fields include:

- ``format`` (``onnx_webgpu_harness_v1``)
- ``model`` (currently ``model.onnx``)
- ``execution_providers`` (``webgpu``/``wasm``)
- ``benchmark`` (``warmup_runs`` and ``measure_runs``)
- ``inputs`` (name/shape/dtype for each model input)
- optional ``external_data`` file list when external-data export is enabled

Run smoke test
--------------

Use ``smoke_test_onnx_webgpu_harness`` to validate harness wiring and runtime
selection.

.. code-block:: ruby

   telemetry = MLX::GraphIR.smoke_test_onnx_webgpu_harness(
     "artifacts/web_harness",
     timeout_seconds: 30,
     mock_ort: false,
     local_ort: true,
     node_bin: "node"
   )

   puts telemetry.fetch("format") # => "onnx_webgpu_telemetry_v1"
   puts telemetry.fetch("selected_provider", "unknown")
   puts telemetry.fetch("requested_providers").inspect
   puts telemetry.fetch("fallback_used")
   puts telemetry.fetch("run_timings_ms").inspect

Notes
-----

- Smoke tests require Node.js.
- Real runtime checks require Playwright + ``onnxruntime-web`` in the harness
  environment.
- ``bundle exec rake deps:web`` installs/checks ``onnx`` and the web runtime
  dependencies used by real-runtime smoke checks.
- Harness export only accepts ``webgpu`` and ``wasm`` execution providers.
