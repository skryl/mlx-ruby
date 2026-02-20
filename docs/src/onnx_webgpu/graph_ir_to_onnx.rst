Graph IR To ONNX
================

MLX Ruby supports JSON ONNX export plus binary ONNX model assembly.

Ownership boundary
------------------

Use ``MLX::GraphIR`` as the public API:

- ``MLX::GraphIR.to_onnx_stub``
- ``MLX::GraphIR.graph_ir_to_onnx_json``
- ``MLX::GraphIR.export_onnx_json``
- ``MLX::GraphIR.onnx_json_to_onnx``

Implementation modules:

- ``MLX::GraphIR::ONNX::Exporter`` handles JSON and binary export bridging.
- ``MLX::GraphIR::ONNX::PythonBuilder`` builds binary ONNX via Python ``onnx``.

Generate ONNX JSON from Graph IR payload
----------------------------------------

Use ``graph_ir_to_onnx_json`` when you already have Graph IR payload/source.

.. code-block:: ruby

   onnx_json = MLX::GraphIR.graph_ir_to_onnx_json(
     payload,
     opset: 18,
     model_name: "demo_graph"
   )

   File.binwrite(
     "artifacts/model_stub.json",
     JSON.pretty_generate(JSON.parse(onnx_json))
   )

Generate ONNX JSON directly from trace
--------------------------------------

Use ``export_onnx_json`` to capture Graph IR + lower to ONNX JSON in one call.

.. code-block:: ruby

   onnx_json = MLX::GraphIR.export_onnx_json(
     trace,
     x,
     y,
     opset: 18,
     model_name: "demo_graph"
   )

Export binary ONNX
------------------

Use ``onnx_json_to_onnx`` for ``.onnx`` output.

.. code-block:: ruby

   MLX::GraphIR.onnx_json_to_onnx("artifacts/model.onnx", onnx_json)

``MLX::GraphIR.onnx_json_to_onnx`` behavior:

- IO-like target: returns ``.onnx`` bytes and writes to the IO.
- Path-like target: writes the file and returns ``nil``.

External data mode
------------------

For large models, enable external initializer data:

.. code-block:: ruby

   MLX::GraphIR.onnx_json_to_onnx(
     "artifacts/model.onnx",
     onnx_json,
     external_data: true,
     external_data_size_threshold: 1024,
     external_data_file: "model.data"
   )

This writes ``model.onnx`` plus ``model.data`` in the target directory.

External-data notes:

- ``external_data: true`` requires a path-like ``target`` (not IO-like).
- ``external_data_size_threshold`` must be a non-negative integer.
- If ``external_data_file`` is omitted, the default is
  ``<target_basename>.data``.

Next step
---------

Continue with :doc:`webgpu_harness_and_smoke`.
