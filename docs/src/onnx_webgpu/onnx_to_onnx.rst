Graph IR To ONNX
================

MLX Ruby supports native JSON ONNX export for debugging plus native binary ONNX
model assembly.

Ownership boundary
------------------

Use ``MLX::ONNX`` as the public API:

- ``MLX::ONNX.graph_ir_to_onnx``
- ``MLX::ONNX.graph_ir_to_onnx_json``
- ``MLX::ONNX.export_onnx``
- ``MLX::ONNX.export_onnx_json``

Implementation modules:

- ``MLX::ONNX::Native`` owns Graph IR/ONNX export and conversion runtime logic.

Generate ONNX JSON from Graph IR payload
----------------------------------------

Use ``graph_ir_to_onnx_json`` when you already have Graph IR payload/source.

.. code-block:: ruby

   onnx_json = MLX::ONNX.graph_ir_to_onnx_json(
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

   onnx_json = MLX::ONNX.export_onnx_json(
     trace,
     x,
     y,
     opset: 18,
     model_name: "demo_graph"
   )

Export binary ONNX
------------------

Use ``graph_ir_to_onnx`` when you already have Graph IR payload/source, or
``export_onnx`` for direct trace -> ONNX binary export.

.. code-block:: ruby

   MLX::ONNX.graph_ir_to_onnx("artifacts/model.onnx", payload)

.. code-block:: ruby

   MLX::ONNX.export_onnx(
     "artifacts/model.onnx",
     trace,
     x,
     y,
     opset: 18,
     model_name: "demo_graph"
   )

Both methods require a path-like target and return the written path.

External data mode
------------------

For large models, enable external initializer data:

.. code-block:: ruby

   MLX::ONNX.graph_ir_to_onnx(
     "artifacts/model.onnx",
     payload,
     external_data: true,
     external_data_size_threshold: 1024,
     external_data_file: "model.data"
   )

This writes ``model.onnx`` plus ``model.data`` in the target directory.

External-data notes:

- ``graph_ir_to_onnx`` and ``export_onnx`` require a path-like ``target``
  (not IO-like).
- ``external_data_size_threshold`` must be a non-negative integer.
- If ``external_data_file`` is omitted, the default is
  ``<target_basename>.data``.

Next step
---------

Continue with :doc:`webgpu_harness_and_smoke`.
