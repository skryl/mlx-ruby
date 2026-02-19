Graph IR To ONNX
================

MLX Ruby supports both JSON ONNX stubs and binary ONNX model export.

Generate ONNX stub JSON
-----------------------

Use ``graph_ir_to_onnx_stub`` for an in-memory stub hash, or
``export_onnx_stub`` to write JSON.

.. code-block:: ruby

   stub = MLX::Core.graph_ir_to_onnx_stub(payload, opset: 18, model_name: "demo_graph")
   MLX::Core.export_onnx_stub("artifacts/model_stub.json", payload, model_name: "demo_graph")

Export binary ONNX
------------------

Use ``export_onnx`` for ``.onnx`` output:

.. code-block:: ruby

   MLX::Core.export_onnx(
     "artifacts/model.onnx",
     payload,
     opset: 18,
     model_name: "demo_graph"
   )

External data mode
------------------

For large models, enable external initializer data:

.. code-block:: ruby

   MLX::Core.export_onnx(
     "artifacts/model.onnx",
     payload,
     model_name: "demo_graph",
     external_data: true,
     external_data_size_threshold: 1024,
     external_data_file: "model.data"
   )

This writes ``model.onnx`` plus ``model.data`` in the target directory.

Next step
---------

Continue with :doc:`webgpu_harness_and_smoke`.
