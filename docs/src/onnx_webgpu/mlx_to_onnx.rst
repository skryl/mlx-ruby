MLX To Graph IR
===============

Use ``MLX::ONNX.export_graph_ir`` to trace a function/module call and emit a
Graph IR Hash, or ``MLX::ONNX.export_graph_ir_json`` for debug JSON.

Ownership boundary
------------------

- Public API: ``MLX::ONNX.export_graph_ir`` and
  ``MLX::ONNX.export_graph_ir_json``.
- Runtime implementation module: ``MLX::ONNX::Native``.

Basic capture
-------------

``export_graph_ir`` returns a normalized Hash payload.

.. code-block:: ruby

   require "json"
   require "mlx"

   mx = MLX::Core
   x = mx.array([[1.0, 2.0]], mx.float32)
   y = mx.array([[0.5, 0.25]], mx.float32)

   trace = ->(lhs, rhs) { MLX::Core.add(lhs, rhs) }
   payload = MLX::ONNX.export_graph_ir(trace, x, y)

   puts payload.fetch("ir_version")

Debug JSON export
-----------------

Write a JSON artifact explicitly when needed:

.. code-block:: ruby

   payload_json = MLX::ONNX.export_graph_ir_json(trace, x, y)
   ir_path = "artifacts/ir.json"
   File.binwrite(ir_path, payload_json)

The emitted payload includes:

- ``inputs`` / ``keyword_inputs``
- ``outputs``
- ``constants``
- ``nodes``
- ``shapeless`` and ``ir_version`` metadata

Next step
---------

Continue with :doc:`validation_and_compatibility` before ONNX conversion.
