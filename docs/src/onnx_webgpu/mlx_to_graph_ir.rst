MLX To Graph IR
===============

Use ``MLX::Core.export_graph_ir`` to trace a function/module call and emit
Graph IR JSON.

Basic capture
-------------

Use an IO target when you want the JSON string in memory:

.. code-block:: ruby

   require "json"
   require "stringio"
   require "mlx"

   mx = MLX::Core
   x = mx.array([[1.0, 2.0]], mx.float32)
   y = mx.array([[0.5, 0.25]], mx.float32)

   trace = ->(lhs, rhs) { MLX::Core.add(lhs, rhs) }
   payload_json = MLX::Core.export_graph_ir(StringIO.new, trace, x, y)
   payload = JSON.parse(payload_json)

   puts payload.fetch("format", "mlxir_v1")

Write to disk
-------------

Use a file path target when you want reusable artifacts:

.. code-block:: ruby

   graph_ir_path = "artifacts/graph_ir.json"
   MLX::Core.export_graph_ir(graph_ir_path, trace, x, y)

The emitted payload includes:

- ``inputs`` / ``keyword_inputs``
- ``outputs``
- ``constants``
- ``nodes``
- ``shapeless`` and ``ir_version`` metadata

Next step
---------

Continue with :doc:`validation_and_compatibility` before ONNX conversion.
