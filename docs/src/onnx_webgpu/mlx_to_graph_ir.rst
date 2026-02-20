MLX To Graph IR
===============

Use ``MLX::GraphIR.export_graph_ir_json`` to trace a function/module call and
emit Graph IR JSON.

Ownership boundary
------------------

- Public API: ``MLX::GraphIR.export_graph_ir_json``.
- Implementation module: ``MLX::GraphIR::Exporter``.
- ``MLX::GraphIR`` owns payload normalization/schema semantics.

Basic capture
-------------

``export_graph_ir_json`` returns a normalized JSON string.

.. code-block:: ruby

   require "json"
   require "mlx"

   mx = MLX::Core
   x = mx.array([[1.0, 2.0]], mx.float32)
   y = mx.array([[0.5, 0.25]], mx.float32)

   trace = ->(lhs, rhs) { MLX::Core.add(lhs, rhs) }
   payload_json = MLX::GraphIR.export_graph_ir_json(trace, x, y)
   payload = JSON.parse(payload_json)

   puts payload.fetch("ir_version")

Write to disk
-------------

Write the JSON artifact explicitly when needed:

.. code-block:: ruby

   graph_ir_path = "artifacts/graph_ir.json"
   File.binwrite(graph_ir_path, payload_json)

The emitted payload includes:

- ``inputs`` / ``keyword_inputs``
- ``outputs``
- ``constants``
- ``nodes``
- ``shapeless`` and ``ir_version`` metadata

Next step
---------

Continue with :doc:`validation_and_compatibility` before ONNX conversion.
