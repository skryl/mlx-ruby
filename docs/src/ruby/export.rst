.. _export:

Export Functions
================

.. currentmodule:: mlx.core

.. autosummary::
  :toctree: _autosummary

   export_function
   import_function
   exporter
   export_to_dot
   export_graph_ir
   validate_graph_ir
   graph_ir_to_onnx_stub
   graph_ir_webgpu_compatibility_report
   export_onnx_stub
   export_onnx
   export_onnx_webgpu_harness
   smoke_test_onnx_webgpu_harness

Graph IR schema reference:

- ``docs/src/ruby/mlxir_v1.schema.json``

Current ``export_onnx`` scope:

- Elementwise ops: ``Add``, ``Subtract``, ``Multiply``, ``Divide``, ``Exp``,
  ``Log``, ``Sqrt``, ``Abs``, ``Negative``, ``Relu``, ``Sigmoid``, ``Tanh``,
  ``Maximum``, ``Minimum``, ``Power``.
- ``Softmax`` (when exported as a direct ``Softmax`` node by MLX tracing).
- Comparison/select ops: ``Greater`` and ``Select`` (to ONNX ``Where``).
- ``Full`` (current traced form) lowered as identity on broadcasted fill tensors.
- ``Matmul``.
- ``Convolution`` (including traced ``conv1d``/``conv2d``/``conv3d`` and
  ``conv_general`` with ``flip == false``) lowered via layout transposes around
  ONNX ``Conv`` with mapped ``strides``/``pads``/``dilations``/``group``
  attributes.
- ``conv_transpose1d``/``conv_transpose2d``/``conv_transpose3d`` traces (exported
  as ``Convolution`` with ``flip == true``) are lowered to ONNX
  ``ConvTranspose`` with derived ``pads``/``output_padding`` attributes.
- ``Convolution`` with ``flip == false`` and non-unit ``input_dilation`` remains
  unsupported.
- Shape ops: ``Transpose`` (perm attribute), ``Reshape``, ``Squeeze``,
  ``ExpandDims`` (to ONNX ``Unsqueeze``), and ``Broadcast`` (to ONNX ``Expand``)
  using generated int64 initializer inputs for shape/axes.
- ``Flatten`` (lowered to ONNX ``Reshape``) when the source tensor has a known
  static shape at lowering time.
- ``Flatten`` currently remains unsupported when lowering cannot infer the
  input tensor shape from the exported graph metadata.
- ``Concatenate`` (to ONNX ``Concat``) when exported with explicit axis form
  ``arguments == [axis]``.
- ``Gather`` (including ``take``-style traces) via ``Gather`` plus internal
  shape reordering/unsqueeze lowering to preserve MLX gather tensor semantics.
- ``Slice`` using ONNX ``Slice`` with generated int64
  ``starts``/``ends``/``axes``/``steps`` inputs.
- ``Split`` (including multi-output exports) with generated int64 split-length
  input and axis attribute lowering.
- ``ScatterAxis`` (from ``put_along_axis``) to ONNX ``ScatterElements`` for
  update mode.
- Reductions via MLX ``Reduce`` code mapping:
  ``0/1`` (all/any) are lowered via cast decomposition
  ``Cast(BOOL) -> Cast(INT64) -> ReduceMin/ReduceMax -> Cast(BOOL)``.
  ``2 -> ReduceSum``, ``3 -> ReduceProd``, ``4 -> ReduceMin``,
  ``5 -> ReduceMax``.
- Requires ``python3`` with ``onnx`` importable.
- ``export_onnx`` supports optional ONNX external-data emission for initializers
  via ``external_data: true`` on path-like targets, with
  ``external_data_size_threshold`` and ``external_data_file`` controls.
- ``export_onnx_webgpu_harness`` emits a browser harness directory that includes:
  ``model.onnx``, ``harness.manifest.json``, ``inputs.example.json``,
  ``index.html``, and ``harness.js``.
- ``smoke_test_onnx_webgpu_harness`` runs a Playwright Chromium smoke pass
  against an exported harness directory and returns parsed telemetry JSON.
  It supports deterministic mock mode (``mock_ort: true``) and real-runtime
  mode using local ``web/node_modules/onnxruntime-web`` assets
  (``local_ort: true``).
- Smoke telemetry includes ``sample_outputs`` from the measured run, enabling
  browser-runtime output parity checks for small model fixtures.
- Browser-facing harness templates are maintained under the top-level
  ``web/onnx_webgpu_harness`` directory.
- Constants/initializers are lowered for ``bool``/integer/float dtypes.
- ``complex64`` initializers are lowered via explicit JSON marker encoding
  in stubs and converted to ONNX ``COMPLEX64`` tensors during export.
- For JSON graph payloads, ``complex64`` constant leaves may be provided as
  marker objects ``{"__mlx_complex__": [real, imag]}`` or Ruby-style complex
  literal strings (for example ``"1.0+2.0i"``).
- Runtime parity tests can use ``onnxruntime`` (CPU execution provider).
- Runtime parity coverage includes both op-focused tests and small
  model-fixture exports (for example tiny MLP/conv/attention-style graphs).
