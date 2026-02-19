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

For a step-by-step workflow guide, see :doc:`../onnx_webgpu/index`.

Onnx/WebGPU Support
-------------------

MLX Ruby supports an end-to-end browser export path:

1. Trace and export Graph IR via ``export_graph_ir``.
2. Validate payload/schema via ``validate_graph_ir``.
3. Preflight for WebGPU/ONNX conversion via
   ``graph_ir_webgpu_compatibility_report``.
4. Emit ONNX via ``export_onnx`` (or JSON stub via ``export_onnx_stub``).
5. Package browser harness assets via ``export_onnx_webgpu_harness``.
6. Run browser smoke verification via ``smoke_test_onnx_webgpu_harness``.

``graph_ir_webgpu_compatibility_report`` is the recommended conversion gate.
If ``unsupported_nodes`` is non-zero, the payload is not ready for ONNX/WebGPU
stub conversion.

Harness artifact output from ``export_onnx_webgpu_harness``:

- ``model.onnx``
- ``harness.manifest.json``
- ``inputs.example.json``
- ``index.html``
- ``harness.js``

The default harness provider order is ``["webgpu", "wasm"]``. Smoke telemetry
uses ``onnx_webgpu_telemetry_v1`` and includes provider selection/fallback and
``sample_outputs`` for parity assertions.

Runtime/tooling requirements:

- ``export_onnx`` requires ``python3`` with ``onnx`` importable.
- Real-runtime smoke tests require Node.js + Playwright + ``onnxruntime-web``.
- ``export_onnx_webgpu_harness`` only accepts ``webgpu`` and ``wasm`` execution
  providers.

Web demo generation is wired through ``bundle exec rake web:assets`` and emits:

- GPT-2 assets under ``web/assets/gpt2``
- nanoGPT assets under ``web/assets/nanogpt`` (trained artifacts required)
- Stable Diffusion assets under ``web/assets/stable_diffusion`` (text encoder,
  UNet, VAE decoder ONNX files)

Examples coverage/parity status:

- Current coverage/parity gates validate full examples export and ORT runtime
  parity across the benchmark model set.

Current ``export_onnx`` scope:

- Elementwise ops: ``Add``, ``Subtract``, ``Multiply``, ``Divide``, ``Maximum``,
  ``Minimum``, ``Power``.
- Unary/activation ops: ``Exp``, ``Log``, ``Sin``, ``Cos``, ``Erf``, ``Sqrt``,
  ``Abs``, ``Floor``, ``Negative``, ``Relu``, ``Sigmoid``, ``Tanh``.
- ``Square`` (lowered as ``Mul`` with identical inputs).
- ``Softmax`` (when exported as a direct ``Softmax`` node by MLX tracing).
- Type/compare/select ops: ``AsType`` (to ``Cast``), ``Greater``, ``Less``,
  ``Equal`` (with ``equal_nan=false``), and ``Select`` (to ``Where``).
- ``Full`` (current traced form) lowered as identity on broadcasted fill
  tensors.
- ``Matmul`` and ``AddMM`` (to ``Gemm``).
- ``Convolution`` (including traced ``conv1d``/``conv2d``/``conv3d`` and
  ``conv_general`` with ``flip == false``) lowered via layout transposes around
  ONNX ``Conv`` with mapped ``strides``/``pads``/``dilations``/``group``
  attributes.
- ``conv_transpose1d``/``conv_transpose2d``/``conv_transpose3d`` traces
  (exported as ``Convolution`` with ``flip == true``) lowered to ONNX
  ``ConvTranspose`` with derived ``pads``/``output_padding`` attributes.
- Shape ops: ``Transpose`` (perm attribute), ``Reshape``, ``Flatten``,
  ``Unflatten``, ``Squeeze``, ``ExpandDims`` (to ``Unsqueeze``), and
  ``Broadcast`` (to ``Expand``) using generated int64 initializer inputs for
  shape/axes.
- Indexing ops: ``Gather``, ``GatherAxis`` (to ``GatherElements``), ``Slice``,
  ``Split``, and ``AsStrided`` (current traced pattern to ``Gather``).
- ``Concatenate`` (to ``Concat``) when exported with explicit axis form
  ``arguments == [axis]``.
- ``Pad`` (constant mode).
- ``Scan`` for ``CumSum`` lowering.
- ``ScatterAxis`` (from ``put_along_axis``) to ONNX ``ScatterElements`` for
  update mode.
- Reductions via MLX ``Reduce`` code mapping:
  ``0/1`` (all/any) are lowered via cast decomposition
  ``Cast(BOOL) -> Cast(INT64) -> ReduceMin/ReduceMax -> Cast(BOOL)``.
  ``2 -> ReduceSum``, ``3 -> ReduceProd``, ``4 -> ReduceMin``,
  ``5 -> ReduceMax``.
- ``LogSumExp`` (to ``ReduceLogSumExp``) and ``ArgReduce`` (to
  ``ArgMin``/``ArgMax`` + cast).
- ``Arange`` lowered as ONNX initializer-backed constants.
- ``export_onnx`` supports optional ONNX external-data emission for initializers
  via ``external_data: true`` on path-like targets, with
  ``external_data_size_threshold`` and ``external_data_file`` controls.
- Constants/initializers are lowered for ``bool``/integer/float dtypes.
- ``complex64`` initializers are lowered via explicit JSON marker encoding in
  stubs and converted to ONNX ``COMPLEX64`` tensors during export.
- For JSON graph payloads, ``complex64`` constant leaves may be provided as
  marker objects ``{"__mlx_complex__": [real, imag]}`` or Ruby-style complex
  literal strings (for example ``"1.0+2.0i"``).

Known constraints/caveats:

- ``Convolution`` with ``flip == false`` and non-unit ``input_dilation`` is
  unsupported.
- ``Flatten`` requires known static input shape metadata.
- Some lowerings (for example ``Gather``, ``GatherAxis``, ``Pad``,
  ``LogSumExp``) require known static shapes from Graph IR metadata.
- ``Scan`` lowering currently supports CumSum-compatible ``reduce_type`` only.
- Harness input tensor building does not currently support ``complex64`` input
  tensors.
