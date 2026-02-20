# frozen_string_literal: true

require "json"
require "open3"
require "stringio"
require "tmpdir"
require_relative "test_helper"

class Phase312ModelFixtureOnnxRuntimeParityTest < Minitest::Test
  PY_RUN_ONNX_TYPED = <<~PY.freeze
    import json
    import sys

    import numpy as np
    import onnxruntime as ort

    model_path = sys.argv[1]
    feeds = json.loads(sys.argv[2])

    dtype_map = {
      "tensor(bool)": np.bool_,
      "tensor(uint8)": np.uint8,
      "tensor(uint16)": np.uint16,
      "tensor(uint32)": np.uint32,
      "tensor(uint64)": np.uint64,
      "tensor(int8)": np.int8,
      "tensor(int16)": np.int16,
      "tensor(int32)": np.int32,
      "tensor(int64)": np.int64,
      "tensor(float16)": np.float16,
      "tensor(float)": np.float32,
      "tensor(double)": np.float64,
    }

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    runtime_inputs = {}
    for inp in session.get_inputs():
      if inp.name not in feeds:
        raise RuntimeError(f"missing feed for input {inp.name!r}")
      dtype = dtype_map.get(inp.type)
      if dtype is None:
        raise RuntimeError(f"unsupported runtime input dtype: {inp.type!r}")
      runtime_inputs[inp.name] = np.array(feeds[inp.name], dtype=dtype)

    outputs = session.run(None, runtime_inputs)
    print(json.dumps([out.tolist() for out in outputs]))
  PY

  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
    @previous_device = MLX::Core.default_device
    MLX::Core.set_default_device(MLX::Core.cpu)
    skip "python onnx module is required for phase312 tests" unless python_module_available?("onnx")
    skip "python onnxruntime module is required for phase312 tests" unless python_module_available?("onnxruntime")
  end

  def teardown
    MLX::Core.set_default_device(@previous_device) if defined?(@previous_device) && @previous_device
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_tiny_mlp_fixture_runtime_parity
    payload, feeds, expected = tiny_mlp_fixture
    assert_operator payload.fetch("constants").length, :>=, 4

    result = run_exported_onnx(payload, feeds, model_name: "tiny_mlp_case")
    assert_nested_close(expected, result.first, atol: 1e-4)
  end

  def test_tiny_conv_head_fixture_runtime_parity
    payload, feeds, expected = tiny_conv_head_fixture
    assert_operator payload.fetch("constants").length, :>=, 3

    result = run_exported_onnx(payload, feeds, model_name: "tiny_conv_head_case")
    assert_nested_close(expected, result.first, atol: 1e-4)
  end

  def test_tiny_attention_fixture_runtime_parity
    payload, feeds, expected = tiny_attention_fixture
    assert_operator payload.fetch("constants").length, :>=, 3

    result = run_exported_onnx(payload, feeds, model_name: "tiny_attention_case")
    assert_nested_close(expected, result.first, atol: 1e-4)
  end

  private

  def python_module_available?(name)
    _out, _err, status = Open3.capture3(
      "python3",
      "-c",
      "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('#{name}') else 1)"
    )
    status.success?
  end

  def tiny_mlp_fixture
    w1 = MLX::Core.array(
      [
        [0.2, -0.4, 0.8, 0.1],
        [0.5, 0.3, -0.2, 0.7],
        [-0.1, 0.6, 0.4, -0.5]
      ],
      MLX::Core.float32
    )
    b1 = MLX::Core.array([0.1, -0.2, 0.3, 0.0], MLX::Core.float32)
    w2 = MLX::Core.array(
      [
        [0.5, -0.3],
        [0.2, 0.4],
        [-0.6, 0.1],
        [0.7, 0.8]
      ],
      MLX::Core.float32
    )
    b2 = MLX::Core.array([-0.25, 0.15], MLX::Core.float32)

    fun = lambda do |x|
      hidden = MLX::Core.add(MLX::Core.matmul(x, w1), b1)
      hidden = MLX::Core.maximum(hidden, 0.0)
      MLX::Core.add(MLX::Core.matmul(hidden, w2), b2)
    end

    x = MLX::Core.array(
      [
        [1.0, -2.0, 0.5],
        [0.25, 1.5, -1.0]
      ],
      MLX::Core.float32
    )

    payload = JSON.parse(MLX::GraphIR.export_graph_ir_json(fun, x))
    expected = fun.call(x).to_a
    feeds = { payload.fetch("inputs").first.fetch("name") => x.to_a }
    [payload, feeds, expected]
  end

  def tiny_conv_head_fixture
    conv_weight = MLX::Core.array(
      [
        [
          [[0.25], [-0.10]],
          [[0.40], [0.15]]
        ],
        [
          [[-0.35], [0.45]],
          [[0.20], [0.30]]
        ]
      ],
      MLX::Core.float32
    )
    head_weight = MLX::Core.array(
      [
        [0.10, -0.20, 0.30],
        [0.40, 0.15, -0.25],
        [-0.05, 0.35, 0.45],
        [0.50, -0.40, 0.20],
        [0.30, 0.25, -0.10],
        [-0.15, 0.05, 0.60],
        [0.22, -0.33, 0.11],
        [0.18, 0.09, -0.27]
      ],
      MLX::Core.float32
    )
    head_bias = MLX::Core.array([0.05, -0.10, 0.20], MLX::Core.float32)

    fun = lambda do |x|
      y = MLX::Core.conv2d(x, conv_weight)
      y = MLX::Core.maximum(y, 0.0)
      y = MLX::Core.reshape(y, [1, 8])
      MLX::Core.add(MLX::Core.matmul(y, head_weight), head_bias)
    end

    x = MLX::Core.array(
      [
        [
          [[1.0], [0.2], [-0.1]],
          [[0.5], [0.8], [0.3]],
          [[-0.4], [0.9], [0.7]]
        ]
      ],
      MLX::Core.float32
    )

    payload = JSON.parse(MLX::GraphIR.export_graph_ir_json(fun, x))
    expected = fun.call(x).to_a
    feeds = { payload.fetch("inputs").first.fetch("name") => x.to_a }
    [payload, feeds, expected]
  end

  def tiny_attention_fixture
    wq = MLX::Core.array(
      [
        [0.20, -0.10, 0.35, 0.40],
        [0.55, 0.25, -0.30, 0.15],
        [-0.45, 0.60, 0.05, -0.20]
      ],
      MLX::Core.float32
    )
    wk = MLX::Core.array(
      [
        [0.15, 0.45, -0.20, 0.10],
        [0.30, -0.55, 0.25, 0.35],
        [0.40, 0.05, 0.50, -0.15]
      ],
      MLX::Core.float32
    )
    wv = MLX::Core.array(
      [
        [0.10, -0.25],
        [0.35, 0.40],
        [-0.20, 0.50]
      ],
      MLX::Core.float32
    )

    fun = lambda do |x|
      q = MLX::Core.matmul(x, wq)
      k = MLX::Core.matmul(x, wk)
      v = MLX::Core.matmul(x, wv)
      logits = MLX::Core.matmul(q, MLX::Core.transpose(k, [1, 0]))
      probs = MLX::Core.softmax(logits, 1)
      MLX::Core.matmul(probs, v)
    end

    x = MLX::Core.array(
      [
        [0.4, -0.8, 1.2],
        [1.0, 0.3, -0.6]
      ],
      MLX::Core.float32
    )

    payload = JSON.parse(MLX::GraphIR.export_graph_ir_json(fun, x))
    expected = fun.call(x).to_a
    feeds = { payload.fetch("inputs").first.fetch("name") => x.to_a }
    [payload, feeds, expected]
  end

  def run_exported_onnx(payload, feeds, model_name:)
    Dir.mktmpdir do |dir|
      onnx_path = File.join(dir, "graph.onnx")
      TestSupport.export_onnx_from_graph_ir_source(onnx_path, payload, model_name: model_name)
      out, err, status = Open3.capture3("python3", "-c", PY_RUN_ONNX_TYPED, onnx_path, JSON.generate(feeds))
      raise "onnxruntime execution failed:\n#{err}" unless status.success?

      return JSON.parse(out)
    end
  end

  def assert_nested_close(expected, actual, atol: 1e-5)
    if expected.is_a?(Array)
      assert_equal expected.length, actual.length
      expected.each_with_index do |item, index|
        assert_nested_close(item, actual[index], atol: atol)
      end
      return
    end

    if expected.is_a?(Integer)
      assert_equal expected, actual
    else
      assert_in_delta expected.to_f, actual.to_f, atol
    end
  end
end
