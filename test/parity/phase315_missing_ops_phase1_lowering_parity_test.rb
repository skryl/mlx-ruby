# frozen_string_literal: true

require "json"
require "open3"
require "stringio"
require "tmpdir"
require_relative "test_helper"

class Phase315MissingOpsPhase1LoweringParityTest < Minitest::Test
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
  end

  def teardown
    MLX::Core.set_default_device(@previous_device) if defined?(@previous_device) && @previous_device
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_astype_payload_and_stub_lowering
    payload = astype_payload
    assert_equal ["AsType"], payload.fetch("nodes").map { |node| node.fetch("op") }

    stub = MLX::GraphIR.to_onnx_stub(payload)
    onnx_node = stub.fetch("graph").fetch("nodes").first
    assert_equal "Cast", onnx_node.fetch("op_type")
    assert_equal({ "to" => "FLOAT16" }, onnx_node.fetch("attributes"))
  end

  def test_sin_payload_and_stub_lowering
    payload = sin_payload
    assert_equal ["Sin"], payload.fetch("nodes").map { |node| node.fetch("op") }

    stub = MLX::GraphIR.to_onnx_stub(payload)
    onnx_node = stub.fetch("graph").fetch("nodes").first
    assert_equal "Sin", onnx_node.fetch("op_type")
    assert_equal({}, onnx_node.fetch("attributes"))
  end

  def test_cos_payload_and_stub_lowering
    payload = cos_payload
    assert_equal ["Cos"], payload.fetch("nodes").map { |node| node.fetch("op") }

    stub = MLX::GraphIR.to_onnx_stub(payload)
    onnx_node = stub.fetch("graph").fetch("nodes").first
    assert_equal "Cos", onnx_node.fetch("op_type")
    assert_equal({}, onnx_node.fetch("attributes"))
  end

  def test_erf_payload_and_stub_lowering
    payload = erf_payload
    assert_equal ["Erf"], payload.fetch("nodes").map { |node| node.fetch("op") }

    stub = MLX::GraphIR.to_onnx_stub(payload)
    onnx_node = stub.fetch("graph").fetch("nodes").first
    assert_equal "Erf", onnx_node.fetch("op_type")
    assert_equal({}, onnx_node.fetch("attributes"))
  end

  def test_less_payload_and_stub_lowering
    payload = less_payload
    assert_equal ["Less"], payload.fetch("nodes").map { |node| node.fetch("op") }

    stub = MLX::GraphIR.to_onnx_stub(payload)
    onnx_node = stub.fetch("graph").fetch("nodes").first
    assert_equal "Less", onnx_node.fetch("op_type")
    assert_equal({}, onnx_node.fetch("attributes"))
  end

  def test_astype_runtime_parity
    skip_unless_onnxruntime!
    payload, x, expected = astype_payload_with_values

    result = run_exported_onnx(payload, {
      payload.fetch("inputs")[0].fetch("name") => x
    })
    assert_nested_close(expected, result.first, atol: 1e-3)
  end

  def test_sin_runtime_parity
    skip_unless_onnxruntime!
    payload, x, expected = sin_payload_with_values

    result = run_exported_onnx(payload, {
      payload.fetch("inputs")[0].fetch("name") => x
    })
    assert_nested_close(expected, result.first, atol: 1e-5)
  end

  def test_cos_runtime_parity
    skip_unless_onnxruntime!
    payload, x, expected = cos_payload_with_values

    result = run_exported_onnx(payload, {
      payload.fetch("inputs")[0].fetch("name") => x
    })
    assert_nested_close(expected, result.first, atol: 1e-5)
  end

  def test_erf_runtime_parity
    skip_unless_onnxruntime!
    payload, x, expected = erf_payload_with_values

    result = run_exported_onnx(payload, {
      payload.fetch("inputs")[0].fetch("name") => x
    })
    assert_nested_close(expected, result.first, atol: 1e-5)
  end

  def test_less_runtime_parity
    skip_unless_onnxruntime!
    payload, x, y, expected = less_payload_with_values

    result = run_exported_onnx(payload, {
      payload.fetch("inputs")[0].fetch("name") => x,
      payload.fetch("inputs")[1].fetch("name") => y
    })
    assert_nested_close(expected, result.first)
  end

  private

  def skip_unless_onnxruntime!
    skip "python onnx module is required for phase315 tests" unless python_module_available?("onnx")
    skip "python onnxruntime module is required for phase315 tests" unless python_module_available?("onnxruntime")
  end

  def python_module_available?(name)
    _out, _err, status = Open3.capture3(
      "python3",
      "-c",
      "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('#{name}') else 1)"
    )
    status.success?
  end

  def astype_payload
    payload, = astype_payload_with_values
    payload
  end

  def astype_payload_with_values
    fun = lambda do |x|
      MLX::Core.astype(x, MLX::Core.float16)
    end

    x = [[1.0, -2.25], [3.5, 0.125]]
    x_array = MLX::Core.array(x, MLX::Core.float32)
    payload = JSON.parse(MLX::GraphIR.export_graph_ir_json(fun, x_array))
    expected = MLX::Core.astype(x_array, MLX::Core.float16).to_a
    [payload, x, expected]
  end

  def sin_payload
    payload, = sin_payload_with_values
    payload
  end

  def sin_payload_with_values
    fun = lambda do |x|
      MLX::Core.sin(x)
    end

    x = [[0.0, 0.5, 1.0], [-0.5, -1.0, 1.5]]
    x_array = MLX::Core.array(x, MLX::Core.float32)
    payload = JSON.parse(MLX::GraphIR.export_graph_ir_json(fun, x_array))
    expected = MLX::Core.sin(x_array).to_a
    [payload, x, expected]
  end

  def cos_payload
    payload, = cos_payload_with_values
    payload
  end

  def cos_payload_with_values
    fun = lambda do |x|
      MLX::Core.cos(x)
    end

    x = [[0.0, 0.5, 1.0], [-0.5, -1.0, 1.5]]
    x_array = MLX::Core.array(x, MLX::Core.float32)
    payload = JSON.parse(MLX::GraphIR.export_graph_ir_json(fun, x_array))
    expected = MLX::Core.cos(x_array).to_a
    [payload, x, expected]
  end

  def erf_payload
    payload, = erf_payload_with_values
    payload
  end

  def erf_payload_with_values
    fun = lambda do |x|
      MLX::Core.erf(x)
    end

    x = [[-1.0, -0.5, 0.0], [0.5, 1.0, 1.5]]
    x_array = MLX::Core.array(x, MLX::Core.float32)
    payload = JSON.parse(MLX::GraphIR.export_graph_ir_json(fun, x_array))
    expected = MLX::Core.erf(x_array).to_a
    [payload, x, expected]
  end

  def less_payload
    payload, = less_payload_with_values
    payload
  end

  def less_payload_with_values
    fun = lambda do |x, y|
      MLX::Core.less(x, y)
    end

    x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    y = [[1.5, 2.0, 2.5], [3.0, 5.0, 7.0]]
    x_array = MLX::Core.array(x, MLX::Core.float32)
    y_array = MLX::Core.array(y, MLX::Core.float32)
    payload = JSON.parse(MLX::GraphIR.export_graph_ir_json(fun, x_array, y_array))
    expected = MLX::Core.less(x_array, y_array).to_a
    [payload, x, y, expected]
  end

  def run_exported_onnx(payload, feeds)
    Dir.mktmpdir do |dir|
      onnx_path = File.join(dir, "graph.onnx")
      TestSupport.export_onnx_from_graph_ir_source(onnx_path, payload, model_name: "phase315_missing_ops_case")
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

    if expected == true || expected == false
      assert_equal expected, actual
    elsif expected.is_a?(Integer)
      assert_equal expected, actual
    else
      assert_in_delta expected.to_f, actual.to_f, atol
    end
  end
end
