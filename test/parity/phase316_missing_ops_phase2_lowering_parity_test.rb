# frozen_string_literal: true

require "json"
require "open3"
require "stringio"
require "tmpdir"
require_relative "test_helper"

class Phase316MissingOpsPhase2LoweringParityTest < Minitest::Test
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
    @onnx_available = python_module_available?("onnx")
    @onnxruntime_available = python_module_available?("onnxruntime")
  end

  def teardown
    MLX::Core.set_default_device(@previous_device) if defined?(@previous_device) && @previous_device
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_gather_axis_lowers_with_axis_attribute
    payload = gather_axis_payload
    assert_equal ["GatherAxis"], payload.fetch("nodes").map { |node| node.fetch("op") }
    assert_equal [1], payload.fetch("nodes").first.fetch("arguments")
    assert_equal [2, 3], payload.fetch("outputs").first.fetch("shape")

    stub = MLX::Core.graph_ir_to_onnx_stub(payload)
    onnx_node = stub.fetch("graph").fetch("nodes").first
    assert_equal "GatherElements", onnx_node.fetch("op_type")
    assert_equal({ "axis" => 1 }, onnx_node.fetch("attributes"))
  end

  def test_gather_axis_runtime_parity
    ensure_onnx_runtime!
    payload, x, idx = gather_axis_payload_with_values

    result = run_exported_onnx(payload, {
      payload.fetch("inputs")[0].fetch("name") => x,
      payload.fetch("inputs")[1].fetch("name") => idx
    })
    expected = [[30.0, 20.0, 10.0], [40.0, 60.0, 50.0]]
    assert_nested_close(expected, result.first)
  end

  def test_gather_axis_lowers_with_broadcasted_non_axis_dims
    payload = gather_axis_broadcast_payload
    gather_axis = payload.fetch("nodes").find { |node| node.fetch("op") == "GatherAxis" }
    assert_equal [1], gather_axis.fetch("arguments")

    stub = MLX::Core.graph_ir_to_onnx_stub(payload)
    graph_nodes = stub.fetch("graph").fetch("nodes")
    assert_equal "Expand", graph_nodes[-2].fetch("op_type")
    assert_equal "GatherElements", graph_nodes[-1].fetch("op_type")
    assert_equal({ "axis" => 1 }, graph_nodes[-1].fetch("attributes"))
  end

  def test_gather_axis_with_broadcast_runtime_parity
    ensure_onnx_runtime!
    payload, x, idx = gather_axis_broadcast_payload_with_values

    result = run_exported_onnx(payload, {
      payload.fetch("inputs")[0].fetch("name") => x,
      payload.fetch("inputs")[1].fetch("name") => idx
    })
    expected = [[20.0], [20.0]]
    assert_nested_close(expected, result.first)
  end

  def test_pad_constant_with_value_input_lowers_with_pads_tensor
    payload = pad_constant_payload
    assert_equal ["Pad"], payload.fetch("nodes").map { |node| node.fetch("op") }
    assert_equal [[0, 1], [1, 0], [0, 2]], payload.fetch("nodes").first.fetch("arguments")
    assert_equal [3, 4], payload.fetch("outputs").first.fetch("shape")

    stub = MLX::Core.graph_ir_to_onnx_stub(payload)
    onnx_node = stub.fetch("graph").fetch("nodes").first
    assert_equal "Pad", onnx_node.fetch("op_type")
    assert_equal 3, onnx_node.fetch("inputs").length
    assert_equal payload.fetch("inputs")[1].fetch("name"), onnx_node.fetch("inputs")[2]

    attrs = onnx_node.fetch("attributes")
    assert_equal "constant", attrs.fetch("mode", "constant")

    pads = initializer_by_name(stub, onnx_node.fetch("inputs")[1])
    assert_equal "int64", pads.fetch("dtype")
    assert_equal [4], pads.fetch("shape")
    assert_equal [1, 0, 0, 2], pads.fetch("values")
  end

  def test_pad_constant_with_value_input_runtime_parity
    ensure_onnx_runtime!
    payload, x, pad_value = pad_constant_payload_with_values

    result = run_exported_onnx(payload, {
      payload.fetch("inputs")[0].fetch("name") => x,
      payload.fetch("inputs")[1].fetch("name") => pad_value
    })
    expected = [[9.0, 9.0, 9.0, 9.0], [1.0, 2.0, 9.0, 9.0], [3.0, 4.0, 9.0, 9.0]]
    assert_nested_close(expected, result.first)
  end

  def test_equal_lowers_to_onnx_equal
    payload = equal_payload
    assert_equal %w[Broadcast Equal], payload.fetch("nodes").map { |node| node.fetch("op") }
    assert_equal [false], payload.fetch("nodes").last.fetch("arguments")
    assert_equal [2, 3], payload.fetch("outputs").first.fetch("shape")

    stub = MLX::Core.graph_ir_to_onnx_stub(payload)
    onnx_node = stub.fetch("graph").fetch("nodes").last
    assert_equal "Equal", onnx_node.fetch("op_type")
    assert_equal({}, onnx_node.fetch("attributes"))
  end

  def test_equal_runtime_parity
    ensure_onnx_runtime!
    payload, x, y = equal_payload_with_values

    result = run_exported_onnx(payload, {
      payload.fetch("inputs")[0].fetch("name") => x,
      payload.fetch("inputs")[1].fetch("name") => y
    })
    assert_equal [[true, false, true], [false, false, false]], result.first
  end

  def test_floor_lowers_to_onnx_floor
    payload = floor_payload
    assert_equal ["Floor"], payload.fetch("nodes").map { |node| node.fetch("op") }
    assert_equal [2, 3], payload.fetch("outputs").first.fetch("shape")

    stub = MLX::Core.graph_ir_to_onnx_stub(payload)
    onnx_node = stub.fetch("graph").fetch("nodes").first
    assert_equal "Floor", onnx_node.fetch("op_type")
    assert_equal({}, onnx_node.fetch("attributes"))
  end

  def test_floor_runtime_parity
    ensure_onnx_runtime!
    payload, x = floor_payload_with_values

    result = run_exported_onnx(payload, {
      payload.fetch("inputs")[0].fetch("name") => x
    })
    expected = [[-2.0, 0.0, 1.0], [2.0, -4.0, 4.0]]
    assert_nested_close(expected, result.first)
  end

  private

  def ensure_onnx_runtime!
    skip "python onnx module is required for phase316 runtime parity tests" unless @onnx_available
    skip "python onnxruntime module is required for phase316 runtime parity tests" unless @onnxruntime_available
  end

  def python_module_available?(name)
    _out, _err, status = Open3.capture3(
      "python3",
      "-c",
      "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('#{name}') else 1)"
    )
    status.success?
  end

  def gather_axis_payload
    payload, = gather_axis_payload_with_values
    payload
  end

  def gather_axis_payload_with_values
    fun = lambda do |x, indices|
      MLX::Core.take_along_axis(x, indices, 1)
    end

    x = [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]
    idx = [[2, 1, 0], [0, 2, 1]]
    x_array = MLX::Core.array(x, MLX::Core.float32)
    idx_array = MLX::Core.array(idx, MLX::Core.int32)
    payload = JSON.parse(MLX::Core.export_graph_ir(StringIO.new, fun, x_array, idx_array))
    [payload, x, idx]
  end

  def gather_axis_broadcast_payload
    payload, = gather_axis_broadcast_payload_with_values
    payload
  end

  def gather_axis_broadcast_payload_with_values
    x = [[10.0, 20.0]]
    idx = [[1], [1]]
    payload = {
      "ir_version" => 1,
      "shapeless" => false,
      "inputs" => [
        {"name" => "x", "shape" => [1, 2], "dtype" => "float32"},
        {"name" => "idx", "shape" => [2, 1], "dtype" => "int32"}
      ],
      "keyword_inputs" => [],
      "outputs" => [
        {"name" => "y", "shape" => [2, 1], "dtype" => "float32"}
      ],
      "constants" => [],
      "nodes" => [
        {
          "op" => "GatherAxis",
          "inputs" => ["x", "idx"],
          "outputs" => ["y"],
          "arguments" => [1]
        }
      ]
    }
    [payload, x, idx]
  end

  def pad_constant_payload
    payload, = pad_constant_payload_with_values
    payload
  end

  def pad_constant_payload_with_values
    fun = lambda do |x, pad_value|
      MLX::Core.pad(x, [[1, 0], [0, 2]], "constant", pad_value)
    end

    x = [[1.0, 2.0], [3.0, 4.0]]
    pad_value = 9.0
    x_array = MLX::Core.array(x, MLX::Core.float32)
    pad_value_array = MLX::Core.array(pad_value, MLX::Core.float32)
    payload = JSON.parse(MLX::Core.export_graph_ir(StringIO.new, fun, x_array, pad_value_array))
    [payload, x, pad_value]
  end

  def equal_payload
    payload, = equal_payload_with_values
    payload
  end

  def equal_payload_with_values
    fun = lambda do |x, y|
      MLX::Core.equal(x, y)
    end

    x = [[1.0, 2.0, 3.0], [2.0, 2.0, 0.0]]
    y = [[1.0, 0.0, 3.0]]
    x_array = MLX::Core.array(x, MLX::Core.float32)
    y_array = MLX::Core.array(y, MLX::Core.float32)
    payload = JSON.parse(MLX::Core.export_graph_ir(StringIO.new, fun, x_array, y_array))
    [payload, x, y]
  end

  def floor_payload
    payload, = floor_payload_with_values
    payload
  end

  def floor_payload_with_values
    fun = lambda do |x|
      MLX::Core.floor(x)
    end

    x = [[-1.2, 0.0, 1.7], [2.5, -3.9, 4.0]]
    x_array = MLX::Core.array(x, MLX::Core.float32)
    payload = JSON.parse(MLX::Core.export_graph_ir(StringIO.new, fun, x_array))
    [payload, x]
  end

  def run_exported_onnx(payload, feeds)
    Dir.mktmpdir do |dir|
      onnx_path = File.join(dir, "graph.onnx")
      MLX::Core.export_onnx(onnx_path, payload, model_name: "missing_ops_phase2_case")
      out, err, status = Open3.capture3("python3", "-c", PY_RUN_ONNX_TYPED, onnx_path, JSON.generate(feeds))
      raise "onnxruntime execution failed:\n#{err}" unless status.success?

      return JSON.parse(out)
    end
  end

  def initializer_by_name(stub, name)
    stub.fetch("graph").fetch("initializers").find { |init| init.fetch("name") == name }
  end

  def assert_nested_close(expected, actual, atol: 1e-5)
    if expected.is_a?(Array)
      assert_equal expected.length, actual.length
      expected.each_with_index do |item, index|
        assert_nested_close(item, actual[index], atol: atol)
      end
      return
    end

    assert_in_delta expected.to_f, actual.to_f, atol
  end
end
