# frozen_string_literal: true

require "json"
require "open3"
require "stringio"
require "tmpdir"
require_relative "test_helper"

class Phase302ScatterAxisLoweringParityTest < Minitest::Test
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
    skip "python onnx module is required for phase302 tests" unless python_module_available?("onnx")
    skip "python onnxruntime module is required for phase302 tests" unless python_module_available?("onnxruntime")
  end

  def teardown
    MLX::Core.set_default_device(@previous_device) if defined?(@previous_device) && @previous_device
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_scatter_axis_lowers_to_onnx_scatter_elements
    payload = put_axis1_payload
    assert_equal ["ScatterAxis"], payload.fetch("nodes").map { |node| node.fetch("op") }

    stub = TestSupport.parse_onnx_stub(payload)
    onnx_node = stub.fetch("graph").fetch("nodes").first
    assert_equal "ScatterElements", onnx_node.fetch("op_type")
    assert_equal({ "axis" => 1 }, onnx_node.fetch("attributes"))
  end

  def test_put_along_axis_runtime_parity
    payload, x, idx, vals = put_axis1_payload_with_values

    result = run_exported_onnx(payload, {
      payload.fetch("inputs")[0].fetch("name") => x,
      payload.fetch("inputs")[1].fetch("name") => idx,
      payload.fetch("inputs")[2].fetch("name") => vals
    })

    expected = [[2.0, 3.0, 1.0], [6.0, 5.0, 4.0]]
    assert_nested_close(expected, result.first)
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

  def put_axis1_payload
    payload, = put_axis1_payload_with_values
    payload
  end

  def put_axis1_payload_with_values
    fun = lambda do |x, indices, values|
      MLX::Core.put_along_axis(x, indices, values, 1)
    end

    x = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    idx = [[2, 0, 1], [2, 1, 0]]
    vals = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    x_array = MLX::Core.array(x, MLX::Core.float32)
    idx_array = MLX::Core.array(idx, MLX::Core.int32)
    vals_array = MLX::Core.array(vals, MLX::Core.float32)

    payload = JSON.parse(MLX::ONNX.export_graph_ir_json(fun, x_array, idx_array, vals_array))
    [payload, x, idx, vals]
  end

  def run_exported_onnx(payload, feeds)
    Dir.mktmpdir do |dir|
      onnx_path = File.join(dir, "graph.onnx")
      TestSupport.export_onnx_from_graph_ir_source(onnx_path, payload, model_name: "scatter_axis_case")
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

    assert_in_delta expected.to_f, actual.to_f, atol
  end
end
