# frozen_string_literal: true

require "json"
require "open3"
require "stringio"
require "tmpdir"
require_relative "test_helper"

class Phase300SliceLoweringParityTest < Minitest::Test
  PY_RUN_ONNX = <<~PY.freeze
    import json
    import sys

    import numpy as np
    import onnxruntime as ort

    model_path = sys.argv[1]
    feeds = json.loads(sys.argv[2])
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    runtime_inputs = {}
    for inp in session.get_inputs():
      runtime_inputs[inp.name] = np.array(feeds[inp.name], dtype=np.float32)

    outputs = session.run(None, runtime_inputs)
    print(json.dumps([out.tolist() for out in outputs]))
  PY

  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
    @previous_device = MLX::Core.default_device
    MLX::Core.set_default_device(MLX::Core.cpu)
    skip "python onnx module is required for phase300 tests" unless python_module_available?("onnx")
    skip "python onnxruntime module is required for phase300 tests" unless python_module_available?("onnxruntime")
  end

  def teardown
    MLX::Core.set_default_device(@previous_device) if defined?(@previous_device) && @previous_device
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_slice_is_lowered_to_onnx_slice_with_shape_inputs
    payload = slice_payload
    assert_equal ["Slice"], payload.fetch("nodes").map { |node| node.fetch("op") }

    stub = MLX::Core.graph_ir_to_onnx_stub(payload)
    onnx_node = stub.fetch("graph").fetch("nodes").first
    assert_equal "Slice", onnx_node.fetch("op_type")
    assert_equal 5, onnx_node.fetch("inputs").length

    starts = initializer_by_name(stub, onnx_node.fetch("inputs")[1])
    assert_equal "int64", starts.fetch("dtype")
    assert_equal [2], starts.fetch("shape")
    assert_equal [1, 0], starts.fetch("values")

    ends = initializer_by_name(stub, onnx_node.fetch("inputs")[2])
    assert_equal "int64", ends.fetch("dtype")
    assert_equal [2], ends.fetch("shape")
    assert_equal [2, 3], ends.fetch("values")

    axes = initializer_by_name(stub, onnx_node.fetch("inputs")[3])
    assert_equal "int64", axes.fetch("dtype")
    assert_equal [2], axes.fetch("shape")
    assert_equal [0, 1], axes.fetch("values")

    steps = initializer_by_name(stub, onnx_node.fetch("inputs")[4])
    assert_equal "int64", steps.fetch("dtype")
    assert_equal [2], steps.fetch("shape")
    assert_equal [1, 1], steps.fetch("values")
  end

  def test_row_indexing_runtime_parity_via_slice_plus_squeeze
    payload, x = getitem_row_payload_with_values

    result = run_exported_onnx(payload, {
      payload.fetch("inputs")[0].fetch("name") => x
    })
    expected = [3.0, 4.0, 5.0]
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

  def slice_payload
    fun = lambda do |x|
      MLX::Core.slice(x, [1, 0], [2, 3], [1, 1])
    end
    x = MLX::Core.reshape(MLX::Core.arange(0, 9, 1, MLX::Core.float32), [3, 3])
    JSON.parse(MLX::Core.export_graph_ir(StringIO.new, fun, x))
  end

  def getitem_row_payload_with_values
    fun = lambda do |x|
      x[1]
    end
    x = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]
    x_array = MLX::Core.array(x, MLX::Core.float32)
    payload = JSON.parse(MLX::Core.export_graph_ir(StringIO.new, fun, x_array))
    [payload, x]
  end

  def run_exported_onnx(payload, feeds)
    Dir.mktmpdir do |dir|
      onnx_path = File.join(dir, "graph.onnx")
      MLX::Core.export_onnx(onnx_path, payload, model_name: "slice_case")
      out, err, status = Open3.capture3("python3", "-c", PY_RUN_ONNX, onnx_path, JSON.generate(feeds))
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
