# frozen_string_literal: true

require "json"
require "open3"
require "stringio"
require "tmpdir"
require_relative "test_helper"

class Phase291ReduceShapeInputLoweringParityTest < Minitest::Test
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
    skip "python onnx module is required for phase291 tests" unless python_module_available?("onnx")
    skip "python onnxruntime module is required for phase291 tests" unless python_module_available?("onnxruntime")
  end

  def teardown
    MLX::Core.set_default_device(@previous_device) if defined?(@previous_device) && @previous_device
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_mean_axis1_stub_lowers_reduce_squeeze_broadcast_with_shape_initializers
    payload = mean_axis1_payload
    stub = MLX::GraphIR.graph_ir_to_onnx_payload(payload)
    nodes = stub.fetch("graph").fetch("nodes")

    assert_equal %w[ReduceSum Squeeze Expand Mul], nodes.map { |node| node.fetch("op_type") }

    reduce = nodes[0]
    assert_equal 2, reduce.fetch("inputs").length
    assert_equal({ "keepdims" => 1 }, reduce.fetch("attributes"))
    reduce_axes = initializer_by_name(stub, reduce.fetch("inputs")[1])
    assert_equal [1], reduce_axes.fetch("shape")
    assert_equal "int64", reduce_axes.fetch("dtype")
    assert_equal [1], reduce_axes.fetch("values")

    squeeze = nodes[1]
    assert_equal 2, squeeze.fetch("inputs").length
    squeeze_axes = initializer_by_name(stub, squeeze.fetch("inputs")[1])
    assert_equal [1], squeeze_axes.fetch("shape")
    assert_equal "int64", squeeze_axes.fetch("dtype")
    assert_equal [1], squeeze_axes.fetch("values")

    expand = nodes[2]
    assert_equal 2, expand.fetch("inputs").length
    expand_shape = initializer_by_name(stub, expand.fetch("inputs")[1])
    assert_equal [1], expand_shape.fetch("shape")
    assert_equal "int64", expand_shape.fetch("dtype")
    assert_equal [2], expand_shape.fetch("values")
  end

  def test_mean_axis1_runtime_parity
    payload = mean_axis1_payload
    x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    result = run_exported_onnx(payload, { payload.fetch("inputs").first.fetch("name") => x })
    expected = [2.0, 5.0]
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

  def mean_axis1_payload
    fun = lambda do |x|
      MLX::Core.mean(x, 1)
    end
    x = MLX::Core.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], MLX::Core.float32)
    JSON.parse(MLX::GraphIR.export_graph_ir_json(fun, x))
  end

  def run_exported_onnx(payload, feeds)
    Dir.mktmpdir do |dir|
      onnx_path = File.join(dir, "graph.onnx")
      TestSupport.export_onnx_from_graph_ir_source(onnx_path, payload, model_name: "reduce_shape_input_case")
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
