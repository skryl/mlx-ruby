# frozen_string_literal: true

require "json"
require "open3"
require "stringio"
require "tmpdir"
require_relative "test_helper"

class Phase296ConcatenateDefaultFlattenLoweringParityTest < Minitest::Test
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
    skip "python onnx module is required for phase296 tests" unless python_module_available?("onnx")
    skip "python onnxruntime module is required for phase296 tests" unless python_module_available?("onnxruntime")
  end

  def teardown
    MLX::Core.set_default_device(@previous_device) if defined?(@previous_device) && @previous_device
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_default_concatenate_lowers_flatten_to_reshape_and_concat
    payload = concatenate_default_payload
    assert_equal %w[Flatten Flatten Concatenate], payload.fetch("nodes").map { |node| node.fetch("op") }

    stub = TestSupport.parse_onnx_stub(payload)
    nodes = stub.fetch("graph").fetch("nodes")
    assert_equal %w[Reshape Reshape Concat], nodes.map { |node| node.fetch("op_type") }

    reshape_0 = nodes[0]
    assert_equal 2, reshape_0.fetch("inputs").length
    shape_0 = initializer_by_name(stub, reshape_0.fetch("inputs")[1])
    assert_equal [1], shape_0.fetch("shape")
    assert_equal "int64", shape_0.fetch("dtype")
    assert_equal [4], shape_0.fetch("values")

    reshape_1 = nodes[1]
    assert_equal 2, reshape_1.fetch("inputs").length
    shape_1 = initializer_by_name(stub, reshape_1.fetch("inputs")[1])
    assert_equal [1], shape_1.fetch("shape")
    assert_equal "int64", shape_1.fetch("dtype")
    assert_equal [4], shape_1.fetch("values")

    concat = nodes[2]
    assert_equal({ "axis" => 0 }, concat.fetch("attributes"))
  end

  def test_default_concatenate_runtime_parity
    payload, x, y = concatenate_default_payload_with_values

    result = run_exported_onnx(payload, {
      payload.fetch("inputs")[0].fetch("name") => x,
      payload.fetch("inputs")[1].fetch("name") => y
    })

    expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
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

  def concatenate_default_payload
    payload, = concatenate_default_payload_with_values
    payload
  end

  def concatenate_default_payload_with_values
    fun = lambda do |x, y|
      MLX::Core.concatenate([x, y])
    end

    x = [[1.0, 2.0], [3.0, 4.0]]
    y = [[5.0, 6.0], [7.0, 8.0]]
    x_array = MLX::Core.array(x, MLX::Core.float32)
    y_array = MLX::Core.array(y, MLX::Core.float32)

    payload = JSON.parse(MLX::ONNX.export_graph_ir_json(fun, x_array, y_array))
    [payload, x, y]
  end

  def run_exported_onnx(payload, feeds)
    Dir.mktmpdir do |dir|
      onnx_path = File.join(dir, "graph.onnx")
      TestSupport.export_onnx_from_graph_ir_source(onnx_path, payload, model_name: "concatenate_default_case")
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
