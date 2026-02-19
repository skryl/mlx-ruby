# frozen_string_literal: true

require "json"
require "open3"
require "stringio"
require "tmpdir"
require_relative "test_helper"

class Phase293ReshapeUnsqueezeLoweringParityTest < Minitest::Test
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
    skip "python onnx module is required for phase293 tests" unless python_module_available?("onnx")
    skip "python onnxruntime module is required for phase293 tests" unless python_module_available?("onnxruntime")
  end

  def teardown
    MLX::Core.set_default_device(@previous_device) if defined?(@previous_device) && @previous_device
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_reshape_and_unsqueeze_use_shape_input_initializers
    payload = reshape_unsqueeze_payload
    stub = MLX::Core.graph_ir_to_onnx_stub(payload)
    nodes = stub.fetch("graph").fetch("nodes")

    assert_equal %w[Reshape Unsqueeze], nodes.map { |node| node.fetch("op_type") }

    reshape = nodes[0]
    assert_equal 2, reshape.fetch("inputs").length
    reshape_shape = initializer_by_name(stub, reshape.fetch("inputs")[1])
    assert_equal [2], reshape_shape.fetch("shape")
    assert_equal "int64", reshape_shape.fetch("dtype")
    assert_equal [3, 2], reshape_shape.fetch("values")

    unsqueeze = nodes[1]
    assert_equal 2, unsqueeze.fetch("inputs").length
    unsqueeze_axes = initializer_by_name(stub, unsqueeze.fetch("inputs")[1])
    assert_equal [1], unsqueeze_axes.fetch("shape")
    assert_equal "int64", unsqueeze_axes.fetch("dtype")
    assert_equal [0], unsqueeze_axes.fetch("values")
  end

  def test_reshape_unsqueeze_runtime_parity
    payload = reshape_unsqueeze_payload
    x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    result = run_exported_onnx(payload, { payload.fetch("inputs").first.fetch("name") => x })
    expected = [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]
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

  def reshape_unsqueeze_payload
    fun = lambda do |x|
      reshaped = MLX::Core.reshape(x, [3, 2])
      MLX::Core.expand_dims(reshaped, [0])
    end
    x = MLX::Core.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], MLX::Core.float32)
    JSON.parse(MLX::Core.export_graph_ir(StringIO.new, fun, x))
  end

  def run_exported_onnx(payload, feeds)
    Dir.mktmpdir do |dir|
      onnx_path = File.join(dir, "graph.onnx")
      MLX::Core.export_onnx(onnx_path, payload, model_name: "reshape_unsqueeze_case")
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
