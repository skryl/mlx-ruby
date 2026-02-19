# frozen_string_literal: true

require "json"
require "open3"
require "stringio"
require "tmpdir"
require_relative "test_helper"

class Phase294ConcatenateLoweringParityTest < Minitest::Test
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
    skip "python onnx module is required for phase294 tests" unless python_module_available?("onnx")
    skip "python onnxruntime module is required for phase294 tests" unless python_module_available?("onnxruntime")
  end

  def teardown
    MLX::Core.set_default_device(@previous_device) if defined?(@previous_device) && @previous_device
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_concatenate_axis_is_lowered_to_onnx_concat_attribute
    payload = concatenate_axis1_payload
    node = payload.fetch("nodes").first
    assert_equal "Concatenate", node.fetch("op")
    assert_equal [1], node.fetch("arguments")

    stub = MLX::Core.graph_ir_to_onnx_stub(payload)
    onnx_node = stub.fetch("graph").fetch("nodes").first
    assert_equal "Concat", onnx_node.fetch("op_type")
    assert_equal({ "axis" => 1 }, onnx_node.fetch("attributes"))
  end

  def test_stack_axis1_runtime_parity
    payload, x, y = stack_axis1_payload

    result = run_exported_onnx(payload, {
      payload.fetch("inputs")[0].fetch("name") => x,
      payload.fetch("inputs")[1].fetch("name") => y
    })
    expected = [
      [[1.0, 2.0], [5.0, 6.0]],
      [[3.0, 4.0], [7.0, 8.0]]
    ]
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

  def concatenate_axis1_payload
    fun = lambda do |x, y|
      MLX::Core.concatenate([x, y], 1)
    end
    x = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)
    y = MLX::Core.array([[5.0, 6.0], [7.0, 8.0]], MLX::Core.float32)
    JSON.parse(MLX::Core.export_graph_ir(StringIO.new, fun, x, y))
  end

  def stack_axis1_payload
    fun = lambda do |x, y|
      MLX::Core.stack([x, y], 1)
    end
    x = [[1.0, 2.0], [3.0, 4.0]]
    y = [[5.0, 6.0], [7.0, 8.0]]
    x_array = MLX::Core.array(x, MLX::Core.float32)
    y_array = MLX::Core.array(y, MLX::Core.float32)
    payload = JSON.parse(MLX::Core.export_graph_ir(StringIO.new, fun, x_array, y_array))
    [payload, x, y]
  end

  def run_exported_onnx(payload, feeds)
    Dir.mktmpdir do |dir|
      onnx_path = File.join(dir, "graph.onnx")
      MLX::Core.export_onnx(onnx_path, payload, model_name: "concatenate_case")
      out, err, status = Open3.capture3("python3", "-c", PY_RUN_ONNX, onnx_path, JSON.generate(feeds))
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
