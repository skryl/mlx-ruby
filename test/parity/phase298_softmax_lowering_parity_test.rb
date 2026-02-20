# frozen_string_literal: true

require "json"
require "open3"
require "stringio"
require "tmpdir"
require_relative "test_helper"

class Phase298SoftmaxLoweringParityTest < Minitest::Test
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
    skip "python onnx module is required for phase298 tests" unless python_module_available?("onnx")
    skip "python onnxruntime module is required for phase298 tests" unless python_module_available?("onnxruntime")
  end

  def teardown
    MLX::Core.set_default_device(@previous_device) if defined?(@previous_device) && @previous_device
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_softmax_node_is_lowered_to_onnx_softmax
    payload = softmax_axis1_payload
    assert_equal ["Softmax"], payload.fetch("nodes").map { |node| node.fetch("op") }

    stub = MLX::GraphIR.to_onnx_stub(payload)
    onnx_node = stub.fetch("graph").fetch("nodes").first
    assert_equal "Softmax", onnx_node.fetch("op_type")
    assert_equal({}, onnx_node.fetch("attributes"))
  end

  def test_softmax_runtime_parity
    payload, x = softmax_axis1_payload_with_values

    result = run_exported_onnx(payload, {
      payload.fetch("inputs")[0].fetch("name") => x
    })
    expected = [
      [0.09003057, 0.24472848, 0.66524094],
      [0.3006096, 0.332225, 0.3671654]
    ]
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

  def softmax_axis1_payload
    payload, = softmax_axis1_payload_with_values
    payload
  end

  def softmax_axis1_payload_with_values
    fun = lambda do |x|
      MLX::Core.softmax(x, 1)
    end

    x = [[1.0, 2.0, 3.0], [0.1, 0.2, 0.3]]
    x_array = MLX::Core.array(x, MLX::Core.float32)
    payload = JSON.parse(MLX::GraphIR.export_graph_ir_json(fun, x_array))
    [payload, x]
  end

  def run_exported_onnx(payload, feeds)
    Dir.mktmpdir do |dir|
      onnx_path = File.join(dir, "graph.onnx")
      TestSupport.export_onnx_from_graph_ir_source(onnx_path, payload, model_name: "softmax_case")
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
