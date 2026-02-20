# frozen_string_literal: true

require "json"
require "open3"
require "stringio"
require "tmpdir"
require_relative "test_helper"

class Phase317MissingOpsPhase3LoweringParityTest < Minitest::Test
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
  end

  def teardown
    MLX::Core.set_default_device(@previous_device) if defined?(@previous_device) && @previous_device
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_logsumexp_lowers_to_reduce_log_sum_exp_with_axes_input
    payload, = logsumexp_axis1_keepdims_payload_with_values
    assert_equal ["LogSumExp"], payload.fetch("nodes").map { |node| node.fetch("op") }

    stub = MLX::GraphIR.to_onnx_stub(payload)
    onnx_node = stub.fetch("graph").fetch("nodes").first
    assert_equal "ReduceLogSumExp", onnx_node.fetch("op_type")
    assert_equal 2, onnx_node.fetch("inputs").length
    assert_equal({ "keepdims" => 1 }, onnx_node.fetch("attributes"))

    axes = initializer_by_name(stub, onnx_node.fetch("inputs")[1])
    assert_equal "int64", axes.fetch("dtype")
    assert_equal [1], axes.fetch("shape")
    assert_equal [1], axes.fetch("values")
  end

  def test_logsumexp_runtime_parity_axis1_keepdims
    payload, x, expected = logsumexp_axis1_keepdims_payload_with_values

    result = run_exported_onnx(payload, {
      payload.fetch("inputs")[0].fetch("name") => x
    }, model_name: "logsumexp_axis1_keepdims_case")

    assert_nested_close(expected, result.first, atol: 1e-4)
  end

  def test_scan_cumsum_axis1_lowers_to_cumsum_with_axis_input
    payload, = cumsum_axis1_forward_payload_with_values
    assert_equal ["Scan"], payload.fetch("nodes").map { |node| node.fetch("op") }
    assert_equal [2, 1, false, true], payload.fetch("nodes").first.fetch("arguments")

    stub = MLX::GraphIR.to_onnx_stub(payload)
    onnx_node = stub.fetch("graph").fetch("nodes").first
    assert_equal "CumSum", onnx_node.fetch("op_type")
    assert_equal 2, onnx_node.fetch("inputs").length
    assert_equal({}, onnx_node.fetch("attributes"))

    axis = initializer_by_name(stub, onnx_node.fetch("inputs")[1])
    assert_equal "int64", axis.fetch("dtype")
    assert_equal [1], axis.fetch("shape")
    assert_equal [1], axis.fetch("values")
  end

  def test_scan_cumsum_reverse_exclusive_lowers_with_attributes
    payload, = cumsum_axis1_reverse_exclusive_payload_with_values
    assert_equal ["Scan"], payload.fetch("nodes").map { |node| node.fetch("op") }
    assert_equal [2, 1, true, false], payload.fetch("nodes").first.fetch("arguments")

    stub = MLX::GraphIR.to_onnx_stub(payload)
    onnx_node = stub.fetch("graph").fetch("nodes").first
    assert_equal "CumSum", onnx_node.fetch("op_type")
    assert_equal 2, onnx_node.fetch("inputs").length
    assert_equal({ "exclusive" => 1, "reverse" => 1 }, onnx_node.fetch("attributes"))

    axis = initializer_by_name(stub, onnx_node.fetch("inputs")[1])
    assert_equal "int64", axis.fetch("dtype")
    assert_equal [1], axis.fetch("shape")
    assert_equal [1], axis.fetch("values")
  end

  def test_scan_cumsum_runtime_parity_forward
    payload, x, expected = cumsum_axis1_forward_payload_with_values
    result = run_exported_onnx(payload, {
      payload.fetch("inputs")[0].fetch("name") => x
    }, model_name: "scan_cumsum_forward_case")

    assert_nested_close(expected, result.first, atol: 1e-4)
  end

  def test_scan_cumsum_runtime_parity_reverse_exclusive
    payload, x, expected = cumsum_axis1_reverse_exclusive_payload_with_values
    result = run_exported_onnx(payload, {
      payload.fetch("inputs")[0].fetch("name") => x
    }, model_name: "scan_cumsum_reverse_exclusive_case")

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

  def logsumexp_axis1_keepdims_payload_with_values
    fun = lambda do |x|
      MLX::Core.logsumexp(x, 1, true)
    end

    x = [
      [0.25, -0.75, 1.5],
      [2.0, 0.1, -0.2]
    ]
    x_array = MLX::Core.array(x, MLX::Core.float32)
    payload = JSON.parse(MLX::GraphIR.export_graph_ir_json(fun, x_array))
    expected = fun.call(x_array).to_a
    [payload, x, expected]
  end

  def cumsum_axis1_forward_payload_with_values
    fun = lambda do |x|
      MLX::Core.cumsum(x, 1, false, true)
    end

    x = [
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0]
    ]
    x_array = MLX::Core.array(x, MLX::Core.float32)
    payload = JSON.parse(MLX::GraphIR.export_graph_ir_json(fun, x_array))
    expected = fun.call(x_array).to_a
    [payload, x, expected]
  end

  def cumsum_axis1_reverse_exclusive_payload_with_values
    fun = lambda do |x|
      MLX::Core.cumsum(x, 1, true, false)
    end

    x = [
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0]
    ]
    x_array = MLX::Core.array(x, MLX::Core.float32)
    payload = JSON.parse(MLX::GraphIR.export_graph_ir_json(fun, x_array))
    expected = fun.call(x_array).to_a
    [payload, x, expected]
  end

  def run_exported_onnx(payload, feeds, model_name:)
    skip "python onnx module is required for phase317 runtime tests" unless python_module_available?("onnx")
    skip "python onnxruntime module is required for phase317 runtime tests" unless python_module_available?("onnxruntime")
    Dir.mktmpdir do |dir|
      onnx_path = File.join(dir, "graph.onnx")
      TestSupport.export_onnx_from_graph_ir_source(onnx_path, payload, model_name: model_name)
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
