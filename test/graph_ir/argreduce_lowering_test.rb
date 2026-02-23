# frozen_string_literal: true

require "json"
require "open3"
require "stringio"
require "tmpdir"
require_relative "test_helper"

class Phase304ArgreduceLoweringParityTest < Minitest::Test
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
    skip "python onnx module is required for phase304 tests" unless python_module_available?("onnx")
    skip "python onnxruntime module is required for phase304 tests" unless python_module_available?("onnxruntime")
  end

  def teardown
    MLX::Core.set_default_device(@previous_device) if defined?(@previous_device) && @previous_device
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_argmax_axis1_lowers_argreduce_to_argmax_plus_cast
    payload = argmax_axis1_payload
    assert_equal %w[ArgReduce Squeeze], payload.fetch("nodes").map { |node| node.fetch("op") }

    stub = TestSupport.parse_onnx_stub(payload)
    types = stub.fetch("graph").fetch("nodes").map { |node| node.fetch("op_type") }
    assert_equal %w[ArgMax Cast Squeeze], types

    argmax = stub.fetch("graph").fetch("nodes")[0]
    assert_equal({ "axis" => 1, "keepdims" => 1 }, argmax.fetch("attributes"))

    cast = stub.fetch("graph").fetch("nodes")[1]
    assert_equal({ "to" => "UINT32" }, cast.fetch("attributes"))
  end

  def test_argmax_and_argmin_runtime_parity
    x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    argmax_payload = argmax_axis1_payload_with_values.first
    argmax_result = run_exported_onnx(argmax_payload, {
      argmax_payload.fetch("inputs")[0].fetch("name") => x
    })
    assert_equal [2, 2], argmax_result.first

    argmin_payload = argmin_axis0_payload_with_values.first
    argmin_result = run_exported_onnx(argmin_payload, {
      argmin_payload.fetch("inputs")[0].fetch("name") => x
    })
    assert_equal [0, 0, 0], argmin_result.first
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

  def argmax_axis1_payload
    argmax_axis1_payload_with_values.first
  end

  def argmax_axis1_payload_with_values
    fun = lambda do |x|
      MLX::Core.argmax(x, 1, false)
    end

    x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    x_array = MLX::Core.array(x, MLX::Core.float32)
    payload = JSON.parse(MLX::GraphIR.export_graph_ir_json(fun, x_array))
    [payload, x]
  end

  def argmin_axis0_payload_with_values
    fun = lambda do |x|
      MLX::Core.argmin(x, 0, false)
    end

    x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    x_array = MLX::Core.array(x, MLX::Core.float32)
    payload = JSON.parse(MLX::GraphIR.export_graph_ir_json(fun, x_array))
    [payload, x]
  end

  def run_exported_onnx(payload, feeds)
    Dir.mktmpdir do |dir|
      onnx_path = File.join(dir, "graph.onnx")
      TestSupport.export_onnx_from_graph_ir_source(onnx_path, payload, model_name: "argreduce_case")
      out, err, status = Open3.capture3("python3", "-c", PY_RUN_ONNX, onnx_path, JSON.generate(feeds))
      raise "onnxruntime execution failed:\n#{err}" unless status.success?

      return JSON.parse(out)
    end
  end
end
