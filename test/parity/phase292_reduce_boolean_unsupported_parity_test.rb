# frozen_string_literal: true

require "json"
require "open3"
require "stringio"
require "tmpdir"
require_relative "test_helper"

class Phase292ReduceBooleanLoweringParityTest < Minitest::Test
  PY_RUN_ONNX_BOOL = <<~PY.freeze
    import json
    import sys

    import numpy as np
    import onnxruntime as ort

    model_path = sys.argv[1]
    feeds = json.loads(sys.argv[2])
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    runtime_inputs = {}
    for inp in session.get_inputs():
      runtime_inputs[inp.name] = np.array(feeds[inp.name], dtype=np.bool_)

    outputs = session.run(None, runtime_inputs)
    print(json.dumps([out.tolist() for out in outputs]))
  PY

  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
    @previous_device = MLX::Core.default_device
    MLX::Core.set_default_device(MLX::Core.cpu)
    skip "python onnx module is required for phase292 tests" unless python_module_available?("onnx")
    skip "python onnxruntime module is required for phase292 tests" unless python_module_available?("onnxruntime")
  end

  def teardown
    MLX::Core.set_default_device(@previous_device) if defined?(@previous_device) && @previous_device
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_any_axis1_stub_uses_bool_reduce_decomposition
    payload = reduce_payload(1) { |x| MLX::Core.any(x, 1, false) }
    stub = MLX::GraphIR.to_onnx_stub(payload)
    nodes = stub.fetch("graph").fetch("nodes")

    assert_equal %w[Cast Cast ReduceMax Cast Squeeze], nodes.map { |node| node.fetch("op_type") }
    assert_equal({ "to" => "BOOL" }, nodes[0].fetch("attributes"))
    assert_equal({ "to" => "INT64" }, nodes[1].fetch("attributes"))
    assert_equal({ "keepdims" => 1 }, nodes[2].fetch("attributes"))
    assert_equal({ "to" => "BOOL" }, nodes[3].fetch("attributes"))

    axes_name = nodes[2].fetch("inputs")[1]
    axes_init = initializer_by_name(stub, axes_name)
    assert_equal [1], axes_init.fetch("shape")
    assert_equal "int64", axes_init.fetch("dtype")
    assert_equal [1], axes_init.fetch("values")
  end

  def test_all_and_any_runtime_parity
    x = [[true, false, true], [true, true, true]]

    all_payload = reduce_payload(0) { |arr| MLX::Core.all(arr, 1, false) }
    all_result = run_exported_onnx_bool(all_payload, { all_payload.fetch("inputs")[0].fetch("name") => x })
    assert_equal [false, true], all_result.first

    any_payload = reduce_payload(1) { |arr| MLX::Core.any(arr, 1, false) }
    any_result = run_exported_onnx_bool(any_payload, { any_payload.fetch("inputs")[0].fetch("name") => x })
    assert_equal [true, true], any_result.first
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

  def reduce_payload(_code)
    fun = ->(x) { yield(x) }
    x = MLX::Core.array([[true, false, true], [true, true, true]], MLX::Core.bool_)
    JSON.parse(MLX::GraphIR.export_graph_ir_json(fun, x))
  end

  def run_exported_onnx_bool(payload, feeds)
    Dir.mktmpdir do |dir|
      onnx_path = File.join(dir, "graph.onnx")
      TestSupport.export_onnx_from_graph_ir_source(onnx_path, payload, model_name: "reduce_bool_case")
      out, err, status = Open3.capture3("python3", "-c", PY_RUN_ONNX_BOOL, onnx_path, JSON.generate(feeds))
      raise "onnxruntime execution failed:\n#{err}" unless status.success?

      return JSON.parse(out)
    end
  end

  def initializer_by_name(stub, name)
    stub.fetch("graph").fetch("initializers").find { |init| init.fetch("name") == name }
  end
end
