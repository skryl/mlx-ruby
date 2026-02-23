# frozen_string_literal: true

require "json"
require "open3"
require "stringio"
require "tmpdir"
require_relative "test_helper"

class Phase301SplitLoweringParityTest < Minitest::Test
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
    skip "python onnx module is required for phase301 tests" unless python_module_available?("onnx")
    skip "python onnxruntime module is required for phase301 tests" unless python_module_available?("onnxruntime")
  end

  def teardown
    MLX::Core.set_default_device(@previous_device) if defined?(@previous_device) && @previous_device
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_equal_split_lowers_with_split_lengths_initializer
    payload = split_equal_payload
    assert_equal ["Split"], payload.fetch("nodes").map { |node| node.fetch("op") }

    stub = TestSupport.parse_onnx_stub(payload)
    split_node = stub.fetch("graph").fetch("nodes").first
    assert_equal "Split", split_node.fetch("op_type")
    assert_equal({ "axis" => 0 }, split_node.fetch("attributes"))
    assert_equal 2, split_node.fetch("inputs").length

    lengths = initializer_by_name(stub, split_node.fetch("inputs")[1])
    assert_equal "int64", lengths.fetch("dtype")
    assert_equal [2], lengths.fetch("shape")
    assert_equal [2, 2], lengths.fetch("values")
  end

  def test_index_split_runtime_parity_multi_output
    payload, x = split_indices_payload_with_values

    results = run_exported_onnx(payload, {
      payload.fetch("inputs")[0].fetch("name") => x
    })

    assert_equal [[[0.0, 1.0]], [[2.0, 3.0], [4.0, 5.0]], [[6.0, 7.0]]], results
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

  def split_equal_payload
    fun = lambda do |x|
      MLX::Core.split(x, 2, 0)
    end
    x = MLX::Core.reshape(MLX::Core.arange(0, 8, 1, MLX::Core.float32), [4, 2])
    JSON.parse(MLX::ONNX.export_graph_ir_json(fun, x))
  end

  def split_indices_payload_with_values
    fun = lambda do |x|
      MLX::Core.split(x, [1, 3], 0)
    end
    x = [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]
    x_array = MLX::Core.array(x, MLX::Core.float32)
    payload = JSON.parse(MLX::ONNX.export_graph_ir_json(fun, x_array))
    [payload, x]
  end

  def run_exported_onnx(payload, feeds)
    Dir.mktmpdir do |dir|
      onnx_path = File.join(dir, "graph.onnx")
      TestSupport.export_onnx_from_graph_ir_source(onnx_path, payload, model_name: "split_case")
      out, err, status = Open3.capture3("python3", "-c", PY_RUN_ONNX, onnx_path, JSON.generate(feeds))
      raise "onnxruntime execution failed:\n#{err}" unless status.success?

      return JSON.parse(out)
    end
  end

  def initializer_by_name(stub, name)
    stub.fetch("graph").fetch("initializers").find { |init| init.fetch("name") == name }
  end
end
