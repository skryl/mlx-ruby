# frozen_string_literal: true

require "json"
require "open3"
require "stringio"
require "tmpdir"
require_relative "test_helper"

class Phase290TransposeAttributeLoweringParityTest < Minitest::Test
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
    skip "python onnx module is required for phase290 tests" unless python_module_available?("onnx")
    skip "python onnxruntime module is required for phase290 tests" unless python_module_available?("onnxruntime")
  end

  def teardown
    MLX::Core.set_default_device(@previous_device) if defined?(@previous_device) && @previous_device
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_transpose_arguments_are_exported_and_lowered_to_onnx_attributes
    payload = transpose_payload
    node = payload.fetch("nodes").first
    assert_equal "Transpose", node.fetch("op")
    assert_equal [[1, 0]], node.fetch("arguments")

    stub = MLX::GraphIR.graph_ir_to_onnx_payload(payload)
    attrs = stub.fetch("graph").fetch("nodes").first.fetch("attributes")
    assert_equal [1, 0], attrs.fetch("perm")
  end

  def test_transpose_runtime_parity
    payload = transpose_payload
    x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    result = run_exported_onnx(payload, { payload.fetch("inputs").first.fetch("name") => x })
    expected = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    assert_equal expected, result.first
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

  def transpose_payload
    fun = lambda do |x|
      MLX::Core.transpose(x, [1, 0])
    end
    x = MLX::Core.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], MLX::Core.float32)
    JSON.parse(MLX::GraphIR.export_graph_ir_json(fun, x))
  end

  def run_exported_onnx(payload, feeds)
    Dir.mktmpdir do |dir|
      onnx_path = File.join(dir, "graph.onnx")
      TestSupport.export_onnx_from_graph_ir_source(onnx_path, payload, model_name: "transpose_case")
      out, err, status = Open3.capture3("python3", "-c", PY_RUN_ONNX, onnx_path, JSON.generate(feeds))
      raise "onnxruntime execution failed:\n#{err}" unless status.success?

      return JSON.parse(out)
    end
  end
end
