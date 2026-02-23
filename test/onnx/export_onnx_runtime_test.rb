# frozen_string_literal: true

require "json"
require "open3"
require "stringio"
require "tmpdir"
require_relative "test_helper"

class Phase287ExportOnnxRuntimeParityTest < Minitest::Test
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
      value = feeds[inp.name]
      runtime_inputs[inp.name] = np.array(value, dtype=np.float32)

    outputs = session.run(None, runtime_inputs)
    print(json.dumps([out.tolist() for out in outputs]))
  PY

  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
    @previous_device = MLX::Core.default_device
    MLX::Core.set_default_device(MLX::Core.cpu)
    skip "python onnx module is required for phase287 tests" unless python_module_available?("onnx")
    skip "python onnxruntime module is required for phase287 tests" unless python_module_available?("onnxruntime")
  end

  def teardown
    MLX::Core.set_default_device(@previous_device) if defined?(@previous_device) && @previous_device
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_exported_onnx_executes_with_expected_output_for_exp_add
    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    y = MLX::Core.array([3.0, 4.0], MLX::Core.float32)
    fun = lambda do |x_arg, y:|
      MLX::Core.add(MLX::Core.exp(x_arg), y)
    end
    payload = JSON.parse(MLX::ONNX.export_graph_ir_json(fun, x, y: y))
    expected = MLX::Core.add(MLX::Core.exp(x), y).to_a

    result = run_exported_onnx(payload, {
      payload.fetch("inputs")[0].fetch("name") => x.to_a,
      payload.fetch("inputs")[1].fetch("name") => y.to_a
    })
    assert_nested_close(expected, result.first)
  end

  def test_exported_onnx_executes_with_initializer_constants
    payload = {
      "ir_version" => 1,
      "shapeless" => false,
      "inputs" => [{ "name" => "x", "shape" => [2], "dtype" => "float32" }],
      "keyword_inputs" => [],
      "outputs" => [{ "name" => "z", "shape" => [2], "dtype" => "float32" }],
      "constants" => [{ "name" => "c", "shape" => [2], "dtype" => "float32", "values" => [10.0, 20.0] }],
      "nodes" => [{ "op" => "Add", "inputs" => %w[x c], "outputs" => ["z"] }]
    }

    x = [1.0, 2.0]
    result = run_exported_onnx(payload, { "x" => x })
    expected = [11.0, 22.0]
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

  def run_exported_onnx(payload, feeds)
    Dir.mktmpdir do |dir|
      onnx_path = File.join(dir, "graph.onnx")
      TestSupport.export_onnx_from_graph_ir_source(onnx_path, payload, model_name: "runtime_case")
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
    else
      assert_in_delta expected.to_f, actual.to_f, atol
    end
  end
end
