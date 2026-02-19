# frozen_string_literal: true

require "json"
require "open3"
require "stringio"
require "tmpdir"
require_relative "test_helper"

class Phase288OnnxRuntimeOpCoverageParityTest < Minitest::Test
  PY_RUN_ONNX_TYPED = <<~PY.freeze
    import json
    import sys

    import numpy as np
    import onnxruntime as ort

    model_path = sys.argv[1]
    feeds = json.loads(sys.argv[2])

    dtype_map = {
      "tensor(bool)": np.bool_,
      "tensor(uint8)": np.uint8,
      "tensor(uint16)": np.uint16,
      "tensor(uint32)": np.uint32,
      "tensor(uint64)": np.uint64,
      "tensor(int8)": np.int8,
      "tensor(int16)": np.int16,
      "tensor(int32)": np.int32,
      "tensor(int64)": np.int64,
      "tensor(float16)": np.float16,
      "tensor(float)": np.float32,
      "tensor(double)": np.float64,
    }

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    runtime_inputs = {}
    for inp in session.get_inputs():
      if inp.name not in feeds:
        raise RuntimeError(f"missing feed for input {inp.name!r}")
      dtype = dtype_map.get(inp.type)
      if dtype is None:
        raise RuntimeError(f"unsupported runtime input dtype: {inp.type!r}")
      runtime_inputs[inp.name] = np.array(feeds[inp.name], dtype=dtype)

    outputs = session.run(None, runtime_inputs)
    print(json.dumps([out.tolist() for out in outputs]))
  PY

  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
    @previous_device = MLX::Core.default_device
    MLX::Core.set_default_device(MLX::Core.cpu)
    skip "python onnx module is required for phase288 tests" unless python_module_available?("onnx")
    skip "python onnxruntime module is required for phase288 tests" unless python_module_available?("onnxruntime")
  end

  def teardown
    MLX::Core.set_default_device(@previous_device) if defined?(@previous_device) && @previous_device
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_runtime_parity_for_matmul
    x = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)
    y = MLX::Core.array([[5.0, 6.0], [7.0, 8.0]], MLX::Core.float32)
    fun = lambda do |x_arg, y:|
      MLX::Core.matmul(x_arg, y)
    end
    payload = JSON.parse(MLX::Core.export_graph_ir(StringIO.new, fun, x, y: y))
    expected = MLX::Core.matmul(x, y).to_a

    result = run_exported_onnx(payload, {
      payload.fetch("inputs")[0].fetch("name") => x.to_a,
      payload.fetch("inputs")[1].fetch("name") => y.to_a
    })
    assert_nested_close(expected, result.first)
  end

  def test_runtime_parity_for_int32_add_with_typed_feeds
    x = MLX::Core.array([1, 2, 3], MLX::Core.int32)
    y = MLX::Core.array([10, 20, 30], MLX::Core.int32)
    fun = lambda do |x_arg, y:|
      MLX::Core.add(x_arg, y)
    end
    payload = JSON.parse(MLX::Core.export_graph_ir(StringIO.new, fun, x, y: y))
    expected = MLX::Core.add(x, y).to_a

    result = run_exported_onnx(payload, {
      payload.fetch("inputs")[0].fetch("name") => x.to_a,
      payload.fetch("inputs")[1].fetch("name") => y.to_a
    })
    assert_nested_close(expected, result.first)
  end

  def test_runtime_parity_for_log_exp_chain
    x = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)
    fun = lambda do |x_arg|
      MLX::Core.log(MLX::Core.exp(x_arg))
    end
    payload = JSON.parse(MLX::Core.export_graph_ir(StringIO.new, fun, x))
    expected = fun.call(x).to_a

    result = run_exported_onnx(payload, {
      payload.fetch("inputs")[0].fetch("name") => x.to_a
    })
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

  def run_exported_onnx(payload, feeds)
    Dir.mktmpdir do |dir|
      onnx_path = File.join(dir, "graph.onnx")
      MLX::Core.export_onnx(onnx_path, payload, model_name: "op_coverage_case")
      out, err, status = Open3.capture3("python3", "-c", PY_RUN_ONNX_TYPED, onnx_path, JSON.generate(feeds))
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

    if expected.is_a?(Integer)
      assert_equal expected, actual
    else
      assert_in_delta expected.to_f, actual.to_f, atol
    end
  end
end
