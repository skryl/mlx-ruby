# frozen_string_literal: true

require "json"
require "open3"
require "stringio"
require "tmpdir"
require_relative "test_helper"

class Phase285ExportOnnxBinaryParityTest < Minitest::Test
  PY_ONNX_INSPECT = <<~PY.freeze
    import json
    import sys
    import onnx

    model = onnx.load(sys.argv[1])
    payload = {
      "graph_name": model.graph.name,
      "node_ops": [node.op_type for node in model.graph.node],
      "input_count": len(model.graph.input),
      "output_count": len(model.graph.output),
      "initializer_count": len(model.graph.initializer),
      "initializer_names": [init.name for init in model.graph.initializer],
    }
    print(json.dumps(payload))
  PY

  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
    @previous_device = MLX::Core.default_device
    MLX::Core.set_default_device(MLX::Core.cpu)
    skip "python onnx module is required for phase285 tests" unless python_onnx_available?
  end

  def teardown
    MLX::Core.set_default_device(@previous_device) if defined?(@previous_device) && @previous_device
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_export_onnx_supports_path_targets
    Dir.mktmpdir do |dir|
      onnx_path = File.join(dir, "graph.onnx")
      written = TestSupport.export_onnx_from_graph_ir_source(onnx_path, exported_payload, model_name: "exp_add")
      assert_equal onnx_path, written

      assert File.exist?(onnx_path)
      metadata = inspect_onnx(onnx_path)
      assert_equal "exp_add", metadata.fetch("graph_name")
      assert_equal %w[Exp Add], metadata.fetch("node_ops")
      assert_equal 2, metadata.fetch("input_count")
      assert_equal 1, metadata.fetch("output_count")
    end
  end

  def test_export_onnx_rejects_file_like_targets
    io = StringIO.new
    error = assert_raises(ArgumentError) do
      TestSupport.export_onnx_from_graph_ir_source(io, exported_payload, model_name: "exp_add")
    end
    assert_match("path-like target", error.message)
  end

  def test_export_onnx_lowers_constants_to_initializers
    payload = {
      "ir_version" => 1,
      "shapeless" => false,
      "inputs" => [{ "name" => "x", "shape" => [2], "dtype" => "float32" }],
      "keyword_inputs" => [],
      "outputs" => [{ "name" => "z", "shape" => [2], "dtype" => "float32" }],
      "constants" => [{ "name" => "c", "shape" => [2], "dtype" => "float32", "values" => [10.0, 20.0] }],
      "nodes" => [{ "op" => "Add", "inputs" => %w[x c], "outputs" => ["z"] }]
    }

    Dir.mktmpdir do |dir|
      onnx_path = File.join(dir, "graph_with_initializer.onnx")
      written = TestSupport.export_onnx_from_graph_ir_source(onnx_path, payload, model_name: "add_const")
      assert_equal onnx_path, written
      metadata = inspect_onnx(onnx_path)
      assert_equal 1, metadata.fetch("initializer_count")
      assert_equal ["c"], metadata.fetch("initializer_names")
      assert_equal %w[Add], metadata.fetch("node_ops")
    end
  end

  private

  def python_onnx_available?
    _out, _err, status = Open3.capture3(
      "python3",
      "-c",
      "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('onnx') else 1)"
    )
    status.success?
  end

  def inspect_onnx(path)
    out, err, status = Open3.capture3("python3", "-c", PY_ONNX_INSPECT, path)
    raise "failed to inspect onnx:\\n#{err}" unless status.success?

    JSON.parse(out)
  end

  def exported_payload
    fun = lambda do |x, y:|
      MLX::Core.add(MLX::Core.exp(x), y)
    end
    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    y = MLX::Core.array([3.0, 4.0], MLX::Core.float32)
    JSON.parse(MLX::GraphIR.export_graph_ir_json(fun, x, y: y))
  end
end
