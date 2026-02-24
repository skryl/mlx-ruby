# frozen_string_literal: true

require "json"
require "open3"
require "stringio"
require "tmpdir"
require_relative "test_helper"

class Phase307ExportOnnxExternalDataParityTest < Minitest::Test
  PY_ONNX_EXTERNAL_INSPECT = <<~PY.freeze
    import json
    import sys
    import onnx

    model = onnx.load(sys.argv[1], load_external_data=False)
    initializer = model.graph.initializer[0]
    payload = {
      "data_location": int(initializer.data_location),
      "external_data": {entry.key: entry.value for entry in initializer.external_data},
    }
    print(json.dumps(payload))
  PY

  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
    @previous_device = MLX::Core.default_device
    MLX::Core.set_default_device(MLX::Core.cpu)
    skip "python onnx module is required for phase307 tests" unless python_onnx_available?
  end

  def teardown
    MLX::Core.set_default_device(@previous_device) if defined?(@previous_device) && @previous_device
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_export_onnx_can_emit_external_data_for_path_targets
    payload = constant_payload

    Dir.mktmpdir do |dir|
      onnx_path = File.join(dir, "graph.onnx")
      written = TestSupport.export_onnx_from_graph_ir_source(
        onnx_path,
        payload,
        model_name: "ext_data_case",
        external_data: true,
        external_data_size_threshold: 0
      )
      assert_equal onnx_path, written

      external_path = File.join(dir, "graph.data")
      assert File.exist?(onnx_path)
      assert File.exist?(external_path)
      assert_operator File.size(external_path), :>, 0

      metadata = inspect_external_data(onnx_path)
      assert_equal 1, metadata.fetch("data_location")
      assert_equal "graph.data", metadata.fetch("external_data").fetch("location")
    end
  end

  def test_export_onnx_external_data_accepts_custom_filename
    payload = constant_payload

    Dir.mktmpdir do |dir|
      onnx_path = File.join(dir, "custom_name.onnx")
      written = TestSupport.export_onnx_from_graph_ir_source(
        onnx_path,
        payload,
        model_name: "ext_data_custom",
        external_data: true,
        external_data_size_threshold: 0,
        external_data_file: "weights.bin"
      )
      assert_equal onnx_path, written

      external_path = File.join(dir, "weights.bin")
      assert File.exist?(onnx_path)
      assert File.exist?(external_path)
      metadata = inspect_external_data(onnx_path)
      assert_equal "weights.bin", metadata.fetch("external_data").fetch("location")
    end
  end

  def test_export_onnx_external_data_rejects_io_targets
    payload = constant_payload
    error = assert_raises(ArgumentError) do
      TestSupport.export_onnx_from_graph_ir_source(
        StringIO.new,
        payload,
        model_name: "ext_data_io",
        external_data: true
      )
    end
    assert_match("path-like target", error.message)
  end

  def test_export_onnx_rejects_io_targets_even_without_external_data
    payload = constant_payload
    error = assert_raises(ArgumentError) do
      TestSupport.export_onnx_from_graph_ir_source(
        StringIO.new,
        payload,
        model_name: "path_only_case",
        external_data: false
      )
    end
    assert_match("path-like target", error.message)
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

  def inspect_external_data(path)
    out, err, status = Open3.capture3("python3", "-c", PY_ONNX_EXTERNAL_INSPECT, path)
    raise "failed to inspect onnx external data:\n#{err}" unless status.success?

    JSON.parse(out)
  end

  def constant_payload
    values = Array.new(4096) { |index| index.to_f / 10.0 }
    {
      "ir_version" => 1,
      "shapeless" => false,
      "inputs" => [{ "name" => "x", "shape" => [4096], "dtype" => "float32" }],
      "keyword_inputs" => [],
      "outputs" => [{ "name" => "z", "shape" => [4096], "dtype" => "float32" }],
      "constants" => [{ "name" => "c", "shape" => [4096], "dtype" => "float32", "values" => values }],
      "nodes" => [{ "op" => "Add", "inputs" => %w[x c], "outputs" => ["z"] }]
    }
  end
end
