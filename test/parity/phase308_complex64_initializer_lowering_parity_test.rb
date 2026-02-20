# frozen_string_literal: true

require "json"
require "open3"
require "stringio"
require "tmpdir"
require_relative "test_helper"

class Phase308Complex64InitializerLoweringParityTest < Minitest::Test
  PY_ONNX_COMPLEX_INIT_INSPECT = <<~PY.freeze
    import json
    import sys
    import onnx

    model = onnx.load(sys.argv[1], load_external_data=False)
    init = model.graph.initializer[0]
    payload = {
      "name": init.name,
      "data_type": int(init.data_type),
      "complex64_type": int(onnx.TensorProto.COMPLEX64),
      "dims": [int(dim) for dim in init.dims],
      "float_data": [float(value) for value in init.float_data],
    }
    print(json.dumps(payload))
  PY

  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
    @previous_device = MLX::Core.default_device
    MLX::Core.set_default_device(MLX::Core.cpu)
    skip "python onnx module is required for phase308 tests" unless python_onnx_available?
  end

  def teardown
    MLX::Core.set_default_device(@previous_device) if defined?(@previous_device) && @previous_device
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_export_onnx_lowers_complex64_initializers
    payload = complex_initializer_payload

    Dir.mktmpdir do |dir|
      onnx_path = File.join(dir, "complex_init.onnx")
      assert_nil TestSupport.export_onnx_from_graph_ir_source(onnx_path, payload, model_name: "complex_init_case")
      metadata = inspect_complex_initializer(onnx_path)

      assert_equal "c", metadata.fetch("name")
      assert_equal metadata.fetch("complex64_type"), metadata.fetch("data_type")
      assert_equal [2], metadata.fetch("dims")
      assert_equal [1.0, 2.0, -3.5, 0.25], metadata.fetch("float_data")
    end
  end

  def test_export_onnx_stub_serializes_complex_values_with_markers
    payload = complex_initializer_payload
    content = TestSupport.export_onnx_json_dump(StringIO.new, payload, model_name: "complex_stub_case")
    stub = JSON.parse(content)
    initializer = stub.fetch("graph").fetch("initializers").first
    values = initializer.fetch("values")
    assert_equal({ "__mlx_complex__" => [1.0, 2.0] }, values[0])
    assert_equal({ "__mlx_complex__" => [-3.5, 0.25] }, values[1])
  end

  def test_export_onnx_accepts_json_payload_with_complex_markers
    payload = complex_initializer_payload
    payload.fetch("constants").first["values"] = [
      { "__mlx_complex__" => [1.0, 2.0] },
      { "__mlx_complex__" => [-3.5, 0.25] }
    ]

    Dir.mktmpdir do |dir|
      onnx_path = File.join(dir, "complex_marker_init.onnx")
      source = JSON.generate(payload)
      assert_nil TestSupport.export_onnx_from_graph_ir_source(onnx_path, source, model_name: "complex_marker_init_case")
      metadata = inspect_complex_initializer(onnx_path)
      assert_equal [1.0, 2.0, -3.5, 0.25], metadata.fetch("float_data")
    end
  end

  def test_export_onnx_accepts_json_payload_with_ruby_complex_literal_strings
    payload = complex_initializer_payload
    payload.fetch("constants").first["values"] = [
      "1.0+2.0i",
      "-3.5+0.25i"
    ]

    Dir.mktmpdir do |dir|
      onnx_path = File.join(dir, "complex_string_init.onnx")
      source = JSON.generate(payload)
      assert_nil TestSupport.export_onnx_from_graph_ir_source(onnx_path, source, model_name: "complex_string_init_case")
      metadata = inspect_complex_initializer(onnx_path)
      assert_equal [1.0, 2.0, -3.5, 0.25], metadata.fetch("float_data")
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

  def inspect_complex_initializer(path)
    out, err, status = Open3.capture3("python3", "-c", PY_ONNX_COMPLEX_INIT_INSPECT, path)
    raise "failed to inspect complex initializer:\n#{err}" unless status.success?

    JSON.parse(out)
  end

  def complex_initializer_payload
    {
      "ir_version" => 1,
      "shapeless" => false,
      "inputs" => [{ "name" => "x", "shape" => [2], "dtype" => "complex64" }],
      "keyword_inputs" => [],
      "outputs" => [{ "name" => "z", "shape" => [2], "dtype" => "complex64" }],
      "constants" => [
        {
          "name" => "c",
          "shape" => [2],
          "dtype" => "complex64",
          "values" => [Complex(1.0, 2.0), Complex(-3.5, 0.25)]
        }
      ],
      "nodes" => [{ "op" => "Add", "inputs" => %w[x c], "outputs" => ["z"] }]
    }
  end
end
