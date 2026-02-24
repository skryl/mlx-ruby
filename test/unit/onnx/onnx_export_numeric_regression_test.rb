# frozen_string_literal: true

require "open3"
require "tmpdir"
require_relative "test_helper"

class Phase331OnnxExportNumericRegressionParityTest < Minitest::Test
  WRAPPED_NEGATIVE_ONE_UINT64 = 18_446_744_073_709_551_615

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

  def test_arange_with_float_arguments_is_stub_compatible
    payload = arange_float_payload

    report = JSON.parse(
      MLX::ONNX::Native.ir_compatibility_report_json(payload)
    )
    assert_equal 0, report.fetch("unsupported_nodes")
    assert_equal [], report.fetch("unsupported_ops")

    stub = TestSupport.parse_onnx_stub(payload)
    assert_equal [], stub.fetch("graph").fetch("nodes")

    initializer = initializer_by_name(stub, "range_out")
    assert_equal "int64", initializer.fetch("dtype")
    assert_equal [8], initializer.fetch("shape")
    assert_equal (0...8).to_a, initializer.fetch("values")
  end

  def test_export_onnx_normalizes_wrapped_int64_shape_values
    payload = wrapped_reshape_payload
    stub = TestSupport.parse_onnx_stub(payload)
    reshape = stub.fetch("graph").fetch("nodes").find { |node| node.fetch("op_type") == "Reshape" }
    refute_nil reshape

    shape_init = initializer_by_name(stub, reshape.fetch("inputs")[1])
    assert_equal "int64", shape_init.fetch("dtype")
    assert_equal [-1, 2], shape_init.fetch("values")

    skip "python onnx module is required for phase331 tests" unless python_module_available?("onnx")

    Dir.mktmpdir do |dir|
      onnx_path = File.join(dir, "wrapped_shape.onnx")
      written = TestSupport.export_onnx_from_graph_ir_source(onnx_path, payload, model_name: "phase331_wrapped_shape_case")
      assert_equal onnx_path, written
      assert_operator File.size(onnx_path), :>, 0
    end
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

  def initializer_by_name(stub, name)
    stub.fetch("graph").fetch("initializers").find { |init| init.fetch("name") == name }
  end

  def arange_float_payload
    {
      "ir_version" => 1,
      "shapeless" => false,
      "inputs" => [],
      "keyword_inputs" => [],
      "outputs" => [
        {"name" => "range_out", "shape" => [8], "dtype" => "int64"}
      ],
      "constants" => [],
      "nodes" => [
        {
          "op" => "Arange",
          "inputs" => [],
          "outputs" => ["range_out"],
          "arguments" => [0.0, 8.0, 1.0]
        }
      ]
    }
  end

  def wrapped_reshape_payload
    {
      "ir_version" => 1,
      "shapeless" => false,
      "inputs" => [
        {"name" => "x", "shape" => [4], "dtype" => "float32"}
      ],
      "keyword_inputs" => [],
      "outputs" => [
        {"name" => "y", "shape" => [2, 2], "dtype" => "float32"}
      ],
      "constants" => [],
      "nodes" => [
        {
          "op" => "Reshape",
          "inputs" => ["x"],
          "outputs" => ["y"],
          "arguments" => [[WRAPPED_NEGATIVE_ONE_UINT64, 2]]
        }
      ]
    }
  end
end
