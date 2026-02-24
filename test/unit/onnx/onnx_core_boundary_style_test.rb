# frozen_string_literal: true

require "json"
require_relative "test_helper"

class Phase410GraphIrCoreBoundaryStyleTest < Minitest::Test
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

  def test_public_core_header_does_not_expose_json_types
    header = File.read(File.join(RUBY_ROOT, "submodules", "mlx-onnx", "include", "mlx", "ir.hpp"))

    refute_match(/nlohmann/i, header)
    refute_match(/\bOrderedJson\b/, header)
    assert_match(/namespace mlx::onnx/, header)
  end

  def test_internal_json_header_contains_json_boundary_types
    header = File.read(File.join(RUBY_ROOT, "submodules", "mlx-onnx", "src", "json.hpp"))

    assert_match(/json\.hpp/, header)
    assert_match(/\bOrderedJson\b/, header)
  end

  def test_ir_api_parse_errors_are_tagged
    error = assert_raises(RuntimeError) do
      MLX::ONNX.graph_ir_to_onnx_json("{not-json", opset: 18, model_name: "broken")
    end

    assert_match(/\[ir\.api\]/, error.message)
  end

  def test_unsupported_lowering_errors_remain_not_implemented
    ir = {
      "ir_version" => 1,
      "shapeless" => false,
      "inputs" => [{ "name" => "x", "shape" => [2], "dtype" => "float32" }],
      "keyword_inputs" => [],
      "outputs" => [{ "name" => "y", "shape" => [2], "dtype" => "float32" }],
      "constants" => [],
      "nodes" => [
        {
          "op" => "FutureCustomFusionOp",
          "inputs" => ["x"],
          "outputs" => ["y"],
          "arguments" => []
        }
      ]
    }

    error = assert_raises(NotImplementedError) do
      MLX::ONNX.graph_ir_to_onnx_json(ir, opset: 18, model_name: "unsupported")
    end

    assert_match(/\[ir\.lowering\] unsupported/, error.message)
    assert_match(/FutureCustomFusionOp/, error.message)
  end

  def test_core_is_split_into_focused_lowering_and_onnx_modules
    core_dir = File.join(RUBY_ROOT, "submodules", "mlx-onnx", "src")

    expected_files = [
      "mappings.hpp",
      "mappings.cpp",
      "lowering.cpp",
      "onnx.cpp"
    ]

    expected_files.each do |name|
      path = File.join(core_dir, name)
      assert(File.exist?(path), "expected #{name} to exist")
      assert(File.size(path).to_i > 0, "expected #{name} to be non-empty")
    end
  end
end
