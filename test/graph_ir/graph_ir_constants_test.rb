# frozen_string_literal: true

require "json"
require "stringio"
require_relative "test_helper"

class Phase286GraphIrConstantsParityTest < Minitest::Test
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

  def test_export_graph_ir_includes_constant_values
    fun = lambda do |x|
      constant = MLX::Core.array([2.0, 3.0], MLX::Core.float32)
      MLX::Core.add(x, constant)
    end
    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)

    payload = JSON.parse(MLX::GraphIR.export_graph_ir_json(fun, x))
    constants = payload.fetch("constants")
    assert_equal 1, constants.length

    constant = constants.first
    assert_equal "float32", constant.fetch("dtype")
    assert_equal [2], constant.fetch("shape")
    assert_equal [2.0, 3.0], constant.fetch("values")
    assert_includes payload.fetch("nodes").first.fetch("inputs"), constant.fetch("name")
  end

  def test_graph_ir_to_onnx_json_requires_constant_values
    payload = {
      "ir_version" => 1,
      "shapeless" => false,
      "inputs" => [{ "name" => "x", "shape" => [2], "dtype" => "float32" }],
      "keyword_inputs" => [],
      "outputs" => [{ "name" => "z", "shape" => [2], "dtype" => "float32" }],
      "constants" => [{ "name" => "c", "shape" => [2], "dtype" => "float32" }],
      "nodes" => [{ "op" => "Add", "inputs" => %w[x c], "outputs" => ["z"] }]
    }

    error = assert_raises(RuntimeError) { MLX::GraphIR.graph_ir_to_onnx_json(payload) }
    assert_match(/values/, error.message)
  end

  def test_graph_ir_to_onnx_rejects_constant_value_count_mismatch
    payload = {
      "ir_version" => 1,
      "shapeless" => false,
      "inputs" => [{ "name" => "x", "shape" => [2], "dtype" => "float32" }],
      "keyword_inputs" => [],
      "outputs" => [{ "name" => "z", "shape" => [2], "dtype" => "float32" }],
      "constants" => [{ "name" => "c", "shape" => [2], "dtype" => "float32", "values" => [1.0] }],
      "nodes" => [{ "op" => "Add", "inputs" => %w[x c], "outputs" => ["z"] }]
    }

    error = assert_raises(RuntimeError) do
      Dir.mktmpdir do |dir|
        MLX::GraphIR.graph_ir_to_onnx(File.join(dir, "broken.onnx"), payload)
      end
    end
    assert_match(/expected 2/, error.message)
  end
end
