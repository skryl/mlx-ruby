# frozen_string_literal: true

require "json"
require "stringio"
require_relative "test_helper"

class Phase281GraphIrValidationParityTest < Minitest::Test
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

  def test_validate_graph_ir_accepts_exported_payload
    payload = exported_payload
    validated = MLX::Core.validate_graph_ir(payload)
    assert_equal payload, validated
  end

  def test_validate_graph_ir_rejects_invalid_output_reference
    payload = exported_payload
    payload["outputs"][0]["name"] = "missing_tensor"

    error = assert_raises(ArgumentError) do
      MLX::Core.validate_graph_ir(payload)
    end
    assert_match(/outputs\[0\]/, error.message)
  end

  private

  def exported_payload
    fun = lambda do |x, y:|
      MLX::Core.add(MLX::Core.exp(x), y)
    end
    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    y = MLX::Core.array([3.0, 4.0], MLX::Core.float32)
    JSON.parse(MLX::Core.export_graph_ir(StringIO.new, fun, x, y: y))
  end
end
