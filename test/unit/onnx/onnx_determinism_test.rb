# frozen_string_literal: true

require "stringio"
require_relative "test_helper"

class Phase280GraphIrDeterminismParityTest < Minitest::Test
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

  def test_export_ir_is_deterministic_for_same_trace_inputs
    fun = lambda do |x, y:|
      MLX::Core.add(MLX::Core.exp(x), y)
    end

    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    y = MLX::Core.array([3.0, 4.0], MLX::Core.float32)

    first = MLX::ONNX.export_graph_ir_json(fun, x, y: y)
    second = MLX::ONNX.export_graph_ir_json(fun, x, y: y)
    assert_equal first, second
  end
end
