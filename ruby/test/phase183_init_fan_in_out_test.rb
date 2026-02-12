# frozen_string_literal: true

require_relative "test_helper"

class Phase183InitFanInOutTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_calculate_fan_in_fan_out_for_linear_and_conv_shapes
    linear = MLX::Core.zeros([4, 8], MLX::Core.float32)
    fan_in, fan_out = MLX::NN::Init.calculate_fan_in_fan_out(linear)
    assert_equal 8, fan_in
    assert_equal 4, fan_out

    conv = MLX::Core.zeros([16, 3, 3, 8], MLX::Core.float32)
    fan_in2, fan_out2 = MLX::NN::Init.calculate_fan_in_fan_out(conv)
    assert_equal 72, fan_in2
    assert_equal 144, fan_out2
  end

  def test_calculate_fan_in_fan_out_validates_ndim
    vector = MLX::Core.zeros([8], MLX::Core.float32)
    assert_raises(ArgumentError) { MLX::NN::Init.calculate_fan_in_fan_out(vector) }
  end
end
