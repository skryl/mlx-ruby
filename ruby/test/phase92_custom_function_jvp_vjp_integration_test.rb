# frozen_string_literal: true

require_relative "test_helper"

class Phase92CustomFunctionJvpVjpIntegrationTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_custom_function_jvp_override_is_used
    fun = MLX::Core.custom_function do |x|
      MLX::Core.square(x)
    end
    fun.jvp do |primals, _tangents|
      [MLX::Core.multiply(MLX::Core.ones_like(primals[0]), 5.0)]
    end

    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    t = MLX::Core.ones_like(x)
    outs, jvps = MLX::Core.jvp(fun, [x], [t])

    assert_equal [1.0, 4.0], outs[0].to_a
    assert_equal [5.0, 5.0], jvps[0].to_a
  end

  def test_custom_function_vjp_override_is_used
    fun = MLX::Core.custom_function do |x|
      MLX::Core.square(x)
    end
    fun.vjp do |_primals, cotangents, _outputs|
      [MLX::Core.multiply(cotangents[0], 9.0)]
    end

    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    t = MLX::Core.ones_like(x)
    outs, vjps = MLX::Core.vjp(fun, [x], [t])

    assert_equal [1.0, 4.0], outs[0].to_a
    assert_equal [9.0, 9.0], vjps[0].to_a
  end
end
