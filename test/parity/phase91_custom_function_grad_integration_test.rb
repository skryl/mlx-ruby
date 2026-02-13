# frozen_string_literal: true

require_relative "test_helper"

class Phase91CustomFunctionGradIntegrationTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_custom_function_vjp_overrides_grad_and_value_and_grad
    fun = MLX::Core.custom_function do |x|
      MLX::Core.sum(MLX::Core.square(x))
    end

    fun.vjp do |primals, _cotangents, _outputs|
      [MLX::Core.multiply(MLX::Core.ones_like(primals[0]), 7.0)]
    end

    x = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)

    grad_fn = MLX::Core.grad(fun)
    grads = grad_fn.call(x)
    assert_equal [7.0, 7.0, 7.0], grads.to_a

    vag_fn = MLX::Core.value_and_grad(fun)
    value, grads2 = vag_fn.call(x)
    assert_in_delta 14.0, value.to_a, 1e-5
    assert_equal [7.0, 7.0, 7.0], grads2.to_a
  end
end
