# frozen_string_literal: true

require_relative "test_helper"

class Phase231AutogradStabilityParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_reduce_jvp_and_cumprod_grad
    x = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)
    t = MLX::Core.ones_like(x)

    outs, jvps = MLX::Core.jvp(->(v) { MLX::Core.sum(v) }, [x], [t])
    assert_in_delta 6.0, outs[0].to_a, 1e-6
    assert_in_delta 3.0, jvps[0].to_a, 1e-6

    grad = MLX::Core.grad(->(v) { MLX::Core.sum(MLX::Core.cumprod(v, 0)) }).call(x)
    assert_equal [9.0, 4.0, 2.0], grad.to_a
  end

  def test_repeated_grad_calls_are_stable
    x = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)
    grad_fn = MLX::Core.grad(->(v) { MLX::Core.sum(MLX::Core.square(v)) })

    g1 = grad_fn.call(x)
    g2 = grad_fn.call(x)

    assert_equal [2.0, 4.0, 6.0], g1.to_a
    assert_equal [2.0, 4.0, 6.0], g2.to_a
    assert_equal g1.shape, g2.shape
  end
end
