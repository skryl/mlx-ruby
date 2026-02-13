# frozen_string_literal: true

require_relative "test_helper"

class Phase228AutogradScatterExtremaTieParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_maximum_vjp_routes_gradient_to_selected_branch
    src = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)
    updates = MLX::Core.array([3.0, 2.0, 1.0], MLX::Core.float32)
    cotangent = MLX::Core.array([4.0, 5.0, 6.0], MLX::Core.float32)

    _, vjps = MLX::Core.vjp(->(x, u) { MLX::Core.maximum(x, u) }, [src, updates], [cotangent])
    assert_equal [0.0, 0.0, 6.0], vjps[0].to_a
    assert_equal [4.0, 5.0, 0.0], vjps[1].to_a
  end

  def test_minimum_vjp_routes_gradient_to_selected_branch
    src = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)
    updates = MLX::Core.array([3.0, 2.0, 1.0], MLX::Core.float32)
    cotangent = MLX::Core.array([4.0, 5.0, 6.0], MLX::Core.float32)

    _, vjps = MLX::Core.vjp(->(x, u) { MLX::Core.minimum(x, u) }, [src, updates], [cotangent])
    assert_equal [4.0, 0.0, 0.0], vjps[0].to_a
    assert_equal [0.0, 5.0, 6.0], vjps[1].to_a
  end
end
