# frozen_string_literal: true

require_relative "test_helper"

class Phase13ElementwiseClipTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_minimum_and_maximum_with_broadcast
    a = MLX::Core.array([[1.0, 5.0], [7.0, 2.0]], MLX::Core.float32)
    b = MLX::Core.array([3.0, 4.0], MLX::Core.float32)

    assert_equal [[1.0, 4.0], [3.0, 2.0]], MLX::Core.minimum(a, b).to_a
    assert_equal [[3.0, 5.0], [7.0, 4.0]], MLX::Core.maximum(a, b).to_a
  end

  def test_floor_and_ceil
    x = MLX::Core.array([-1.2, -0.1, 0.0, 1.1, 1.9], MLX::Core.float32)

    assert_equal [-2.0, -1.0, 0.0, 1.0, 1.0], MLX::Core.floor(x).to_a
    assert_equal [-1.0, 0.0, 0.0, 2.0, 2.0], MLX::Core.ceil(x).to_a
  end

  def test_clip_with_optional_bounds
    x = MLX::Core.array([-1.0, 0.0, 1.0, 2.0, 3.0], MLX::Core.float32)

    assert_equal [0.0, 0.0, 1.0, 2.0, 2.0], MLX::Core.clip(x, 0.0, 2.0).to_a
    assert_equal [-1.0, 0.0, 1.0, 1.5, 1.5], MLX::Core.clip(x, nil, 1.5).to_a
    assert_equal [1.0, 1.0, 1.0, 2.0, 3.0], MLX::Core.clip(x, 1.0, nil).to_a
  end
end
