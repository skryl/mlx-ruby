# frozen_string_literal: true

require_relative "test_helper"

class Phase22MinMaxArgReductionsTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_min_and_max_reductions
    x = MLX::Core.array([[3.0, 1.0, 2.0], [0.0, -1.0, 4.0]], MLX::Core.float32)

    assert_in_delta 4.0, MLX::Core.max(x).to_a, 1e-5
    assert_in_delta(-1.0, MLX::Core.min(x).to_a, 1e-5)
    assert_equal [3.0, 4.0], MLX::Core.max(x, 1).to_a
    assert_equal [0.0, -1.0, 2.0], MLX::Core.min(x, 0).to_a
  end

  def test_argmin_and_argmax_reductions
    x = MLX::Core.array([[3.0, 1.0, 2.0], [0.0, -1.0, 4.0]], MLX::Core.float32)

    assert_equal 5, MLX::Core.argmax(x).to_a
    assert_equal 4, MLX::Core.argmin(x).to_a
    assert_equal [0, 2], MLX::Core.argmax(x, 1).to_a
    assert_equal [1, 1, 0], MLX::Core.argmin(x, 0).to_a
  end
end
