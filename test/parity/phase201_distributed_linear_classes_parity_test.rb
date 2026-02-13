# frozen_string_literal: true

require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase201DistributedLinearClassesParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_all_to_sharded_linear_from_linear_matches_linear_on_single_rank
    group = MLX::Core.init(false, "any")
    linear = MLX::NN::Linear.new(3, 2, bias: true)
    linear.weight = MLX::Core.array([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]], MLX::Core.float32)
    linear.bias = MLX::Core.array([0.5, -1.0], MLX::Core.float32)

    sharded = MLX::NN::AllToShardedLinear.from_linear(linear, group: group)
    x = MLX::Core.array([[1.0, 2.0, 3.0]], MLX::Core.float32)

    ref = linear.call(x)
    out = sharded.call(x)
    assert_nested_close ref.to_a, out.to_a
  end

  def test_sharded_to_all_linear_from_linear_matches_linear_on_single_rank
    group = MLX::Core.init(false, "any")
    linear = MLX::NN::Linear.new(3, 2, bias: true)
    linear.weight = MLX::Core.array([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]], MLX::Core.float32)
    linear.bias = MLX::Core.array([0.5, -1.0], MLX::Core.float32)

    sharded = MLX::NN::ShardedToAllLinear.from_linear(linear, group: group)
    x = MLX::Core.array([[1.0, 2.0, 3.0]], MLX::Core.float32)

    ref = linear.call(x)
    out = sharded.call(x)
    assert_nested_close ref.to_a, out.to_a
  end

  private

  def assert_nested_close(expected, actual, atol = 1e-4)
    expected.flatten.zip(actual.flatten).each do |exp, got|
      assert_in_delta exp, got, atol
    end
  end
end
