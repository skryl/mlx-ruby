# frozen_string_literal: true

require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase202QuantizedDistributedClassesParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    MLX::Core.random_seed(7)
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_quantized_distributed_from_quantized_linear_single_rank
    group = MLX::Core.init(false, "any")
    linear = MLX::NN::Linear.new(32, 4, bias: true)
    qlinear = linear.to_quantized(group_size: 32, bits: 4, mode: "affine")

    a2s = MLX::NN::QuantizedAllToShardedLinear.from_quantized_linear(qlinear, group: group)
    s2a = MLX::NN::QuantizedShardedToAllLinear.from_quantized_linear(qlinear, group: group)

    x = MLX::Core.ones([2, 32], MLX::Core.float32)
    ref = qlinear.call(x)
    out_a2s = a2s.call(x)
    out_s2a = s2a.call(x)

    assert_nested_close ref.to_a, out_a2s.to_a, 1e-3
    assert_nested_close ref.to_a, out_s2a.to_a, 1e-3
  end

  def test_shard_linear_dispatches_quantized_wrappers
    group = MLX::Core.init(false, "any")
    linear = MLX::NN::Linear.new(32, 4, bias: true)
    qlinear = linear.to_quantized(group_size: 32, bits: 4, mode: "affine")

    a2s = MLX::NN.shard_linear(qlinear, "all-to-sharded", group: group)
    s2a = MLX::NN.shard_linear(qlinear, "sharded-to-all", group: group)

    assert_instance_of MLX::NN::QuantizedAllToShardedLinear, a2s
    assert_instance_of MLX::NN::QuantizedShardedToAllLinear, s2a
  end

  private

  def assert_nested_close(expected, actual, atol = 1e-4)
    expected.flatten.zip(actual.flatten).each do |exp, got|
      assert_in_delta exp, got, atol
    end
  end
end
