# frozen_string_literal: true

require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase200DistributedHelpersParityTest < Minitest::Test
  class TinyModule < MLX::NN::Module
    def initialize
      super()
      self.weight = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)
      self.bias = MLX::Core.array([0.5, -0.5], MLX::Core.float32)
    end
  end

  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_sum_gradients_is_identity_for_single_rank
    group = MLX::Core.init(false, "any")
    fn = MLX::NN.sum_gradients(group)
    x = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)

    out = fn.call(x)
    assert_equal x.to_a, out.to_a
  end

  def test_shard_inplace_validates_sharding_string
    mod = TinyModule.new
    assert_raises(ArgumentError) { MLX::NN.shard_inplace(mod, "bad-sharding") }
  end

  def test_shard_inplace_noop_on_single_rank_group
    group = MLX::Core.init(false, "any")
    mod = TinyModule.new
    before = mod.parameters

    MLX::NN.shard_inplace(mod, "all-to-sharded", group: group)
    after = mod.parameters

    assert_equal before["weight"].to_a, after["weight"].to_a
    assert_equal before["bias"].to_a, after["bias"].to_a
  end
end
