# frozen_string_literal: true

require_relative "test_helper"

class Phase144PackageParityTest < Minitest::Test
def setup
  TestSupport.build_native_extension!
  $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
  require "mlx"
end

def teardown
  $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
end

  def test_phase_144_contract
    %i[AllToShardedLinear ShardedToAllLinear QuantizedAllToShardedLinear QuantizedShardedToAllLinear].each { |n| assert MLX::NN.const_defined?(n) }
    %i[sum_gradients shard_inplace shard_linear].each { |n| assert_respond_to MLX::NN, n }
  end
end
