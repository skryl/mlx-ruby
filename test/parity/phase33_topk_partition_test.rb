# frozen_string_literal: true

require_relative "test_helper"

class Phase33TopkPartitionTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_topk_on_flattened_array
    x = MLX::Core.array([3, 1, 4, 2], MLX::Core.int32)
    assert_equal [3, 4], MLX::Core.topk(x, 2).to_a
  end

  def test_partition_and_argpartition_invariants
    raw = [3, 1, 4, 2]
    x = MLX::Core.array(raw, MLX::Core.int32)
    kth = 2

    p = MLX::Core.partition(x, kth).to_a
    assert_equal raw.sort[kth], p[kth]
    assert p[0...kth].all? { |v| v <= p[kth] }
    assert p[(kth + 1)..].all? { |v| v >= p[kth] }

    idx = MLX::Core.argpartition(x, kth).to_a
    values = idx.map { |i| raw[i] }
    assert_equal raw.sort[kth], values[kth]
    assert values[0...kth].all? { |v| v <= values[kth] }
    assert values[(kth + 1)..].all? { |v| v >= values[kth] }
  end
end
