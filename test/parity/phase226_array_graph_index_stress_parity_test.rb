# frozen_string_literal: true

require_relative "test_helper"

class Phase226ArrayGraphIndexStressParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_deep_graph_chain_evaluates
    x = MLX::Core.array([1.0], MLX::Core.float32)
    y = x
    256.times { y = MLX::Core.add(y, 1.0) }
    MLX::Core.eval(y)
    assert_equal [257.0], y.to_a
  end

  def test_siblings_can_be_evaluated_independently
    x = MLX::Core.array([2.0, 3.0], MLX::Core.float32)
    a = MLX::Core.multiply(x, 2.0)
    b = MLX::Core.add(x, 5.0)

    MLX::Core.eval(b)
    assert_equal [7.0, 8.0], b.to_a

    MLX::Core.eval(a)
    assert_equal [4.0, 6.0], a.to_a
  end

  def test_large_index_access
    x = MLX::Core.arange(0, 100_000, 1, MLX::Core.int32)
    tail = x.__getitem__(99_999)
    assert_equal 99_999, tail.to_a
  end
end
