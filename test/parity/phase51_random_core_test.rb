# frozen_string_literal: true

require_relative "test_helper"

class Phase51RandomCoreTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_seed_key_and_split
    MLX::Core.seed(123)
    key = MLX::Core.key(123)
    assert_equal [2], key.shape

    a, b = MLX::Core.random_split(key)
    assert_equal [2], a.shape
    assert_equal [2], b.shape

    many = MLX::Core.random_split(key, 3)
    assert_equal [3, 2], many.shape
  end

  def test_uniform_normal_and_randint
    u = MLX::Core.uniform([2, 3], 0.0, 1.0, MLX::Core.float32)
    assert_equal [2, 3], u.shape

    n = MLX::Core.normal([2, 3], 0.0, 1.0, MLX::Core.float32)
    assert_equal [2, 3], n.shape

    r = MLX::Core.randint(3, 9, [8], MLX::Core.int32)
    assert_equal [8], r.shape
    r.to_a.each do |v|
      assert_operator v, :>=, 3
      assert_operator v, :<, 9
    end
  end
end
