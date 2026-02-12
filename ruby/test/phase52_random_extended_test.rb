# frozen_string_literal: true

require_relative "test_helper"

class Phase52RandomExtendedTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_bernoulli_and_truncated_normal
    b = MLX::Core.bernoulli(0.25, [32])
    assert_equal [32], b.shape
    b.to_a.each do |v|
      assert_includes [0, 1, true, false], v
    end

    t = MLX::Core.truncated_normal(-1.0, 1.0, [64], MLX::Core.float32)
    assert_equal [64], t.shape
    t.to_a.each do |v|
      assert_operator v, :>=, -1.0
      assert_operator v, :<=, 1.0
    end
  end

  def test_gumbel_categorical_laplace_and_permutation
    g = MLX::Core.gumbel([4, 5], MLX::Core.float32)
    assert_equal [4, 5], g.shape

    logits = MLX::Core.array([0.1, 0.2, 0.3, 0.4], MLX::Core.float32)
    c = MLX::Core.categorical(logits, -1, 10)
    assert_equal [10], c.shape

    l = MLX::Core.laplace([6], 0.0, 1.0, MLX::Core.float32)
    assert_equal [6], l.shape

    p = MLX::Core.permutation(10)
    assert_equal [10], p.shape
    assert_equal((0..9).to_a.sort, p.to_a.sort)
  end

  def test_multivariate_normal
    mean = MLX::Core.array([0.0, 0.0], MLX::Core.float32)
    cov = MLX::Core.array([[1.0, 0.0], [0.0, 1.0]], MLX::Core.float32)

    out = MLX::Core.multivariate_normal(mean, cov, [5], MLX::Core.float32)
    assert_equal [5, 2], out.shape
  end
end
