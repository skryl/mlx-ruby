# frozen_string_literal: true

require_relative "test_helper"

class Phase55LinalgDecompositionsTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
    @previous_device = MLX::Core.default_device
    MLX::Core.set_default_device(MLX::Core.cpu)
  end

  def teardown
    MLX::Core.set_default_device(@previous_device) if defined?(@previous_device) && @previous_device
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_norm_variants
    x = MLX::Core.array([3.0, 4.0], MLX::Core.float32)
    n2 = MLX::Core.norm(x)
    assert_in_delta 5.0, n2.to_a, 1e-5

    m = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)
    fro = MLX::Core.norm(m, "fro")
    assert_in_delta Math.sqrt(30.0), fro.to_a, 1e-5

    axis = MLX::Core.norm(m, nil, 1, true)
    assert_equal [2, 1], axis.shape
  end

  def test_qr_and_svd
    a = MLX::Core.array([[2.0, 3.0], [1.0, 2.0]], MLX::Core.float32)

    q, r = MLX::Core.qr(a)
    assert_equal [2, 2], q.shape
    assert_equal [2, 2], r.shape

    reconstructed = MLX::Core.matmul(q, r)
    assert_nested_close a.to_a, reconstructed.to_a

    u, s, vt = MLX::Core.svd(a)
    assert_equal [2, 2], u.shape
    assert_equal [2], s.shape
    assert_equal [2, 2], vt.shape

    only_s = MLX::Core.svd(a, false)
    assert_equal [2], only_s.shape
  end

  private

  def assert_nested_close(expected, actual, atol = 1e-3)
    assert_equal structure_signature(expected), structure_signature(actual)
    flatten(expected).zip(flatten(actual)).each do |exp, got|
      assert_in_delta exp, got, atol
    end
  end

  def flatten(value)
    return [value] unless value.is_a?(Array)

    value.flat_map { |v| flatten(v) }
  end

  def structure_signature(value)
    return :scalar unless value.is_a?(Array)

    [value.length, *(value.map { |v| structure_signature(v) })]
  end
end
