# frozen_string_literal: true

require_relative "test_helper"

class Phase98ArrayRemainingSurfaceTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_remaining_python_style_array_methods_are_present
    x = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)

    %i[
      __div__ __dlpack_device__ __ge__ __gt__ __imatmul__ __init__ __le__ __lt__
      __next__ __rdiv__ __rfloordiv__ abs all any argmax argmin astype conj
      cummax cummin cumprod cumsum diag diagonal eps flatten log log10 log1p
      log2 logcumsumexp logsumexp maximum minimum moveaxis prod round split
      swapaxes view
    ].each do |name|
      assert_respond_to x, name
    end
  end

  def test_remaining_wrappers_delegate_to_core
    x = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)
    y = MLX::Core.array([2.0, 2.0, 2.0], MLX::Core.float32)

    assert_nested_close [0.5, 1.0, 1.5], x.__div__(y).to_a
    assert_nested_close [2.0, 1.0, (2.0 / 3.0)], x.__rdiv__(2.0).to_a
    assert_nested_close [2.0, 1.0, 0.0], x.__rfloordiv__(2.0).to_a
    assert_nested_close [1.0, 2.0, 3.0], x.abs.to_a
    assert_equal [3], x.__init__.shape
    assert_operator x.eps, :>, 0.0
  end

  private

  def assert_nested_close(expected, actual, atol = 1e-5)
    assert_equal signature(expected), signature(actual)
    flatten(expected).zip(flatten(actual)).each do |exp, got|
      assert_in_delta exp, got, atol
    end
  end

  def flatten(value)
    return [value] unless value.is_a?(Array)

    value.flat_map { |v| flatten(v) }
  end

  def signature(value)
    return :scalar unless value.is_a?(Array)

    [value.length, *(value.map { |v| signature(v) })]
  end
end
