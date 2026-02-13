# frozen_string_literal: true

require_relative "test_helper"

class Phase79ArrayInstanceSurfaceTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_array_exposes_python_style_instance_methods
    x = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)

    %i[
      add subtract multiply divide exp sin cos mean sum var std max min
      reshape transpose squeeze square sqrt rsqrt reciprocal tolist
      __add__ __sub__ __mul__ __truediv__ __matmul__ __len__ __iter__
    ].each do |name|
      assert_respond_to x, name
    end
  end

  def test_array_instance_method_results_match_core_functions
    x = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)
    y = MLX::Core.array([4.0, 5.0, 6.0], MLX::Core.float32)

    assert_nested_close [5.0, 7.0, 9.0], x.add(y).to_a
    assert_nested_close [-3.0, -3.0, -3.0], x.subtract(y).to_a
    assert_nested_close [4.0, 10.0, 18.0], x.multiply(y).to_a
    assert_nested_close [0.25, 0.4, 0.5], x.divide(y).to_a
    assert_in_delta 2.0, x.mean.to_a, 1e-5
    assert_in_delta 6.0, x.sum.to_a, 1e-5

    m = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)
    assert_nested_close [[1.0, 3.0], [2.0, 4.0]], m.transpose.to_a
    assert_nested_close [[7.0, 10.0], [15.0, 22.0]], m.__matmul__(m).to_a
    assert_equal [1.0, 2.0, 3.0], x.tolist
    assert_equal 3, x.__len__
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
