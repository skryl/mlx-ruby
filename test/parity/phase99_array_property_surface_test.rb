# frozen_string_literal: true

require_relative "test_helper"

class Phase99ArrayPropertySurfaceTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_remaining_python_style_array_properties_are_present
    x = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)

    %i[T at real imag itemsize nbytes].each do |name|
      assert_respond_to x, name
    end
  end

  def test_property_wrappers_match_expected_behavior
    x = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)
    y = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)

    assert_nested_close [[1.0, 3.0], [2.0, 4.0]], x.T.to_a
    assert_equal x.dtype.size, x.itemsize
    assert_equal x.size * x.itemsize, x.nbytes
    assert_nested_close [[1.0, 2.0], [3.0, 4.0]], x.real.to_a
    assert_nested_close [[0.0, 0.0], [0.0, 0.0]], x.imag.to_a
    assert_nested_close [1.0, 12.0, 3.0], y.at[1].add(10.0).to_a
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
