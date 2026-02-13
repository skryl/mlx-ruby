# frozen_string_literal: true

require_relative "test_helper"

class Phase105CoreClassSurfaceParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_missing_core_classes_are_present
    %i[ArrayAt ArrayIterator ArrayLike Finfo Iinfo].each do |name|
      assert MLX::Core.const_defined?(name), "expected MLX::Core::#{name} to be defined"
    end
  end

  def test_array_at_and_iterator_classes
    x = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)

    at = x.at
    assert_instance_of MLX::Core::ArrayAt, at
    assert_nested_close [1.0, 12.0, 3.0], at[1].add(10.0).to_a

    it = x.__iter__
    assert_instance_of MLX::Core::ArrayIterator, it
    assert_equal 1.0, it.__next__.to_a
    assert_equal 2.0, it.__next__.to_a
    assert_equal 3.0, it.__next__.to_a
    assert_raises(StopIteration) { it.__next__ }
  end

  def test_array_like_and_info_classes
    obj = Object.new
    def obj.__mlx__array__
      MLX::Core.array([1.0], MLX::Core.float32)
    end

    wrapped = MLX::Core::ArrayLike.new(obj)
    assert_same obj, wrapped.object
    assert_equal 1.0, wrapped.to_a.to_a[0]
    assert_raises(TypeError) { MLX::Core::ArrayLike.new(Object.new) }

    finfo = MLX::Core::Finfo.new(MLX::Core.float32)
    assert_equal MLX::Core.float32, finfo.dtype
    assert_operator finfo.max, :>, 0.0
    assert_operator finfo.min, :<, 0.0
    assert_operator finfo.eps, :>, 0.0

    iinfo = MLX::Core::Iinfo.new(MLX::Core.int32)
    assert_equal MLX::Core.int32, iinfo.dtype
    assert_equal(-2_147_483_648, iinfo.min)
    assert_equal(2_147_483_647, iinfo.max)

    assert_instance_of MLX::Core::Finfo, MLX::Core.finfo(MLX::Core.float64)
    assert_instance_of MLX::Core::Iinfo, MLX::Core.iinfo(MLX::Core.uint8)
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
