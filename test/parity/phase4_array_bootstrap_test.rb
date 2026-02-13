# frozen_string_literal: true

require_relative "test_helper"

class Phase4ArrayBootstrapTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_scalar_and_vector_construction
    scalar = MLX::Core.array(7)
    assert_instance_of MLX::Core::Array, scalar
    assert_equal 0, scalar.ndim
    assert_equal [], scalar.shape
    assert_equal 1, scalar.size
    assert_equal 7, scalar.item
    assert_equal 7, scalar.to_a

    vector = MLX::Core.array([1, 2, 3], MLX::Core.int32)
    assert_instance_of MLX::Core::Array, vector
    assert_equal 1, vector.ndim
    assert_equal [3], vector.shape
    assert_equal 3, vector.size
    assert_equal :int32, vector.dtype.name
    assert_equal [1, 2, 3], vector.to_a
  end

  def test_basic_array_arithmetic
    a = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)
    b = MLX::Core.array([4.0, 5.0, 6.0], MLX::Core.float32)

    assert_array_close [5.0, 7.0, 9.0], (a + b).to_a
    assert_array_close [4.0, 10.0, 18.0], (a * b).to_a
    assert_array_close [2.0, 3.0, 4.0], (a + 1).to_a
  end

  private

  def assert_array_close(expected, actual, atol = 1e-5)
    assert_equal expected.length, actual.length
    expected.zip(actual).each do |e, a|
      assert_in_delta e, a, atol
    end
  end
end
