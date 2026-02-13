# frozen_string_literal: true

require_relative "test_helper"

class Phase182InitBaseInitializersTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_constant_normal_and_uniform_initializers
    shape_ref = MLX::Core.zeros([2, 2], MLX::Core.float32)

    const_init = MLX::NN.constant(0.5, dtype: MLX::Core.float32)
    out_const = const_init.call(shape_ref)
    assert_nested_close [[0.5, 0.5], [0.5, 0.5]], out_const.to_a

    normal_init = MLX::NN.normal(mean: 1.0, std: 0.0, dtype: MLX::Core.float32)
    out_normal = normal_init.call(shape_ref)
    assert_nested_close [[1.0, 1.0], [1.0, 1.0]], out_normal.to_a

    uniform_init = MLX::NN.uniform(low: -2.0, high: -2.0, dtype: MLX::Core.float32)
    out_uniform = uniform_init.call(shape_ref)
    assert_nested_close [[-2.0, -2.0], [-2.0, -2.0]], out_uniform.to_a
  end

  def test_identity_initializer_validates_square_matrix
    eye_init = MLX::NN.identity(dtype: MLX::Core.float32)
    square = MLX::Core.zeros([3, 3], MLX::Core.float32)
    out = eye_init.call(square)
    assert_nested_close [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], out.to_a

    nonsquare = MLX::Core.zeros([2, 3], MLX::Core.float32)
    assert_raises(ArgumentError) { eye_init.call(nonsquare) }
  end

  private

  def assert_nested_close(expected, actual, atol = 1e-5)
    expected.flatten.zip(actual.flatten).each { |e, a| assert_in_delta e, a, atol }
  end
end
