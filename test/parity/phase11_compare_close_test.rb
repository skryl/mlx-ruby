# frozen_string_literal: true

require_relative "test_helper"

class Phase11CompareCloseTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_less_equal_and_greater_equal_with_scalar_broadcast
    x = MLX::Core.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], MLX::Core.float32)

    le = MLX::Core.less_equal(x, 2.0)
    ge = MLX::Core.greater_equal(x, 2.0)

    assert_equal :bool_, le.dtype.name
    assert_equal [[true, true, false], [false, true, true]], le.to_a
    assert_equal [[false, true, true], [true, true, false]], ge.to_a
  end

  def test_isclose_and_array_equal_handle_nan_controls
    a = MLX::Core.array([1.0, 2.0, MLX::Core.nan], MLX::Core.float32)
    b = MLX::Core.array([1.0 + 1e-6, 2.0 - 1e-6, MLX::Core.nan], MLX::Core.float32)
    c = MLX::Core.array([1.0, 2.0, MLX::Core.nan], MLX::Core.float32)

    assert_equal [true, true, false], MLX::Core.isclose(a, b).to_a
    assert_equal [true, true, true], MLX::Core.isclose(a, b, 1e-5, 1e-8, true).to_a

    refute MLX::Core.array_equal(a, c)
    assert MLX::Core.array_equal(a, c, true)
  end
end
