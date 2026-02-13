# frozen_string_literal: true

require_relative "test_helper"

class Phase12NanInfPredicatesTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_nan_and_inf_predicates_on_vector
    x = MLX::Core.array([-MLX::Core.inf, -1.0, 0.0, MLX::Core.inf, MLX::Core.nan], MLX::Core.float32)

    assert_equal [false, false, false, false, true], MLX::Core.isnan(x).to_a
    assert_equal [true, false, false, true, false], MLX::Core.isinf(x).to_a
    assert_equal [false, false, false, true, false], MLX::Core.isposinf(x).to_a
    assert_equal [true, false, false, false, false], MLX::Core.isneginf(x).to_a
  end

  def test_nan_and_inf_predicates_on_scalar_inputs
    assert_equal true, MLX::Core.isnan(MLX::Core.nan).to_a
    assert_equal true, MLX::Core.isinf(MLX::Core.inf).to_a
    assert_equal false, MLX::Core.isposinf(-MLX::Core.inf).to_a
    assert_equal true, MLX::Core.isneginf(-MLX::Core.inf).to_a
  end
end
