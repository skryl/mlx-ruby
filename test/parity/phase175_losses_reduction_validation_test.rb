# frozen_string_literal: true

require_relative "test_helper"

class Phase175LossesReductionValidationTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_reduction_modes_and_invalid_reduction
    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)

    assert_nested_close [1.0, 2.0], MLX::NN::Losses.reduction(x, "none").to_a
    assert_in_delta 1.5, MLX::NN::Losses.reduction(x, "mean").item, 1e-6
    assert_in_delta 3.0, MLX::NN::Losses.reduction(x, "sum").item, 1e-6

    assert_raises(ArgumentError) do
      MLX::NN::Losses.reduction(x, "invalid")
    end
  end

  def test_l1_and_mse_shape_validation
    a = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    b = MLX::Core.array([1.0], MLX::Core.float32)

    assert_raises(ArgumentError) { MLX::NN.l1_loss(a, b) }
    assert_raises(ArgumentError) { MLX::NN.mse_loss(a, b) }
  end

  private

  def assert_nested_close(expected, actual, atol = 1e-5)
    assert_equal expected.length, actual.length
    expected.zip(actual).each { |e, a| assert_in_delta e, a, atol }
  end
end
