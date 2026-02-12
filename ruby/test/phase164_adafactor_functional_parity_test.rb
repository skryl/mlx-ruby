# frozen_string_literal: true

require_relative "test_helper"

class Phase164AdafactorFunctionalParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_adafactor_unfactored_state_and_first_step
    opt = MLX::Optimizers::Adafactor.new(
      learning_rate: 0.1,
      relative_step: false,
      scale_parameter: false
    )

    params = { "w" => MLX::Core.array([10.0, 10.0], MLX::Core.float32) }
    grads = { "w" => MLX::Core.array([2.0, 2.0], MLX::Core.float32) }

    out = opt.apply_gradients(grads, params)

    state = opt.state.fetch("w")
    assert state.key?("exp_avg_sq")
    refute state.key?("exp_avg_sq_row")
    refute state.key?("exp_avg_sq_col")
    assert_nested_close [9.9, 9.9], out.fetch("w").to_a, 1e-4
  end

  def test_adafactor_factored_state_and_first_step
    opt = MLX::Optimizers::Adafactor.new(
      learning_rate: 0.1,
      relative_step: false,
      scale_parameter: false
    )

    params = {
      "w" => MLX::Core.array([[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]], MLX::Core.float32)
    }
    grads = {
      "w" => MLX::Core.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], MLX::Core.float32)
    }

    out = opt.apply_gradients(grads, params)

    state = opt.state.fetch("w")
    assert state.key?("exp_avg_sq_row")
    assert state.key?("exp_avg_sq_col")
    refute state.key?("exp_avg_sq")

    expected = [[9.9, 9.9, 9.9], [9.9, 9.9, 9.9]]
    assert_nested_close expected, out.fetch("w").to_a, 1e-4
  end

  def test_adafactor_beta1_tracks_first_moment
    opt = MLX::Optimizers::Adafactor.new(
      learning_rate: 0.1,
      relative_step: false,
      scale_parameter: false,
      beta_1: 0.9
    )

    params = { "w" => MLX::Core.array([10.0], MLX::Core.float32) }
    grads = { "w" => MLX::Core.array([2.0], MLX::Core.float32) }

    opt.apply_gradients(grads, params)

    state = opt.state.fetch("w")
    assert state.key?("exp_avg")
    assert_nested_close [0.01], state.fetch("exp_avg").to_a, 1e-6
  end

  private

  def assert_nested_close(expected, actual, atol = 1e-5)
    assert_equal shape_signature(expected), shape_signature(actual)
    flatten(expected).zip(flatten(actual)).each do |exp, got|
      assert_in_delta exp, got, atol
    end
  end

  def flatten(value)
    return [value] unless value.is_a?(Array)

    value.flat_map { |item| flatten(item) }
  end

  def shape_signature(value)
    return :scalar unless value.is_a?(Array)

    [value.length, *(value.map { |item| shape_signature(item) })]
  end
end
