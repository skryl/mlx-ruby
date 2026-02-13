# frozen_string_literal: true

require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase153MultiOptimizerFunctionalParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_multi_optimizer_splits_gradient_tree_using_filters
    left_opt = MLX::Optimizers::Optimizer.new(learning_rate: 0.1)
    right_opt = MLX::Optimizers::Optimizer.new(learning_rate: 0.5)

    filter = lambda do |path, _grad|
      path.start_with?("left")
    end

    opt = MLX::Optimizers::MultiOptimizer.new([left_opt, right_opt], filters: [filter])

    gradients = {
      "left" => MLX::Core.array([1.0], MLX::Core.float32),
      "right" => MLX::Core.array([1.0], MLX::Core.float32)
    }
    parameters = {
      "left" => MLX::Core.array([10.0], MLX::Core.float32),
      "right" => MLX::Core.array([10.0], MLX::Core.float32)
    }

    out = opt.apply_gradients(gradients, parameters)

    assert_nested_close [9.9], out.fetch("left").to_a
    assert_nested_close [9.5], out.fetch("right").to_a
    assert_equal 1, left_opt.step
    assert_equal 1, right_opt.step
  end

  def test_multi_optimizer_filter_count_validation
    err = assert_raises(ArgumentError) do
      MLX::Optimizers::MultiOptimizer.new(
        [MLX::Optimizers::Optimizer.new, MLX::Optimizers::Optimizer.new],
        filters: []
      )
    end
    assert_match(/filters/i, err.message)
  end

  def test_multi_optimizer_state_contract
    left_opt = MLX::Optimizers::Optimizer.new(learning_rate: 0.1)
    right_opt = MLX::Optimizers::Optimizer.new(learning_rate: 0.2)
    opt = MLX::Optimizers::MultiOptimizer.new([left_opt, right_opt], filters: [->(*_) { true }])

    state = opt.state
    assert_equal 2, state.fetch("states").length

    opt.state = {
      "states" => [
        { "step" => 3, "learning_rate" => 0.1 },
        { "step" => 4, "learning_rate" => 0.2 }
      ]
    }

    assert_equal 3, left_opt.step
    assert_equal 4, right_opt.step

    assert_raises(ArgumentError) do
      opt.state = { "states" => [{ "step" => 1 }] }
    end
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
