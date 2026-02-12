# frozen_string_literal: true

require_relative "test_helper"

class Phase150OptimizerSchedulerIntegrationTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_learning_rate_schedule_uses_pre_increment_step
    schedule = MLX::Optimizers.linear_schedule(0.0, 1.0, 10)
    opt = MLX::Optimizers::Optimizer.new(learning_rate: schedule)

    params = { "w" => MLX::Core.array([10.0], MLX::Core.float32) }
    grads = { "w" => MLX::Core.array([1.0], MLX::Core.float32) }

    params = opt.apply_gradients(grads, params)
    assert_equal 1, opt.step
    assert_in_delta 0.0, scalar(opt.learning_rate), 1e-8
    assert_nested_close [10.0], params["w"].to_a

    params = opt.apply_gradients(grads, params)
    assert_equal 2, opt.step
    assert_in_delta 0.1, scalar(opt.learning_rate), 1e-8
    assert_nested_close [9.9], params["w"].to_a
  end

  def test_constant_learning_rate_is_in_optimizer_state
    opt = MLX::Optimizers::Optimizer.new(learning_rate: 0.25)

    assert_kind_of Hash, opt.state
    assert_in_delta 0.25, scalar(opt.learning_rate), 1e-8
    assert_equal 0, opt.step

    params = { "w" => MLX::Core.array([4.0], MLX::Core.float32) }
    grads = { "w" => MLX::Core.array([2.0], MLX::Core.float32) }

    out = opt.apply_gradients(grads, params)
    assert_equal 1, opt.step
    assert_nested_close [3.5], out["w"].to_a
  end

  private

  def scalar(value)
    if value.respond_to?(:item)
      value.item
    elsif value.is_a?(MLX::Core::Array)
      value.to_a
    else
      value
    end
  end

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
