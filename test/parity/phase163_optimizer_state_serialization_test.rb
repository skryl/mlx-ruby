# frozen_string_literal: true

require_relative "test_helper"

class Phase163OptimizerStateSerializationTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_adam_resume_from_state_matches_continuation
    opt_a = MLX::Optimizers::Adam.new(learning_rate: 0.1, betas: [0.9, 0.999], eps: 1e-8)

    params = { "w" => MLX::Core.array([10.0], MLX::Core.float32) }
    grads = { "w" => MLX::Core.array([2.0], MLX::Core.float32) }

    after_one = opt_a.apply_gradients(grads, params)
    snapshot = deep_clone_tree(opt_a.state)

    opt_b = MLX::Optimizers::Adam.new(learning_rate: 0.1, betas: [0.9, 0.999], eps: 1e-8)
    opt_b.state = snapshot

    cont_a = opt_a.apply_gradients(grads, after_one)
    cont_b = opt_b.apply_gradients(grads, after_one)

    assert_equal opt_a.step, opt_b.step
    assert_nested_close cont_a.fetch("w").to_a, cont_b.fetch("w").to_a, 1e-6
    assert_nested_close opt_a.state.fetch("w").fetch("m").to_a, opt_b.state.fetch("w").fetch("m").to_a, 1e-6
    assert_nested_close opt_a.state.fetch("w").fetch("v").to_a, opt_b.state.fetch("w").fetch("v").to_a, 1e-6
  end

  def test_multi_optimizer_state_roundtrip
    left = MLX::Optimizers::SGD.new(learning_rate: 0.1, momentum: 0.9)
    right = MLX::Optimizers::Adam.new(learning_rate: 0.05)

    filter = ->(path, _g) { path.start_with?("left") }
    multi_a = MLX::Optimizers::MultiOptimizer.new([left, right], filters: [filter])

    grads = {
      "left" => MLX::Core.array([1.0], MLX::Core.float32),
      "right" => MLX::Core.array([2.0], MLX::Core.float32)
    }
    params = {
      "left" => MLX::Core.array([10.0], MLX::Core.float32),
      "right" => MLX::Core.array([10.0], MLX::Core.float32)
    }

    after_one = multi_a.apply_gradients(grads, params)
    snapshot = deep_clone_tree(multi_a.state)

    multi_b = MLX::Optimizers::MultiOptimizer.new(
      [MLX::Optimizers::SGD.new(learning_rate: 0.1, momentum: 0.9), MLX::Optimizers::Adam.new(learning_rate: 0.05)],
      filters: [filter]
    )
    multi_b.state = snapshot

    cont_a = multi_a.apply_gradients(grads, after_one)
    cont_b = multi_b.apply_gradients(grads, after_one)

    assert_nested_close cont_a.fetch("left").to_a, cont_b.fetch("left").to_a, 1e-6
    assert_nested_close cont_a.fetch("right").to_a, cont_b.fetch("right").to_a, 1e-6
  end

  private

  def deep_clone_tree(value)
    if value.is_a?(MLX::Core::Array)
      MLX::Core.array(value.to_a, value.dtype)
    elsif value.is_a?(Hash)
      value.each_with_object({}) { |(k, v), out| out[k] = deep_clone_tree(v) }
    elsif value.is_a?(Array)
      value.map { |v| deep_clone_tree(v) }
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
