# frozen_string_literal: true

require_relative "test_helper"

class Phase72TransformsArgnamesTreeTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_grad_supports_keyword_argnames
    loss = lambda do |x, y:|
      z = MLX::Core.add(x, y)
      MLX::Core.sum(MLX::Core.square(z))
    end

    x = MLX::Core.array([1.0, -2.0, 3.0], MLX::Core.float32)
    y = MLX::Core.array([0.5, 1.5, -0.5], MLX::Core.float32)

    grad_fn = MLX::Core.grad(loss, nil, ["y"])
    grads = grad_fn.call(x, y: y)

    assert_nil grads[0]
    assert_includes grads[1].keys.map(&:to_s), "y"
    assert_nested_close [3.0, -1.0, 5.0], grads[1]["y"].to_a
  end

  def test_value_and_grad_preserves_aux_outputs_with_keyword_grads
    fun = lambda do |x, y:|
      z = MLX::Core.add(x, y)
      [MLX::Core.sum(MLX::Core.square(z)), z]
    end

    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    y = MLX::Core.array([3.0, 4.0], MLX::Core.float32)

    vag_fn = MLX::Core.value_and_grad(fun, nil, ["y"])
    value, grads = vag_fn.call(x, y: y)

    assert_equal 2, value.length
    assert_in_delta 52.0, value[0].to_a, 1e-5
    assert_nested_close [4.0, 6.0], value[1].to_a
    assert_nil grads[0]
    assert_nested_close [8.0, 12.0], grads[1]["y"].to_a
  end

  def test_grad_supports_tree_positional_argument
    loss = lambda do |params|
      w2 = MLX::Core.sum(MLX::Core.square(params["w"]))
      b = MLX::Core.sum(params["b"])
      MLX::Core.add(w2, b)
    end

    params = {
      "w" => MLX::Core.array([1.0, -2.0], MLX::Core.float32),
      "b" => MLX::Core.array([3.0], MLX::Core.float32)
    }

    grad_fn = MLX::Core.grad(loss)
    grads = grad_fn.call(params)

    assert_instance_of Hash, grads
    assert_nested_close [2.0, -4.0], grads["w"].to_a
    assert_nested_close [1.0], grads["b"].to_a
  end

  private

  def assert_nested_close(expected, actual, atol = 1e-4)
    assert_equal structure_signature(expected), structure_signature(actual)
    flatten(expected).zip(flatten(actual)).each do |exp, got|
      assert_in_delta exp, got, atol
    end
  end

  def flatten(value)
    return [value] unless value.is_a?(Array)

    value.flat_map { |v| flatten(v) }
  end

  def structure_signature(value)
    return :scalar unless value.is_a?(Array)

    [value.length, *(value.map { |v| structure_signature(v) })]
  end
end
