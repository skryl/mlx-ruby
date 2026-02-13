# frozen_string_literal: true

require_relative "test_helper"

class Phase60TransformsAutodiffTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_jvp_and_vjp
    fun = ->(x) { MLX::Core.sin(x) }
    x = MLX::Core.array([0.0, 1.0], MLX::Core.float32)
    t = MLX::Core.ones_like(x)

    outs, jvps = MLX::Core.jvp(fun, [x], [t])
    assert_equal 1, outs.length
    assert_equal 1, jvps.length

    expected = MLX::Core.cos(x).to_a
    assert_nested_close expected, jvps[0].to_a

    _, vjps = MLX::Core.vjp(fun, [x], [t])
    assert_equal 1, vjps.length
    assert_nested_close expected, vjps[0].to_a
  end

  def test_grad_and_value_and_grad
    loss = ->(x) { MLX::Core.sum(MLX::Core.square(x)) }
    x = MLX::Core.array([1.5, -2.0, 3.0], MLX::Core.float32)

    grad_fn = MLX::Core.grad(loss)
    g = grad_fn.call(x)
    assert_nested_close [3.0, -4.0, 6.0], g.to_a

    vag_fn = MLX::Core.value_and_grad(loss)
    value, grads = vag_fn.call(x)
    assert_in_delta 15.25, value.to_a, 1e-5
    assert_nested_close [3.0, -4.0, 6.0], grads.to_a
  end

  def test_compile_checkpoint_and_vmap_return_callable
    f = ->(x) { MLX::Core.add(x, 1.0) }
    x = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)

    compiled = MLX::Core.compile(f)
    assert_respond_to compiled, :call
    assert_nested_close [2.0, 3.0, 4.0], compiled.call(x).to_a

    checkpointed = MLX::Core.checkpoint(f)
    assert_respond_to checkpointed, :call
    assert_nested_close [2.0, 3.0, 4.0], checkpointed.call(x).to_a

    square = ->(v) { MLX::Core.square(v) }
    vmapped = MLX::Core.vmap(square)
    batched = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)
    assert_nested_close [[1.0, 4.0], [9.0, 16.0]], vmapped.call(batched).to_a
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
