# frozen_string_literal: true

require_relative "test_helper"

class Phase94GradArgValidationTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_argnums_and_argnames_validation
    loss = ->(x) { MLX::Core.sum(MLX::Core.square(x)) }

    assert_raises(ArgumentError) { MLX::Core.grad(loss, [], []) }
    assert_raises(ArgumentError) { MLX::Core.grad(loss, [-1]) }
    assert_raises(ArgumentError) { MLX::Core.grad(loss, [0, 0]) }
    assert_raises(ArgumentError) { MLX::Core.grad(loss, nil, %w[y y]) }
    assert_raises(TypeError) { MLX::Core.grad(loss, "0") }
    assert_raises(TypeError) { MLX::Core.grad(loss, nil, 1.2) }
  end

  def test_keyword_target_must_be_present_at_call_time
    loss = lambda do |x, y:|
      MLX::Core.sum(MLX::Core.square(MLX::Core.add(x, y)))
    end

    grad_fn = MLX::Core.grad(loss, nil, "y")
    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    err = assert_raises(ArgumentError) { grad_fn.call(x) }
    assert_match(/keyword argument 'y'/i, err.message)
  end
end
