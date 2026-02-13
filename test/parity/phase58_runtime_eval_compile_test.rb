# frozen_string_literal: true

require_relative "test_helper"

class Phase58RuntimeEvalCompileTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_eval_and_async_eval_accept_arrays_and_trees
    x = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)
    y = MLX::Core.exp(x)

    assert_nil MLX::Core.eval(y)
    assert_nested_close [Math::E, Math::E**2, Math::E**3], y.to_a, 1e-4

    z = MLX::Core.log(y)
    tree = {"v" => [z]}
    assert_nil MLX::Core.async_eval(tree)
    assert_nested_close [1.0, 2.0, 3.0], z.to_a, 1e-4
  end

  def test_compile_toggles_exist_and_are_callable
    assert_respond_to MLX::Core, :disable_compile
    assert_respond_to MLX::Core, :enable_compile

    assert_nil MLX::Core.disable_compile
    assert_nil MLX::Core.enable_compile
  end

  private

  def assert_nested_close(expected, actual, atol)
    expected.zip(actual).each do |exp, got|
      assert_in_delta exp, got, atol
    end
  end
end
