# frozen_string_literal: true

require_relative "test_helper"

class Phase73CompileCheckpointTreeKwargsTest < Minitest::Test
  def run
    run_without_timeout
  end

  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_compile_handles_tree_inputs_keywords_and_preserves_structure
    fun = lambda do |params, x:, bias: 0.0|
      y = MLX::Core.add(MLX::Core.multiply(params["w"], x), params["b"])
      y = MLX::Core.add(y, bias)
      { "y" => y, "stats" => [MLX::Core.sum(MLX::Core.square(y))] }
    end

    params = {
      "w" => MLX::Core.array([2.0, 3.0], MLX::Core.float32),
      "b" => MLX::Core.array([1.0, -1.0], MLX::Core.float32)
    }
    x = MLX::Core.array([4.0, 5.0], MLX::Core.float32)

    compiled = MLX::Core.compile(fun)
    out1 = compiled.call(params, x: x, bias: 1.0)
    assert_instance_of Hash, out1
    assert_nested_close [10.0, 15.0], out1["y"].to_a
    assert_in_delta 325.0, out1["stats"][0].to_a, 1e-5

    out2 = compiled.call(params, x: x, bias: 2.0)
    assert_nested_close [11.0, 16.0], out2["y"].to_a
    assert_in_delta 377.0, out2["stats"][0].to_a, 1e-5
  end

  def test_checkpoint_handles_tree_inputs_and_keywords
    fun = lambda do |params, x:, scale: 1.0|
      y = MLX::Core.add(MLX::Core.multiply(params["w"], x), params["b"])
      y = MLX::Core.multiply(y, scale)
      { "out" => y, "meta" => { "sum" => MLX::Core.sum(y) } }
    end

    params = {
      "w" => MLX::Core.array([1.0, 2.0], MLX::Core.float32),
      "b" => MLX::Core.array([3.0, 4.0], MLX::Core.float32)
    }
    x = MLX::Core.array([5.0, 6.0], MLX::Core.float32)

    checkpointed = MLX::Core.checkpoint(fun)
    out = checkpointed.call(params, x: x, scale: 0.5)

    assert_instance_of Hash, out
    assert_nested_close [4.0, 8.0], out["out"].to_a
    assert_in_delta 12.0, out["meta"]["sum"].to_a, 1e-5
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
