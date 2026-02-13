# frozen_string_literal: true

require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase170ModuleUpdateStrictnessTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def build_module
    mod = MLX::NN::Module.new
    mod.w = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    mod.child = MLX::NN::Module.new
    mod.child.b = MLX::Core.array([0.5], MLX::Core.float32)
    mod.items = [MLX::NN::Module.new]
    mod.items[0].v = MLX::Core.array([1.5], MLX::Core.float32)
    mod
  end

  def test_update_strict_nested_and_type_validation
    mod = build_module

    mod.update({ "w" => MLX::Core.array([3.0, 4.0], MLX::Core.float32) }, strict: true)
    assert_nested_close [3.0, 4.0], mod.w.to_a

    mod.update({ "child" => { "b" => MLX::Core.array([1.0], MLX::Core.float32) } }, strict: true)
    assert_nested_close [1.0], mod.child.b.to_a

    assert_raises(ArgumentError) do
      mod.update({ "missing" => MLX::Core.array([1.0], MLX::Core.float32) }, strict: true)
    end

    assert_raises(ArgumentError) do
      mod.update({ "w" => "not an array" }, strict: true)
    end
  end

  def test_update_modules_strict_validation_and_replacement
    mod = build_module

    replacement = MLX::NN::Module.new
    replacement.b = MLX::Core.array([9.0], MLX::Core.float32)
    mod.update_modules({ "child" => replacement }, strict: true)
    assert_same replacement, mod.child

    assert_raises(ArgumentError) do
      mod.update_modules({ "child" => "not module" }, strict: true)
    end

    assert_raises(ArgumentError) do
      mod.update_modules({ "missing" => MLX::NN::Module.new }, strict: true)
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
