# frozen_string_literal: true

require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase168ModuleParametersFreezeTest < Minitest::Test
  class TinyModule < MLX::NN::Module
    def initialize
      super()
      self.weight = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
      self.bias = MLX::Core.array([0.5], MLX::Core.float32)
      self.child = MLX::NN::Module.new
      child.scale = MLX::Core.array([3.0], MLX::Core.float32)
      self.name = "tiny"
    end
  end

  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_parameters_include_nested_arrays_and_exclude_plain_attrs
    mod = TinyModule.new

    params = mod.parameters

    assert params.key?("weight")
    assert params.key?("bias")
    assert params.key?("child")
    refute params.key?("name")
    assert_nested_close [3.0], params.fetch("child").fetch("scale").to_a
  end

  def test_freeze_unfreeze_keys_and_strict_validation
    mod = TinyModule.new

    assert_raises(KeyError) do
      mod.freeze(recurse: false, keys: "missing", strict: true)
    end

    mod.freeze(recurse: false, keys: "bias", strict: true)
    trainable = mod.trainable_parameters
    assert trainable.key?("weight")
    refute trainable.key?("bias")

    mod.unfreeze(recurse: false, keys: "bias", strict: true)
    trainable_after = mod.trainable_parameters
    assert trainable_after.key?("bias")
  end

  def test_recursive_freeze_affects_child_modules
    mod = TinyModule.new

    mod.freeze(recurse: true)
    trainable = mod.trainable_parameters

    refute trainable.fetch("child").key?("scale")
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
