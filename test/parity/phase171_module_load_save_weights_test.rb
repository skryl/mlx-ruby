# frozen_string_literal: true

require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase171ModuleLoadSaveWeightsTest < Minitest::Test
  class WeightsModule < MLX::NN::Module
    def initialize
      super()
      self.weight = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)
      self.bias = MLX::Core.array([0.5, -0.5], MLX::Core.float32)
    end
  end

  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_load_weights_from_list_with_strict_validation
    mod = WeightsModule.new

    new_weight = MLX::Core.array([[10.0, 20.0], [30.0, 40.0]], MLX::Core.float32)
    new_bias = MLX::Core.array([1.0, 2.0], MLX::Core.float32)

    mod.load_weights([["weight", new_weight], ["bias", new_bias]], strict: true)
    assert_nested_close [[10.0, 20.0], [30.0, 40.0]], mod.weight.to_a
    assert_nested_close [1.0, 2.0], mod.bias.to_a

    assert_raises(ArgumentError) do
      mod.load_weights([["weight", new_weight]], strict: true)
    end

    assert_raises(ArgumentError) do
      mod.load_weights([["weight", "bad"], ["bias", new_bias]], strict: true)
    end
  end

  def test_save_and_load_npz_weights_roundtrip
    mod = WeightsModule.new

    TestSupport.mktmpdir("mlx-ruby-weights") do |dir|
      path = File.join(dir, "weights.npz")
      mod.save_weights(path)

      other = WeightsModule.new
      other.weight = MLX::Core.array([[0.0, 0.0], [0.0, 0.0]], MLX::Core.float32)
      other.bias = MLX::Core.array([0.0, 0.0], MLX::Core.float32)
      other.load_weights(path, strict: true)

      assert_nested_close mod.weight.to_a, other.weight.to_a
      assert_nested_close mod.bias.to_a, other.bias.to_a
    end
  end

  def test_save_weights_rejects_unsupported_extension
    mod = WeightsModule.new

    assert_raises(ArgumentError) do
      mod.save_weights("weights.bin")
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
